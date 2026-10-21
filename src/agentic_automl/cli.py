"""`agentic-automl` entry point: trial lifecycle, Claude Code launcher, sub-tool dispatch.

The launcher owns workspace setup. It runs `claude` as a child process (not
exec, so the interactive TTY survives `uv run`) with cwd set to the trial and
TRIAL_DIR / UV_PROJECT exported. Arguments after `--` (and unknown flags on
new/resume/fork) are passed through to Claude Code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agentic_automl import __version__, artifacts
from agentic_automl import state as st
from agentic_automl.contracts import (
    ARTIFACT_REGISTRY,
    PACKAGE_DIR,
    claude_template_dir,
    source_root,
    step_templates_dir,
)

SUBTOOLS = {
    "artifacts": "agentic_automl.artifacts",
    "session-analysis": "agentic_automl.session_analysis",
    "bash-permissions": "agentic_automl.bash_permissions",
    "summarize-trials": "agentic_automl.trial_summary",
    "feedback-episodes": "agentic_automl.feedback_episodes",
    "results": "agentic_automl.results",
}

TRUST_FILE = "work/trust.md"
SESSION_BRIEF = "work/session_brief.md"
RUNTIME_IGNORE = shutil.ignore_patterns("settings.local.json", "__pycache__", "*.pyc")
TEMPLATES_IGNORE = shutil.ignore_patterns("dev", "__pycache__", "*.pyc")  # scripts/dev never reaches a trial
PACKAGE_NAME = "agentic-automl"


class CliError(Exception):
    pass


# ---------------------------------------------------------------------------
# Trial workspace
# ---------------------------------------------------------------------------


def sync_runtime_files(trial: Path) -> None:
    """Copy the runtime .claude tree into the trial (settings.local.json is trial-owned), plus the
    step templates from scripts/ as .claude/scripts/."""
    shutil.copytree(claude_template_dir(), trial / ".claude", dirs_exist_ok=True, ignore=RUNTIME_IGNORE)
    shutil.copytree(step_templates_dir(), trial / ".claude" / "scripts", dirs_exist_ok=True, ignore=TEMPLATES_IGNORE)
    materialize_settings(trial)


def materialize_settings(trial: Path) -> None:
    """Merge hooks.json into the trial's settings.json (Claude Code only reads hooks from
    settings files) and point the runtime-python rules at the configured interpreter."""
    claude_dir = trial / ".claude"
    settings_file = claude_dir / "settings.json"
    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    hooks_file = claude_dir / "hooks.json"
    if hooks_file.is_file():
        merged = settings.setdefault("hooks", {})
        for event, entries in json.loads(hooks_file.read_text(encoding="utf-8"))["hooks"].items():
            merged[event] = entries + [e for e in merged.get(event, []) if e not in entries]
    py = st.runtime_python()
    if py != st.DEFAULT_RUNTIME_PYTHON:
        settings["permissions"]["allow"] = [
            rule.replace(st.DEFAULT_RUNTIME_PYTHON, py) for rule in settings["permissions"]["allow"]
        ]
        runtime_md = claude_dir / "CLAUDE.md"
        runtime_md.write_text(runtime_md.read_text(encoding="utf-8").replace(st.DEFAULT_RUNTIME_PYTHON, py),
                              encoding="utf-8")
    settings_file.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def trust_requested(flag: bool) -> bool:
    return flag or os.environ.get("AGENTIC_AUTOML_TRUST") == "1"


def trust_allows() -> list[str]:
    py = st.runtime_python()
    return [
        f"Bash({py} *)",
        f"Bash(cd * && {py} *)",
        f"Bash(PYTHONPATH=* {py} *)",
        f"Bash(timeout * {py} *)",
        f"Bash(nohup {py} *)",
        "Bash(gsutil cp *)",
        "Bash(gsutil -m cp *)",
        "Bash(gsutil mv *)",
        "Bash(gsutil rm *)",
        "Bash(gsutil -m rm *)",
        "Bash(gsutil rsync *)",
        "Bash(gsutil -m rsync *)",
        "Bash(gsutil mb *)",
    ]


def apply_trust(trial: Path) -> None:
    """Sticky per trial: work/trust.md marks the trial; allows go in .claude/settings.local.json."""
    marker = trial / TRUST_FILE
    if not marker.is_file():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "# Trust mode\n\n"
            f"Enabled {artifacts.now_iso()} via `--trust`. Broader auto-approval for the runtime\n"
            "Python and mutating gsutil commands. This is NOT --dangerously-skip-permissions:\n"
            "artifact paths stay write-guarded and everything else still prompts.\n"
            "Delete this file and .claude/settings.local.json to turn it off.\n",
            encoding="utf-8",
        )
    local = trial / ".claude" / "settings.local.json"
    settings: dict[str, Any] = {}
    if local.is_file():
        try:
            settings = json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            settings = {}
    allow = settings.setdefault("permissions", {}).setdefault("allow", [])
    allow.extend(rule for rule in trust_allows() if rule not in allow)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def prepare_trial(trial: Path, trust: bool) -> None:
    sync_runtime_files(trial)
    if trust_requested(trust) or (trial / TRUST_FILE).is_file():
        apply_trust(trial)


def apply_slug(trial: Path) -> Path:
    new_trial = artifacts.apply_pending_slug(trial)
    if new_trial != trial:
        st.migrate_transcripts(trial, new_trial)
    return new_trial


def write_session_brief(trial: Path, include_transcript: bool) -> Path:
    state = artifacts.load_state(trial)
    snap = artifacts.snapshot(trial, state)
    step = artifacts.next_step(trial)
    lines = [
        f"# Session brief: {state['trial_id']}",
        "",
        f"Generated {artifacts.now_iso()} by the launcher for a fresh chat on an existing trial.",
        "The validator state below is authoritative; re-check with `agentic-automl artifacts status`.",
        "",
        f"- Goal: {state.get('goal') or '(not recorded)'}",
        f"- Dataset: {state.get('dataset_path') or '(not recorded)'}",
    ]
    if state.get("forked_from"):
        lines.append(f"- Forked from: {state['forked_from']}")
    lines += ["", "## Artifacts", ""]
    for key, present in snap["artifacts"].items():
        lines.append(f"- [{'x' if present else ' '}] {key} ({ARTIFACT_REGISTRY[key]['path']})")
    lines += ["", "## Next step", ""]
    if step["canonical_workflow_complete"]:
        optional = ", ".join(step.get("optional_next_skills") or []) or "none"
        lines.append(f"Canonical workflow complete. Optional: {optional}.")
    else:
        lines.append(f"`{step['next_skill']}`: {step['reason']}")
    failures = [e for e in state["validation_log"] if not e["ok"]][-5:]
    if failures:
        lines += ["", "## Recent validator rejections", ""]
        for entry in failures:
            detail = "; ".join(entry.get("errors") or [])[:300]
            lines.append(f"- {entry['at']} {entry['action']} {entry.get('key') or entry.get('skill') or ''}: {detail}")
    if state.get("developer_focus"):
        lines += ["", "## Developer focus", ""] + [f"- {f['text']}" for f in state["developer_focus"]]
    if include_transcript:
        from agentic_automl.session_analysis import extract_text

        text = extract_text(trial)
        lines += ["", "## Prior conversation (tail)", "", text or "(no transcript found)"]
    path = trial / SESSION_BRIEF
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Claude launcher
# ---------------------------------------------------------------------------


def claude_executable() -> str:
    exe = os.environ.get("AGENTIC_AUTOML_CLAUDE") or shutil.which("claude")
    if not exe:
        raise CliError("Claude Code (`claude`) not found on PATH. Install it, or set AGENTIC_AUTOML_CLAUDE.")
    return exe


def build_claude_command(
    *,
    resume_session: str | None = None,
    continue_last: bool = False,
    fork_session: bool = False,
    model: str | None = None,
    dev: bool = False,
    extra: list[str] | None = None,
    prompt: str | None = None,
) -> list[str]:
    cmd = [claude_executable()]
    if resume_session:
        cmd += ["--resume", resume_session]
        if fork_session:
            cmd.append("--fork-session")
    elif continue_last:
        cmd.append("--continue")
    if model:
        cmd += ["--model", model]
    if dev:
        cmd += ["--add-dir", str(source_root())]
    cmd += extra or []
    if prompt:
        cmd.append(prompt)
    return cmd


def launch_claude(trial: Path, cmd: list[str], *, mode: str, dev: bool = False) -> int:
    env = os.environ.copy()
    env["TRIAL_DIR"] = str(trial)
    env["UV_PROJECT"] = str(source_root() or PACKAGE_DIR)
    # Hooks shell out to `agentic-automl`; make sure the one running this launcher is found first.
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    if dev:
        env["AGENTIC_AUTOML_DEV"] = "1"
    artifacts.record_session(trial, mode)
    started = time.time()
    # Claude Code owns Ctrl-C while it runs; don't let it kill the launcher too.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        code = subprocess.call(cmd, cwd=trial, env=env)
    finally:
        signal.signal(signal.SIGINT, previous)
    session_id = st.latest_session_id(trial, since=started - 1)
    if session_id:
        artifacts.attach_session_id(trial, session_id)
    update_results(trial)
    return code


def update_results(trial: Path) -> None:
    """Refresh the durable results ledger; never let a ledger problem break the launcher."""
    from agentic_automl import results

    try:
        results.record_trial(trial)
    except Exception as exc:  # noqa: BLE001
        print(f"agentic-automl: results ledger not updated ({exc})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _emit(data: Any) -> None:
    print(json.dumps(data, indent=2))


def _require_trial(ref: str | None) -> Path:
    trial = st.find_trial(ref)
    if trial is None:
        raise CliError(f"Trial not found: {ref}" if ref else "No trials yet. Start one with: agentic-automl new")
    return trial


def _check_dev(args: argparse.Namespace) -> bool:
    dev = getattr(args, "dev", False)
    if dev and source_root() is None:
        raise CliError("--dev needs a source checkout (run via run.py or `pip install -e .`)")
    return dev


def kickoff_prompt(goal: str | None, dataset: str | None) -> str | None:
    if not goal and not dataset:
        return None
    return (
        "New Agentic AutoML trial. "
        f"Goal: {goal or '(ask me)'}. Dataset: {dataset or '(ask me)'}. "
        "Follow the intake in CLAUDE.md."
    )


def cmd_new(args: argparse.Namespace) -> int:
    dev = _check_dev(args)
    trial = st.unique_trial_dir(st.new_trial_id())
    artifacts.ensure_state(trial, goal=args.goal, dataset_path=args.dataset)
    prepare_trial(trial, args.trust)
    update_results(trial)
    if args.no_launch:
        _emit({"ok": True, "trial_dir": str(trial)})
        return 0
    print(f"Trial: {trial}", file=sys.stderr)
    cmd = build_claude_command(model=args.model, dev=dev, extra=args.claude_args,
                               prompt=kickoff_prompt(args.goal, args.dataset))
    return launch_claude(trial, cmd, mode="new", dev=dev)


def cmd_resume(args: argparse.Namespace) -> int:
    dev = _check_dev(args)
    trial = apply_slug(_require_trial(args.trial))
    prepare_trial(trial, args.trust)
    state = artifacts.load_state(trial)
    fresh = args.fresh or args.summarize
    session = state.get("last_claude_session_id")
    has_transcript = bool(st.transcripts(trial))
    if not fresh and not has_transcript:
        fresh = True  # nothing to continue (e.g. transcripts pruned): start fresh with a brief
    if fresh:
        write_session_brief(trial, include_transcript=args.summarize)
        cmd = build_claude_command(
            model=args.model, dev=dev, extra=args.claude_args,
            prompt=f"Fresh chat on an existing trial. Read {SESSION_BRIEF}, then run "
                   "`agentic-automl artifacts next-step` and continue.",
        )
        mode = "fresh"
    else:
        known = session and any(p.stem == session for p in st.transcripts(trial))
        cmd = build_claude_command(resume_session=session if known else None, continue_last=not known,
                                   model=args.model, dev=dev, extra=args.claude_args)
        mode = "resume"
    if args.no_launch:
        _emit({"ok": True, "trial_dir": str(trial), "mode": mode, "command": cmd})
        return 0
    print(f"Trial: {trial}", file=sys.stderr)
    return launch_claude(trial, cmd, mode=mode, dev=dev)


def cmd_fork(args: argparse.Namespace) -> int:
    dev = _check_dev(args)
    src = _require_trial(args.trial)
    if args.name and not artifacts.SLUG_RE.match(args.name):
        raise CliError("--name must be lowercase letters, digits and dashes (max 48 chars)")
    base = st.new_trial_id() + (f"--{args.name}" if args.name else "")
    dst = st.unique_trial_dir(base)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".claude"))
    state = artifacts.load_state(dst)
    state.update(
        trial_id=dst.name, forked_from=src.name, sessions=[], last_claude_session_id=None,
        pending_slug=None, previous_trial_dirs=[], active_skill=None, package_version=__version__,
    )
    artifacts.save_state(dst, state)
    prepare_trial(dst, args.trust)
    update_results(dst)

    session = None
    if args.copy_transcript:
        project = st.claude_project_dir(dst)
        project.mkdir(parents=True, exist_ok=True)
        for path in st.transcripts(src):
            shutil.copy2(path, project / path.name)
        session = st.latest_session_id(dst)
    if session:
        cmd = build_claude_command(resume_session=session, fork_session=True, model=args.model,
                                   dev=dev, extra=args.claude_args)
        mode = "fork-resume"
    else:
        write_session_brief(dst, include_transcript=False)
        cmd = build_claude_command(
            model=args.model, dev=dev, extra=args.claude_args,
            prompt=f"This trial was forked from {src.name}. Read {SESSION_BRIEF}, then ask what to change.",
        )
        mode = "fork"
    if args.no_launch:
        _emit({"ok": True, "trial_dir": str(dst), "forked_from": src.name, "mode": mode, "command": cmd})
        return 0
    print(f"Forked {src.name} -> {dst}", file=sys.stderr)
    return launch_claude(dst, cmd, mode=mode, dev=dev)


def cmd_status(args: argparse.Namespace) -> int:
    trial = _require_trial(args.trial)
    _emit({**artifacts.snapshot(trial), "next_step": artifacts.next_step(trial)})
    return 0


def cmd_pick(argv: list[str]) -> int:
    """Bare `agentic-automl`: pick a recent trial and resume it, or start one if there are none."""
    from agentic_automl.trial_summary import collect_trial

    trials = st.recent_trials(15)
    if not trials:
        return main(["new", *argv])
    for i, trial in enumerate(trials, 1):
        row = collect_trial(trial)
        print(f"{i:>2}. {trial.name:<34} {str(row.get('goal') or '')[:36]:<36} next: {row.get('next_skill')}")
    if not sys.stdin.isatty():
        print("\nResume with: agentic-automl resume <trial>")
        return 0
    answer = input("\nResume which trial? [1, n = new, q = quit] ").strip().lower() or "1"
    if answer == "q":
        return 0
    if answer == "n":
        return main(["new", *argv])
    if not answer.isdigit() or not 1 <= int(answer) <= len(trials):
        raise CliError(f"Invalid choice: {answer}")
    return main(["resume", str(trials[int(answer) - 1]), *argv])


def cmd_community(args: argparse.Namespace) -> int:
    root = source_root()
    summary = root / "community" / "feedback_leaderboard_summary.json" if root else None
    if summary and summary.is_file():
        print(summary.read_text(encoding="utf-8"))
    else:
        print("No community leaderboard in this install yet. Share feedback from a trial with the "
              "`aaml-feedback` skill; maintainers publish the rollup via scripts/update_feedback_credits.py.")
    return 0


def _run_uv_tool(spec: str, index: str | None, dry_run: bool) -> int:
    index = index or st.load_config().get("package_index")
    cmd = ["uv", "tool", "install", "--force", spec] + (["--index", index] if index else [])
    if dry_run:
        _emit({"command": cmd})
        return 0
    if shutil.which("uv") is None:
        raise CliError("uv not found on PATH")
    return subprocess.call(cmd)


def cmd_update(args: argparse.Namespace) -> int:
    return _run_uv_tool(f"{PACKAGE_NAME}@latest", args.index, args.dry_run)


def cmd_rollback(args: argparse.Namespace) -> int:
    return _run_uv_tool(f"{PACKAGE_NAME}=={args.version}", args.index, args.dry_run)


def cmd_trial(args: argparse.Namespace) -> int:
    trial = _require_trial(args.trial)
    remediations = _optional_module("agentic_automl.remediations", "trial repair")
    _emit(remediations.repair_trial(trial, dry_run=args.dry_run))
    return 0


def cmd_overrides(args: argparse.Namespace) -> int:
    skills = {p.parent.name for p in (claude_template_dir() / "skills").glob("*/SKILL.md")}
    rows = []
    for path in sorted(st.overrides_dir().glob("*.md")):
        rows.append({"skill": path.stem, "path": str(path), "base_skill_exists": path.stem in skills})
    if args.action == "list":
        _emit({"overrides_dir": str(st.overrides_dir()), "overrides": rows})
    else:
        orphans = [r for r in rows if not r["base_skill_exists"]]
        _emit({
            "overrides": len(rows),
            "orphaned": orphans,
            "note": "Orphaned overrides target skills that no longer exist; rename or delete them. "
                    "Use the aaml-feedback skill to share overrides that should become platform changes.",
        })
    return 0


CONFIG_SETTERS = {
    "set-dataproc-project": "dataproc_project",
    "set-scratch-bucket": "scratch_bucket",
    "set-runtime-python": "runtime_python",
    "set-package-index": "package_index",
    "set-feedback-repo": "feedback_repo",
    "set-storage-dir": "storage_dir",
}


def cmd_config(args: argparse.Namespace) -> int:
    if args.action == "show":
        from agentic_automl.contracts import results_dir, trials_dir

        _emit({"config_path": str(st.config_path()), "config": st.load_config(),
               "runtime_python": st.runtime_python(), "trials_dir": str(trials_dir()),
               "results_dir": str(results_dir())})
        return 0
    if args.value is None:
        raise CliError(f"config {args.action} needs a value")
    if args.action == "set-scratch-bucket" and not args.value.startswith("gs://"):
        raise CliError("Scratch bucket must be a gs:// URL")
    _emit({"ok": True, "config": st.set_config_value(CONFIG_SETTERS[args.action], args.value)})
    return 0


def cmd_library_skills(args: argparse.Namespace) -> int:
    """List platform skills and user library skills (~/.agentic-automl/skills/<name>/SKILL.md)."""
    def describe(root: Path, origin: str) -> list[dict[str, str]]:
        return [{"name": p.parent.name, "origin": origin, "path": str(p)} for p in sorted(root.glob("*/SKILL.md"))]

    _emit(describe(claude_template_dir() / "skills", "platform") + describe(st.user_home() / "skills", "library"))
    return 0


def _optional_module(name: str, feature: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            raise CliError(f"`{feature}` is not available in this build yet") from exc
        raise


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-automl",
        description="Validator-gated Claude Code harness for supervised tabular ML.",
        epilog="Pass Claude Code flags after `--`, e.g. `agentic-automl resume -- --permission-mode plan`. "
               "Sub-tools: " + ", ".join(SUBTOOLS),
    )
    parser.add_argument("--version", action="version", version=f"{PACKAGE_NAME} {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    def launch_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", help="Claude model to use")
        p.add_argument("--trust", action="store_true",
                       help="Broader auto-approval for runtime Python + gsutil (sticky; env AGENTIC_AUTOML_TRUST=1)")
        p.add_argument("--dev", action="store_true", help="Developer mode from a source checkout")
        p.add_argument("--no-launch", action="store_true", help="Prepare the trial and print JSON; don't start Claude")

    p = sub.add_parser("new", help="Create a trial and launch Claude Code")
    p.add_argument("--dataset", help="Dataset path (local or gs://)")
    p.add_argument("--goal", help="What you want to predict and why")
    launch_opts(p)
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("resume", help="Resume a trial (default: most recent session)")
    p.add_argument("trial", nargs="?", help="Trial dir, id, id prefix, or slug")
    p.add_argument("--fresh", action="store_true", help="New chat on the same trial (writes work/session_brief.md)")
    p.add_argument("--summarize", action="store_true", help="Like --fresh, plus the prior transcript tail in the brief")
    launch_opts(p)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("fork", help="Copy a trial to a new trial and start a fresh chat")
    p.add_argument("trial", nargs="?")
    p.add_argument("--name", help="Slug for the new trial")
    p.add_argument("--copy-transcript", action="store_true", help="Carry the chat over (forks the Claude session)")
    launch_opts(p)
    p.set_defaults(func=cmd_fork)

    p = sub.add_parser("status", help="Trial snapshot + next step")
    p.add_argument("--trial")
    p.set_defaults(func=cmd_status)

    sub.add_parser("community", help="Feedback leaderboard").set_defaults(func=cmd_community)

    p = sub.add_parser("update", help="Upgrade via `uv tool install agentic-automl@latest`")
    p.add_argument("--index", help="Package index URL (default: config package_index)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("rollback", help="Install a specific earlier version")
    p.add_argument("version")
    p.add_argument("--index")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("trial", help="Trial maintenance")
    p.add_argument("action", choices=["repair"])
    p.add_argument("trial", nargs="?")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_trial)

    p = sub.add_parser("overrides", help="Personal skill overrides (~/.agentic-automl/skills/overrides)")
    p.add_argument("action", choices=["list", "reconcile"])
    p.set_defaults(func=cmd_overrides)

    p = sub.add_parser("config", help="User config (~/.agentic-automl/config.json)")
    p.add_argument("action", choices=["show", *CONFIG_SETTERS])
    p.add_argument("value", nargs="?")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("library-skills", help="List platform + library skills")
    p.add_argument("action", nargs="?", choices=["list"], default="list")
    p.set_defaults(func=cmd_library_skills)
    return parser


PASSTHROUGH_COMMANDS = {"new", "resume", "fork"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] in SUBTOOLS:
            module = _optional_module(SUBTOOLS[argv[0]], argv[0])
            return module.main(argv[1:])
        if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version")):
            return cmd_pick(argv)
        extra: list[str] = []
        if "--" in argv:
            i = argv.index("--")
            argv, extra = argv[:i], argv[i + 1:]
        parser = build_parser()
        args, unknown = parser.parse_known_args(argv)
        if args.cmd is None:
            parser.print_help()
            return 2
        if (unknown or extra) and args.cmd not in PASSTHROUGH_COMMANDS:
            parser.error(f"unrecognized arguments: {' '.join(unknown + extra)}")
        args.claude_args = unknown + extra
        return args.func(args)
    except (CliError, artifacts.ArtifactError) as exc:
        message = exc.payload if isinstance(exc, artifacts.ArtifactError) else {"ok": False, "error": str(exc)}
        print(json.dumps(message), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
