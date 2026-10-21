"""Agent eval harness: seed trials from scenarios, optionally run the agent, score invariants.

Tiers (--tier):
  deterministic  no Claude spend: copy each scenario's fixture seed into a fresh trial and run
                 the deterministic scorers against it (proves scorers, fixtures, schemas, gates).
  auth           cheap preflight only: claude CLI, runtime Python + ML stack, GCS credentials,
                 GitHub CLI auth. Runs no scenarios.
  component      real agent run of one skill from a mid-chain fixture seed     (costs money)
  e2e            real agent run of the whole chain from an empty trial           (costs money)
  real           e2e on full-size data                                           (costs money)

Paid tiers need --trust-runtime: the agent runs headless (claude -p) in trust mode inside a
throwaway trial under --output-dir. Clarifying questions are answered from the scenario's
answer_bank, which is appended to the prompt.

Scenarios: benchmarks/evals/scenarios/*.json (schemas/eval_scenario.schema.json).
Results:   <output-dir>/results/<scenario>.json (schemas/eval_result.schema.json) + --report-md.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft7Validator

from agentic_automl import artifacts
from agentic_automl.benchmark_readiness import benchmarks_dir, repo_root
from agentic_automl.contracts import schemas_dir
from agentic_automl.eval_reports import render_results_markdown, render_results_tsv, verdict
from agentic_automl.eval_scorers import artifact_snapshot, score_trial

TIERS = ("deterministic", "auth", "component", "e2e", "real")
PAID_TIERS = {"component", "e2e", "real"}


@dataclass
class EvalRunContext:
    tier: str
    output_dir: Path
    trust_runtime: bool = False
    model: str | None = None
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-"
                        + uuid.uuid4().hex[:6])


@dataclass
class PreparedEvalTrial:
    scenario: dict
    trial_dir: Path


@dataclass
class AgentRunResult:
    exit_code: int | None
    stop_reason: str | None
    cost_usd: float | None
    turns: int | None
    stdout_path: str | None
    timed_out: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validator(name: str) -> Draft7Validator:
    return Draft7Validator(json.loads((schemas_dir() / name).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- scenarios

def scenarios_dir(root: Path | None = None) -> Path:
    return benchmarks_dir(root) / "evals" / "scenarios"


def load_scenario_file(path: Path) -> dict:
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
              for e in _validator("eval_scenario.schema.json").iter_errors(scenario)]
    if errors:
        raise ValueError(f"{Path(path).name}: " + "; ".join(errors))
    return scenario


def load_scenarios(root: Path | None = None, *, dataset: str | None = None, scenario: str | None = None,
                   tier: str | None = None) -> list[dict]:
    out = []
    for p in sorted(scenarios_dir(root).glob("*.json")):
        s = load_scenario_file(p)
        if dataset and s["dataset_id"] != dataset:
            continue
        if scenario and not fnmatch.fnmatch(s["id"], scenario):
            continue
        if tier and tier not in s.get("tiers", ["deterministic"]):
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- trials

REPO_PLACEHOLDER = "{{REPO_ROOT}}"  # committed seeds store paths relative to the checkout


def fill_repo_placeholder(trial: Path, root: Path) -> None:
    for p in trial.rglob("*"):
        if p.suffix in (".json", ".md") and p.is_file():
            text = p.read_text(encoding="utf-8")
            if REPO_PLACEHOLDER in text:
                p.write_text(text.replace(REPO_PLACEHOLDER, root.as_posix()), encoding="utf-8")


def prepare_trial(scenario: dict, ctx: EvalRunContext, root: Path | None = None) -> PreparedEvalTrial:
    from agentic_automl import cli

    # Short on purpose: trials nest .claude/skills/... and Windows caps paths at 260 characters.
    trial = ctx.output_dir / "t" / f"{scenario['id']}-{ctx.run_id[-6:]}"
    seed = scenario["seed"]
    if seed["kind"] == "fixture":
        src = benchmarks_dir(root) / "evals" / "fixtures" / "seeds" / seed["name"]
        if not src.is_dir():
            raise FileNotFoundError(f"fixture seed {seed['name']!r} missing: run scripts/generate_dataset_seeds.py")
        shutil.copytree(src, trial)
        fill_repo_placeholder(trial, root or repo_root())
    else:
        trial.mkdir(parents=True)
    dataset_path = scenario.get("dataset_path")
    if dataset_path and not Path(dataset_path).is_absolute():
        dataset_path = str((root or repo_root()) / dataset_path)
    artifacts.ensure_state(trial, goal=scenario["initial_prompt"][:200], dataset_path=dataset_path)
    cli.prepare_trial(trial, trust=ctx.trust_runtime)
    return PreparedEvalTrial(scenario, trial)


def build_prompt(scenario: dict, dataset_path: str | None) -> str:
    lines = [scenario["initial_prompt"]]
    if dataset_path:
        lines.append(f"\nDataset: {dataset_path}")
    if scenario["answer_bank"]:
        lines.append("\nThis is a headless evaluation run: nobody can reply. When you would ask the user "
                     "something, use the matching answer below and continue; if nothing matches, pick the "
                     "documented default and say so.")
        root = repo_root().as_posix()
        lines += [f"- If asked about \"{a['match']}\": {a['answer'].replace(REPO_PLACEHOLDER, root)}"
                  for a in scenario["answer_bank"]]
    lines.append("\nWork through the chain without waiting for confirmation; treat this message as approval "
                 f"for the prediction contract, the workflow plan and compute within the limits. Start at "
                 f"`{scenario['entry_skill']}`.")
    return "\n".join(lines)


def run_agent(prepared: PreparedEvalTrial, ctx: EvalRunContext) -> AgentRunResult:
    from agentic_automl import cli

    s, trial = prepared.scenario, prepared.trial_dir
    dataset = artifacts.load_state(trial).get("dataset_path")
    cmd = [cli.claude_executable(), "-p", build_prompt(s, dataset), "--output-format", "json",
           "--max-turns", str(s["limits"]["max_turns"]), "--permission-mode", "acceptEdits"]
    if ctx.model:
        cmd += ["--model", ctx.model]
    env = {**os.environ, "TRIAL_DIR": str(trial),
           "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])}
    log = trial / "work" / "eval_agent_output.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(cmd, cwd=trial, env=env, capture_output=True, text=True,
                              timeout=s["limits"]["timeout_seconds"])
    except subprocess.TimeoutExpired:
        return AgentRunResult(None, "timeout", None, None, None, timed_out=True)
    log.write_text(proc.stdout, encoding="utf-8")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {}
    return AgentRunResult(proc.returncode, out.get("subtype") or ("error" if proc.returncode else None),
                          out.get("total_cost_usd"), out.get("num_turns"), str(log))


def run_scenario(scenario: dict, ctx: EvalRunContext, root: Path | None = None) -> dict:
    started, t0 = _now(), time.time()
    result = {"schema_version": 1, "scenario_id": scenario["id"], "dataset_id": scenario["dataset_id"],
              "run_id": ctx.run_id, "tier": ctx.tier,
              "runner": "claude" if ctx.tier in PAID_TIERS else "deterministic",
              "stop_reason": None, "trial_dir": None, "artifact_snapshot": {}, "scores": [],
              "process_metrics": {}, "cost": {}, "logs": {}}
    try:
        prepared = prepare_trial(scenario, ctx, root)
        result["trial_dir"] = str(prepared.trial_dir)
        if ctx.tier in PAID_TIERS:
            run = run_agent(prepared, ctx)
            result["stop_reason"] = run.stop_reason
            result["cost"] = {"usd": run.cost_usd, "budget_usd": scenario["limits"]["max_budget_usd"]}
            result["process_metrics"]["turns"] = run.turns
            result["logs"]["agent_output"] = run.stdout_path
        scored = score_trial(scenario, prepared.trial_dir)
        result["scores"] = scored.score_dicts()
        result["artifact_snapshot"] = scored.artifact_snapshot
        result["process_metrics"].update(scored.process_metrics)
        if ctx.tier in PAID_TIERS and result["stop_reason"] in ("timeout", "error_max_turns"):
            result["status"] = "incomplete"
        else:
            result["status"] = "pass" if scored.passed else "fail"
        over = (result["cost"].get("usd") or 0) > scenario["limits"]["max_budget_usd"] > 0
        if over:
            result["scores"].append({"name": "budget", "passed": False,
                                     "detail": f"cost ${result['cost']['usd']:.2f} > ${scenario['limits']['max_budget_usd']}"})
            result["status"] = "fail"
    except FileNotFoundError as exc:
        result.update(status="harness-unavailable", stop_reason=str(exc))
        result["artifact_snapshot"] = artifact_snapshot(Path(result["trial_dir"])) if result["trial_dir"] else {}
    except Exception as exc:  # noqa: BLE001  a harness bug must not look like an agent failure
        result.update(status="harness-error", stop_reason=f"{type(exc).__name__}: {exc}")
    result.update(started_at=started, ended_at=_now(), elapsed_seconds=round(time.time() - t0, 2))
    errors = list(_validator("eval_result.schema.json").iter_errors(result))
    if errors:
        result.update(status="harness-error", stop_reason=f"invalid eval_result: {errors[0].message}")
    return result


# --------------------------------------------------------------------------- auth tier

def _probe(cmd: list[str], timeout: int = 60) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return p.returncode == 0, (p.stdout or p.stderr).strip().splitlines()[-1:][0] if (p.stdout or p.stderr) else ""


def _probe_claude() -> dict:
    from agentic_automl import cli

    try:
        exe = cli.claude_executable()
    except cli.CliError as exc:
        return {"name": "claude", "ok": False, "detail": str(exc), "fix": "Install Claude Code or set AGENTIC_AUTOML_CLAUDE."}
    ok, detail = _probe([exe, "--version"])
    return {"name": "claude", "ok": ok, "detail": detail, "fix": "Run `claude` once interactively to log in."}


def _probe_runtime() -> dict:
    from agentic_automl.state import runtime_python

    py = runtime_python()
    ok, detail = _probe([py, "-c", "import sklearn, pandas, lightgbm, pyarrow, matplotlib; print('ML stack ok')"])
    return {"name": "runtime_python", "ok": ok, "detail": f"{py}: {detail}",
            "fix": "agentic-automl config set-runtime-python <python with the modeling stack>"}


def _probe_gcs() -> dict:
    from agentic_automl.state import runtime_python

    ok, detail = _probe([runtime_python(), "-c",
                         "import google.auth, gcsfs; c, p = google.auth.default(); print('project', p)"])
    return {"name": "gcs", "ok": ok, "detail": detail,
            "fix": "gcloud auth application-default login (only needed for gcs datasets)"}


def _probe_github() -> dict:
    if not shutil.which("gh"):
        return {"name": "github", "ok": False, "detail": "gh not on PATH", "fix": "Install GitHub CLI, then gh auth login"}
    ok, detail = _probe(["gh", "auth", "status"])
    return {"name": "github", "ok": ok, "detail": detail, "fix": "gh auth login"}


def describe_auth_readiness() -> list[dict]:
    return [_probe_claude(), _probe_runtime(), _probe_gcs(), _probe_github()]


def render_auth_report(probes: list[dict]) -> str:
    lines = ["# Eval auth preflight", "", "| check | ok | detail | remediation |", "|---|---|---|---|"]
    lines += [f"| {p['name']} | {'yes' if p['ok'] else 'NO'} | {p['detail']} | {'' if p['ok'] else p['fix']} |"
              for p in probes]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_agent_evals", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS, default="deterministic")
    ap.add_argument("--output-dir", type=Path, default=Path(os.environ.get("TEMP", "/tmp")) / "agentic-automl-evals")
    ap.add_argument("--report-md", type=Path, default=None)
    ap.add_argument("--results-tsv", type=Path, default=None,
                    help="also write one row per scenario (status, test/CV scores, model) as TSV")
    ap.add_argument("--dataset", default=None, help="only scenarios for this dataset_id")
    ap.add_argument("--scenario", default=None, help="only scenario ids matching this glob")
    ap.add_argument("--trust-runtime", action="store_true", help="required for paid tiers (headless agent runs)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--list", action="store_true", help="list matching scenarios and exit")
    args = ap.parse_args(argv)

    if args.tier == "auth":
        probes = describe_auth_readiness()
        report = render_auth_report(probes)
        print(report)
        if args.report_md:
            args.report_md.parent.mkdir(parents=True, exist_ok=True)
            args.report_md.write_text(report, encoding="utf-8")
        return 0 if all(p["ok"] for p in probes if p["name"] in ("claude", "runtime_python")) else 1

    scenarios = load_scenarios(dataset=args.dataset, scenario=args.scenario, tier=args.tier)
    if args.list:
        for s in scenarios:
            print(f"{s['id']:<48} {s['dataset_id']:<22} entry={s['entry_skill']} seed={s['seed']}")
        return 0
    if args.tier in PAID_TIERS and not args.trust_runtime:
        print(f"--tier {args.tier} runs the real agent and costs money: pass --trust-runtime to confirm.",
              file=sys.stderr)
        return 2
    ctx = EvalRunContext(args.tier, args.output_dir.resolve(), args.trust_runtime, args.model)
    (ctx.output_dir / "results").mkdir(parents=True, exist_ok=True)
    results = []
    for s in scenarios:
        r = run_scenario(s, ctx)
        (ctx.output_dir / "results" / f"{s['id']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        failed = [x["name"] for x in r["scores"] if not x["passed"]]
        print(f"[{r['status']:>22}] {s['id']} ({r['elapsed_seconds']}s)" + (f" failed: {failed}" if failed else "")
              + (f" - {r['stop_reason']}" if r["stop_reason"] else ""), flush=True)
        results.append(r)
    md = render_results_markdown(results, f"Agent eval report: tier {args.tier}")
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(md, encoding="utf-8")
    if args.results_tsv:
        args.results_tsv.parent.mkdir(parents=True, exist_ok=True)
        args.results_tsv.write_text(render_results_tsv(results), encoding="utf-8", newline="\n")
        print(f"TSV: {args.results_tsv}")
    ok, why = verdict(results)
    print(f"\n{'PASS' if ok else 'FAIL'}: {why}. Results: {ctx.output_dir / 'results'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- recommender

RULES: list[tuple[str, str, str, str]] = [
    # (path glob, level, tier, why)
    ("schemas/*.schema.json", "MUST", "deterministic", "artifact/eval schema changed"),
    ("src/agentic_automl/artifacts.py", "MUST", "deterministic", "validator / gates changed"),
    ("src/agentic_automl/contracts.py", "MUST", "deterministic", "artifact registry / gate order changed"),
    ("src/agentic_automl/eval_*.py", "MUST", "deterministic", "eval harness changed"),
    ("benchmarks/evals/*", "MUST", "deterministic", "scenarios or fixtures changed"),
    ("src/agentic_automl/cli.py", "CHECK", "e2e", "launcher / trial setup changed"),
    ("src/agentic_automl/bash_permissions.py", "CHECK", "e2e", "agent permission guard changed"),
    (".claude/hooks.json", "CHECK", "e2e", "PreToolUse guards changed"),
    (".claude/settings.json", "CHECK", "e2e", "agent permissions changed"),
    (".claude/CLAUDE.md", "CHECK", "e2e", "runtime instructions changed"),
    ("scripts/steps/*", "CHECK", "e2e", "step templates the agent copies changed"),
    ("benchmarks/datasets.json", "MUST", "deterministic", "benchmark registry changed"),
]


def recommend(changed: list[str], root: Path | None = None) -> list[dict]:
    recs: dict[tuple[str, str], dict] = {}
    scenarios = {s["id"]: s for s in load_scenarios(root)}

    def add(level, tier, why, path, ids=()):
        key = (tier, level)
        r = recs.setdefault(key, {"level": level, "tier": tier, "why": set(), "paths": set(), "scenarios": set()})
        r["why"].add(why)
        r["paths"].add(path)
        r["scenarios"].update(ids)

    for path in changed:
        for glob, level, tier, why in RULES:
            if fnmatch.fnmatch(path, glob):
                add(level, tier, why, path)
        parts = path.split("/")
        if len(parts) >= 3 and parts[:2] == [".claude", "skills"] and parts[2] not in ("domains", "planned"):
            skill = parts[2]
            ids = [i for i, s in scenarios.items() if s["entry_skill"] == skill and "component" in s.get("tiers", [])]
            add("MUST", "component", f"skill {skill} changed", path, ids)
            add("MUST", "deterministic", "skill text referenced by tests", path)
        if path.startswith(".claude/skills/domains/"):
            add("CHECK", "component", "domain overlay changed", path)
    order = {"MUST": 0, "CHECK": 1}
    tier_order = {t: i for i, t in enumerate(TIERS)}
    out = [{**r, "why": sorted(r["why"]), "paths": sorted(r["paths"]), "scenarios": sorted(r["scenarios"])}
           for r in recs.values()]
    return sorted(out, key=lambda r: (order[r["level"]], tier_order[r["tier"]]))


def changed_files(base: str, head: str) -> list[str]:
    p = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], capture_output=True, text=True,
                       cwd=repo_root())
    if p.returncode != 0:
        raise SystemExit(f"git diff {base}...{head} failed: {p.stderr.strip()}")
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def recommend_evals_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recommend_evals",
                                 description="Which eval tiers a diff needs (recommend-only; never runs a paid tier).")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--files", nargs="*", default=None, help="explicit changed paths instead of git diff")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    files = args.files if args.files is not None else changed_files(args.base, args.head)
    recs = recommend(files)
    if args.json:
        print(json.dumps({"changed": files, "recommendations": recs}, indent=2))
        return 0
    print(f"{len(files)} changed file(s).")
    if not recs:
        print("No eval tier required (docs/tests-only change). `pytest` is still the gate.")
    for r in recs:
        scen = f" scenarios: {', '.join(r['scenarios'])}" if r["scenarios"] else ""
        print(f"{r['level']:<5} --tier {r['tier']:<13} {'; '.join(r['why'])}{scen}")
        for p in r["paths"][:8]:
            print(f"        {p}")
    paid = [r for r in recs if r["tier"] in PAID_TIERS]
    if paid:
        print("\nPaid tiers cost money: run them with --trust-runtime only when you intend to.")
    return 0
