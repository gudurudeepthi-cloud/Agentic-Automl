import json
from pathlib import Path

from agentic_automl import cli
from agentic_automl.contracts import claude_template_dir

TEMPLATE = claude_template_dir()


def load(name: str) -> dict:
    return json.loads((TEMPLATE / name).read_text(encoding="utf-8"))


def test_settings_allowlist_matches_spec():
    allow = set(load("settings.json")["permissions"]["allow"])
    for rule in (
        "Read(./.claude/**)", "Read(./schemas/**)", "Read(./docs/**)", "Read(./src/**)",
        "Bash(ls *)", "Bash(cat *)", "Bash(head *)", "Bash(tail *)", "Bash(pwd)", "Bash(grep:*)",
        "Bash(agentic-automl artifacts *)", "Bash(agentic-automl feedback-episodes *)",
        "Bash(agentic-automl config *)", "Bash(/opt/conda/bin/python *)", "Bash(mkdir -p *)",
        "Edit(/trials/**/work/**)", "Read(~/.agentic-automl/skills/**)",
        "Write(~/.agentic-automl/skills/**)", "Edit(~/.agentic-automl/skills/**)", "Skill(*)",
    ):
        assert rule in allow, rule


def test_settings_has_bell_hooks_and_no_pretooluse():
    hooks = load("settings.json")["hooks"]
    assert {"Notification", "Stop"} <= set(hooks)
    assert "PreToolUse" not in hooks  # lives in hooks.json, merged into trials only


def test_hooks_json_wires_guards():
    pre = load("hooks.json")["hooks"]["PreToolUse"]
    by_matcher = {e["matcher"]: e["hooks"][0]["command"] for e in pre}
    assert by_matcher["Write|Edit|MultiEdit|NotebookEdit"] == \
        "agentic-automl artifacts validate-write"
    assert by_matcher["Bash|PowerShell"] == "agentic-automl bash-permissions check"
    [capture] = load("hooks.json")["hooks"]["PostToolUseFailure"]
    assert capture["matcher"] == "Bash|PowerShell" and capture["hooks"][0]["async"] is True
    assert capture["hooks"][0]["command"].startswith("agentic-automl feedback-episodes capture")


def _trial_settings(capsys) -> tuple[Path, dict]:
    cli.main(["new", "--no-launch"])
    trial = Path(json.loads(capsys.readouterr().out)["trial_dir"])
    return trial, json.loads((trial / ".claude" / "settings.json").read_text(encoding="utf-8"))


def test_trial_settings_merge_hooks_idempotently(capsys):
    trial, settings = _trial_settings(capsys)
    assert len(settings["hooks"]["PreToolUse"]) == 2
    assert "Notification" in settings["hooks"]
    cli.sync_runtime_files(trial)
    cli.sync_runtime_files(trial)
    again = json.loads((trial / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert again == settings


def test_trial_settings_mirror_bash_rules_for_powershell(capsys):
    _, settings = _trial_settings(capsys)
    allow = settings["permissions"]["allow"]
    for rule in ("PowerShell(agentic-automl artifacts *)", "PowerShell(agentic-automl feedback-episodes *)",
                 "PowerShell(cp .claude/scripts/*)", "PowerShell(Copy-Item .claude/scripts/*)"):
        assert rule in allow, rule
    bash = {r[len("Bash("):] for r in allow if r.startswith("Bash(")}
    assert bash <= {r[len("PowerShell("):] for r in allow if r.startswith("PowerShell(")}


def test_hook_commands_are_shell_neutral():
    for entries in load("hooks.json")["hooks"].values():
        for e in entries:
            assert "${" not in e["hooks"][0]["command"], e  # bash-only syntax breaks under PowerShell


def test_claude_env_finds_git_bash_on_windows(monkeypatch, tmp_path):
    git = tmp_path / "Git" / "mingw64" / "bin" / "git.exe"
    bash = tmp_path / "Git" / "bin" / "bash.exe"
    for p in (git, bash):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(git) if name == "git" else None)
    monkeypatch.delenv("CLAUDE_CODE_GIT_BASH_PATH", raising=False)
    env = cli.claude_env(tmp_path)
    assert env["CLAUDE_CODE_GIT_BASH_PATH"] == str(bash) and env["TRIAL_DIR"] == str(tmp_path)


def test_runtime_python_substituted_in_trial(capsys, monkeypatch):
    monkeypatch.setenv("AGENTIC_AUTOML_PYTHON", "/usr/local/bin/python3")
    trial, settings = _trial_settings(capsys)
    assert "Bash(/usr/local/bin/python3 *)" in settings["permissions"]["allow"]
    assert not any("/opt/conda" in r for r in settings["permissions"]["allow"])
    runtime_md = (trial / ".claude" / "CLAUDE.md").read_text(encoding="utf-8")
    assert "/usr/local/bin/python3" in runtime_md and "/opt/conda" not in runtime_md


def test_runtime_claude_md_covers_validator_contract():
    text = (TEMPLATE / "CLAUDE.md").read_text(encoding="utf-8")
    for needle in ("begin-skill", "save-artifact", "next-step", "status", "set-trial-slug",
                   "run_state.json", "work/", "redirect_to", "auto_research", "parity-ml-export"):
        assert needle in text, needle
    assert (TEMPLATE / "trial-backtrack.md").is_file()
