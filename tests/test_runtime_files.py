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
        'agentic-automl artifacts validate-write --trial "${TRIAL_DIR:-.}"'
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
    for rule in ("PowerShell(agentic-automl artifacts *)", "PowerShell(agentic-automl feedback-episodes *)"):
        assert rule in allow, rule
    assert sum(r.startswith("PowerShell(") for r in allow) == sum(r.startswith("Bash(") for r in allow)


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
