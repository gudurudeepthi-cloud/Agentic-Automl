import json
import os
from pathlib import Path

import pytest

from agentic_automl import artifacts, cli
from agentic_automl import state as st
from agentic_automl.contracts import TRIAL_SUBDIRS, trials_dir
from conftest import save, save_through


def write_transcript(trial: Path, session_id: str, *turns: tuple[str, str]) -> Path:
    project = st.claude_project_dir(trial)
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{session_id}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for role, text in turns:
            fh.write(json.dumps({"type": role, "message": {"role": role, "content": [{"type": "text", "text": text}]}}) + "\n")
    return path


class FakeClaude:
    """Stands in for subprocess.call; optionally writes a transcript like Claude Code would."""

    def __init__(self, session_id: str | None = "sess-new"):
        self.calls: list[dict] = []
        self.session_id = session_id

    def __call__(self, cmd, cwd=None, env=None):
        self.calls.append({"cmd": cmd, "cwd": Path(cwd), "env": env})
        if self.session_id:
            write_transcript(Path(cwd), self.session_id, ("user", "hi"), ("assistant", "hello"))
        return 0


@pytest.fixture
def fake_claude(monkeypatch):
    fake = FakeClaude()
    monkeypatch.setattr(cli.subprocess, "call", fake)
    return fake


def run_json(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


# --- new --------------------------------------------------------------------


def test_new_scaffolds_trial(capsys):
    code, out = run_json(capsys, "new", "--dataset", "data/x.csv", "--goal", "predict churn", "--no-launch")
    assert code == 0
    trial = Path(out["trial_dir"])
    assert trial.parent == trials_dir()
    for sub in TRIAL_SUBDIRS:
        assert (trial / sub).is_dir(), sub
    state = artifacts.load_state(trial)
    assert state["goal"] == "predict churn" and state["dataset_path"] == "data/x.csv"
    assert (trial / ".claude" / "CLAUDE.md").is_file()
    assert (trial / ".claude" / "skills" / "domains" / "registry.json").is_file()
    assert not (trial / ".claude" / "settings.local.json").exists()
    assert not (trial / cli.TRUST_FILE).exists()


def test_new_launches_claude_in_trial(fake_claude):
    assert cli.main(["new", "--goal", "churn", "--model", "opus"]) == 0
    call = fake_claude.calls[0]
    trial = call["cwd"]
    assert call["cmd"][0] == "claude-fake"
    assert call["cmd"][1:3] == ["--model", "opus"]
    assert "Goal: churn" in call["cmd"][-1]
    assert call["env"]["TRIAL_DIR"] == str(trial)
    assert call["env"]["UV_PROJECT"]
    state = artifacts.load_state(trial)
    assert state["sessions"][0]["mode"] == "new"
    assert state["last_claude_session_id"] == "sess-new"


def test_unknown_flags_pass_through(fake_claude):
    cli.main(["new", "--verbose", "--", "--permission-mode", "plan"])
    cmd = fake_claude.calls[0]["cmd"]
    assert cmd[-3:] == ["--verbose", "--permission-mode", "plan"]


def test_unknown_flags_rejected_for_non_launch_commands(trial):
    with pytest.raises(SystemExit):
        cli.main(["status", "--trial", str(trial), "--bogus"])


# --- trust ------------------------------------------------------------------


def test_trust_is_sticky_and_idempotent(capsys):
    _, out = run_json(capsys, "new", "--trust", "--no-launch")
    trial = Path(out["trial_dir"])
    assert (trial / cli.TRUST_FILE).is_file()
    local = trial / ".claude" / "settings.local.json"
    allow = json.loads(local.read_text())["permissions"]["allow"]
    assert "Bash(gsutil cp *)" in allow and f"Bash({st.runtime_python()} *)" in allow
    run_json(capsys, "resume", str(trial), "--no-launch")  # no --trust: still applied, not duplicated
    assert json.loads(local.read_text())["permissions"]["allow"] == allow


def test_trust_env(capsys, monkeypatch):
    monkeypatch.setenv("AGENTIC_AUTOML_TRUST", "1")
    _, out = run_json(capsys, "new", "--no-launch")
    assert (Path(out["trial_dir"]) / cli.TRUST_FILE).is_file()


# --- resume -----------------------------------------------------------------


def test_resume_continues_known_session(trial, capsys):
    write_transcript(trial, "sess-1", ("user", "hi"))
    artifacts.record_session(trial, "new", "sess-1")
    _, out = run_json(capsys, "resume", str(trial), "--no-launch")
    assert out["mode"] == "resume"
    assert out["command"][1:3] == ["--resume", "sess-1"]


def test_resume_without_transcript_starts_fresh_with_brief(trial, capsys):
    save(trial, "prediction_contract")
    _, out = run_json(capsys, "resume", str(trial), "--no-launch")
    assert out["mode"] == "fresh"
    brief = (trial / cli.SESSION_BRIEF).read_text(encoding="utf-8")
    assert "[x] prediction_contract" in brief and "`data-profiling`" in brief


def test_resume_summarize_includes_transcript(trial, capsys):
    write_transcript(trial, "sess-1", ("user", "target is churned"), ("assistant", "Got it"))
    run_json(capsys, "resume", str(trial), "--summarize", "--no-launch")
    brief = (trial / cli.SESSION_BRIEF).read_text(encoding="utf-8")
    assert "[user] target is churned" in brief and "[assistant] Got it" in brief


def test_resume_defaults_to_most_recent_session(tmp_path, capsys):
    old = tmp_path / "trials" / "a-old"
    new = tmp_path / "trials" / "b-new"
    for trial, started in ((old, "2026-01-01T00:00:00+00:00"), (new, "2026-02-01T00:00:00+00:00")):
        state = artifacts.ensure_state(trial)
        state["sessions"].append({"started_at": started, "mode": "new", "claude_session_id": None})
        artifacts.save_state(trial, state)
    os.utime(old, (2_000_000_000, 2_000_000_000))  # newer mtime must not beat a newer session
    _, out = run_json(capsys, "resume", "--no-launch")
    assert Path(out["trial_dir"]).name == "b-new"


def test_resume_applies_pending_slug_and_moves_transcripts(trial, capsys):
    write_transcript(trial, "sess-1", ("user", "hi"))
    artifacts.record_session(trial, "new", "sess-1")
    artifacts.set_trial_slug(trial, "churn")
    _, out = run_json(capsys, "resume", str(trial), "--no-launch")
    new_dir = Path(out["trial_dir"])
    assert new_dir.name == "t001--churn" and not trial.exists()
    assert out["command"][1:3] == ["--resume", "sess-1"]
    assert (st.claude_project_dir(new_dir) / "sess-1.jsonl").is_file()


def test_resume_by_slug(trial, capsys):
    artifacts.set_trial_slug(trial, "churn")
    artifacts.apply_pending_slug(trial)
    _, out = run_json(capsys, "resume", "churn", "--no-launch")
    assert Path(out["trial_dir"]).name == "t001--churn"


def test_resume_missing_trial(capsys):
    assert cli.main(["resume", "nope", "--no-launch"]) == 1


# --- fork -------------------------------------------------------------------


def test_fork_copies_state_and_starts_fresh(trial, capsys):
    save_through(trial, "workflow_plan")
    artifacts.record_session(trial, "new", "sess-1")
    _, out = run_json(capsys, "fork", str(trial), "--name", "try-xgb", "--no-launch")
    dst = Path(out["trial_dir"])
    assert dst.name.endswith("--try-xgb") and out["mode"] == "fork"
    state = artifacts.load_state(dst)
    assert state["trial_id"] == dst.name and state["forked_from"] == "t001"
    assert state["sessions"] == [] and state["last_claude_session_id"] is None
    assert artifacts.evaluate_gates(dst)["workflow_plan_approved"] is True
    assert (dst / cli.SESSION_BRIEF).is_file()
    assert artifacts.load_state(trial)["trial_id"] == "t001"  # source untouched


def test_fork_copy_transcript_forks_session(trial, capsys):
    write_transcript(trial, "sess-1", ("user", "hi"))
    _, out = run_json(capsys, "fork", str(trial), "--copy-transcript", "--no-launch")
    assert out["mode"] == "fork-resume"
    assert out["command"][1:4] == ["--resume", "sess-1", "--fork-session"]


def test_fork_rejects_bad_name(trial):
    assert cli.main(["fork", str(trial), "--name", "Bad Name", "--no-launch"]) == 1


# --- misc -------------------------------------------------------------------


def test_status(trial, capsys):
    code, out = run_json(capsys, "status", "--trial", str(trial))
    assert code == 0 and out["trial_id"] == "t001"
    assert out["next_step"]["next_skill"] == "prediction-contract"


def test_config(capsys):
    assert cli.main(["config", "set-scratch-bucket", "not-a-bucket"]) == 1
    capsys.readouterr()
    code, out = run_json(capsys, "config", "set-scratch-bucket", "gs://scratch")
    assert code == 0 and out["config"]["scratch_bucket"] == "gs://scratch"
    run_json(capsys, "config", "set-runtime-python", "/usr/bin/python3")
    assert st.runtime_python() == "/usr/bin/python3"


def test_update_dry_run(capsys):
    _, out = run_json(capsys, "update", "--dry-run", "--index", "https://example.invalid/simple")
    assert out["command"][:5] == ["uv", "tool", "install", "--force", "agentic-automl@latest"]
    _, out = run_json(capsys, "rollback", "0.0.9", "--dry-run")
    assert "agentic-automl==0.0.9" in out["command"]


def test_overrides_list(capsys):
    overrides = st.overrides_dir()
    overrides.mkdir(parents=True)
    (overrides / "ghost-skill.md").write_text("x")
    _, out = run_json(capsys, "overrides", "reconcile")
    assert out["orphaned"][0]["skill"] == "ghost-skill"


def test_summarize_trials(trial, capsys):
    save(trial, "prediction_contract")
    assert cli.main(["summarize-trials"]) == 0
    out = capsys.readouterr().out
    assert "t001" in out and "data-profiling" in out


def test_bare_lists_trials_when_not_a_tty(trial, capsys, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
    assert cli.main([]) == 0
    assert "t001" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "agentic-automl" in capsys.readouterr().out
