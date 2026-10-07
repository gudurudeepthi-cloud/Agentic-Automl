import io
import json

import pytest

from agentic_automl import cli
from agentic_automl import feedback_episodes as fe
from conftest import save, valid


def run(capsys, trial, *argv):
    code = cli.main(["feedback-episodes", argv[0], "--trial", str(trial), *argv[1:]])
    return code, json.loads(capsys.readouterr().out)


def test_insight_resolve_list(trial, capsys):
    _, out = run(capsys, trial, "insight", "lightgbm needs categorical dtype", "--skill", "training", "--layer", "skill")
    episode_id = out["episode"]["id"]
    _, out = run(capsys, trial, "list", "--status", "open")
    assert [e["id"] for e in out["episodes"]] == [episode_id]
    code, _ = run(capsys, trial, "resolve", episode_id, "--note", "fixed in skill")
    assert code == 0
    _, out = run(capsys, trial, "list")
    assert out["episodes"][0]["status"] == "resolved"
    assert out["episodes"][0]["resolution_note"] == "fixed in skill"


def test_resolve_unknown(trial, capsys):
    code, out = run(capsys, trial, "resolve", "nope")
    assert code == 1 and out["ok"] is False


def test_rationale_and_type_filter(trial, capsys):
    run(capsys, trial, "rationale", "temporal split: data drifts", "--decision", "prediction_contract")
    run(capsys, trial, "insight", "x")
    _, out = run(capsys, trial, "list", "--type", "rationale")
    assert len(out["episodes"]) == 1 and out["episodes"][0]["decision"] == "prediction_contract"


@pytest.mark.parametrize(
    "hook,captured",
    [
        ({"tool_name": "Bash", "tool_input": {"command": "python x.py"},
          "tool_response": {"stdout": "", "stderr": "Traceback (most recent call last):\n KeyError", "interrupted": False}}, True),
        ({"tool_name": "Bash", "tool_input": {"command": "python x.py"},
          "tool_response": {"stdout": "ok", "stderr": "", "interrupted": False}}, False),
        ({"tool_name": "Bash", "tool_input": {"command": "python x.py"},
          "tool_response": {"stdout": "", "stderr": "", "interrupted": True}}, False),
        ({"tool_name": "Bash", "tool_input": {"command": "false"}, "hook_event_name": "PostToolUseFailure",
          "error": "Exit code 1"}, True),
        ({"tool_name": "Bash", "tool_input": {"command": "agentic-automl feedback-episodes list"},
          "error": "boom"}, False),
        ({"tool_name": "Bash", "tool_input": {"command": "sleep 99"}, "hook_event_name": "PostToolUseFailure",
          "error": "Interrupted", "is_interrupt": True}, False),
        ({"tool_name": "Write", "tool_input": {}, "error": "boom"}, False),
    ],
)
def test_failure_from_hook(hook, captured):
    assert (fe.failure_from_hook(hook) is not None) is captured


def test_capture_is_silent_and_never_fails(trial, capsys, monkeypatch):
    hook = {"tool_name": "Bash", "tool_input": {"command": "python x.py"}, "error": "Exit code 2"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook)))
    assert cli.main(["feedback-episodes", "capture", "--trial", str(trial)]) == 0
    assert capsys.readouterr().out == ""
    assert fe.episodes(trial)[0]["type"] == "error"
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert cli.main(["feedback-episodes", "capture", "--trial", str(trial)]) == 0


def test_triage_then_filter_redacts(trial, capsys):
    record = fe.append(trial, {"type": "error", "command": "python /home/alice/x.py gs://secret-bucket/data",
                               "error": "contact bob@corp.com, account 1234567890123"})
    _, out = run(capsys, trial, "filter")
    assert out["episodes"] == []  # untriaged errors are never shared
    _, out = run(capsys, trial, "triage", record["id"], "--layer", "skill", "--target", "training")
    _, out = run(capsys, trial, "filter")
    [shared] = out["episodes"]
    text = json.dumps(shared)
    for secret in ("alice", "secret-bucket", "bob@corp.com", "1234567890123"):
        assert secret not in text, secret
    assert shared["layer"] == "skill" and shared["target"] == "training"


def test_decision_values_dropped_when_shared(trial):
    proposal = trial / "work" / "decision_feedback" / "prediction_contract.proposal.json"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(json.dumps(valid("prediction_contract")), encoding="utf-8")
    save(trial, "prediction_contract", valid("prediction_contract", target_column="secret_col"))
    [shared] = fe.filter_for_sharing(fe.episodes(trial))
    assert shared["type"] == "decision"
    assert "secret_col" not in json.dumps(shared)
    assert shared["changes"] == [{"path": "$.target_column", "change": "modified"}]
