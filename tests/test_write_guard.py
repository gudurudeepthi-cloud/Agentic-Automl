import io
import json

import pytest

from agentic_automl import artifacts
from agentic_automl.contracts import ARTIFACT_REGISTRY, artifact_key_for_path


def hook(tool, path, cwd=None):
    data = {"tool_name": tool, "tool_input": {"file_path": str(path)}}
    if cwd is not None:
        data["cwd"] = str(cwd)
    return data


def denied(decision):
    return decision is not None and decision["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("key", sorted(ARTIFACT_REGISTRY))
def test_artifact_paths_denied(trial, key):
    rel = ARTIFACT_REGISTRY[key]["path"]
    assert denied(artifacts.validate_write(trial, hook("Write", trial / rel)))
    assert denied(artifacts.validate_write(trial, hook("Edit", rel, cwd=trial)))
    reason = artifacts.validate_write(trial, hook("Write", trial / rel))["hookSpecificOutput"]["permissionDecisionReason"]
    assert "save-artifact" in reason and key in reason


def test_run_state_denied(trial):
    assert denied(artifacts.validate_write(trial, hook("Write", trial / "state" / "run_state.json")))


@pytest.mark.parametrize("rel", ["work/prediction_contract.json", "work/profile.py", "outputs/model/model.joblib"])
def test_work_paths_allowed(trial, rel):
    assert artifacts.validate_write(trial, hook("Write", trial / rel)) is None


def test_non_write_tools_ignored(trial):
    assert artifacts.validate_write(trial, hook("Read", trial / "prediction_contract.json")) is None


def test_reverse_lookup_normalises_paths(trial):
    assert artifact_key_for_path(trial, trial / "state" / ".." / "prediction_contract.json") == "prediction_contract"
    assert artifact_key_for_path(trial, "outputs/report/evaluation_summary.json") == "evaluation_summary"
    assert artifact_key_for_path(trial, "work/evaluation_summary.json") is None


def test_validate_write_cli(trial, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook("Write", trial / "prediction_contract.json"))))
    assert artifacts.main(["validate-write", "--trial", str(trial)]) == 0
    assert json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecision"] == "deny"

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook("Write", trial / "work" / "x.json"))))
    assert artifacts.main(["validate-write", "--trial", str(trial)]) == 0
    assert capsys.readouterr().out == ""


def test_validate_write_cli_malformed_input_allows(trial, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert artifacts.main(["validate-write", "--trial", str(trial)]) == 0
    assert capsys.readouterr().out == ""
