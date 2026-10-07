import json

from agentic_automl import artifacts, cli
from conftest import valid


def run(capsys, *argv):
    code = cli.main(["artifacts", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_save_artifact_via_cli(trial, capsys):
    payload = trial / "work" / "prediction_contract.json"
    payload.write_text(json.dumps(valid("prediction_contract")), encoding="utf-8")
    code, out = run(capsys, "save-artifact", "prediction_contract", "--trial", str(trial),
                    "--payload", str(payload), "--source-skill", "prediction-contract")
    assert code == 0 and out["ok"]
    assert out["next_step"]["next_skill"] == "data-profiling"


def test_save_artifact_cli_rejects_with_nonzero(trial, capsys):
    payload = trial / "work" / "bad.json"
    payload.write_text(json.dumps({"task_type": "x"}), encoding="utf-8")
    code, out = run(capsys, "save-artifact", "prediction_contract", "--trial", str(trial),
                    "--payload", str(payload), "--source-skill", "prediction-contract")
    assert code == 1 and out["ok"] is False and out["errors"]


def test_trial_resolved_by_id(trial, capsys):
    code, out = run(capsys, "status", "--trial", "t001")
    assert code == 0 and out["trial_id"] == "t001" and out["goal"] == "predict churn"


def test_begin_skill_and_next_step_cli(trial, capsys):
    code, out = run(capsys, "begin-skill", "training", "--trial", str(trial))
    assert code == 0 and out["allowed"] is False and out["redirect_to"] == "prediction-contract"
    code, out = run(capsys, "next-step", "--trial", str(trial))
    assert out["next_skill"] == "prediction-contract"


def test_set_trial_slug_and_apply(trial, capsys):
    code, out = run(capsys, "set-trial-slug", "churn-baseline", "--trial", str(trial))
    assert code == 0 and out["pending_slug"] == "churn-baseline"
    assert trial.exists()  # not renamed mid-session
    new_dir = artifacts.apply_pending_slug(trial)
    assert new_dir.name == "t001--churn-baseline" and not trial.exists()
    state = artifacts.load_state(new_dir)
    assert state["pending_slug"] is None and state["trial_id"] == new_dir.name
    assert state["previous_trial_dirs"] == [str(trial.resolve())]


def test_invalid_slug_rejected(trial, capsys):
    code, out = run(capsys, "set-trial-slug", "Bad Slug!", "--trial", str(trial))
    assert code == 1 and out["ok"] is False


def test_developer_focus(trial):
    artifacts.record_developer_focus(trial, "prioritise recall")
    assert artifacts.load_state(trial)["developer_focus"][0]["text"] == "prioritise recall"
