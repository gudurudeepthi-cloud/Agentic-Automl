import json

import pytest

from agentic_automl import artifacts
from agentic_automl.contracts import (
    ARTIFACT_GATE_REQUIREMENTS,
    ARTIFACT_REGISTRY,
    GATE_REGISTRY,
    SKILL_PRODUCES,
    WORKFLOW_ORDER,
    artifact_path,
)
from conftest import save, save_through, valid


def test_registries_are_consistent():
    assert set(ARTIFACT_GATE_REQUIREMENTS) == set(ARTIFACT_REGISTRY)
    assert {g["satisfied_by"] for g in GATE_REGISTRY.values()} == set(ARTIFACT_REGISTRY)
    for reqs in ARTIFACT_GATE_REQUIREMENTS.values():
        assert set(reqs) <= set(GATE_REGISTRY)
    for keys in SKILL_PRODUCES.values():
        assert set(keys) <= set(ARTIFACT_REGISTRY)
    assert set(WORKFLOW_ORDER) <= set(SKILL_PRODUCES)


def test_new_trial_has_all_gates_false(trial):
    assert not any(artifacts.evaluate_gates(trial).values())


def test_save_contract_flips_gate_and_writes_canonical_file(trial):
    result = save(trial, "prediction_contract")
    assert result["ok"] and result["gate_satisfied"] == "prediction_contract_approved"
    assert artifact_path(trial, "prediction_contract").is_file()
    state = artifacts.load_state(trial)
    assert state["gates"]["prediction_contract_approved"] is True
    assert [a["key"] for a in state["artifacts"]] == ["prediction_contract"]
    assert state["validation_log"][-1]["ok"] is True


@pytest.mark.parametrize(
    "key,missing_gate,redirect",
    [
        ("dataset_profile", "prediction_contract_approved", "prediction-contract"),
        ("workflow_plan", "dataset_profile_complete", "prediction-contract"),
        ("training_summary", "preprocessing_complete", "prediction-contract"),
        ("model_development_documentation", "evaluation_complete", "prediction-contract"),
    ],
)
def test_save_rejected_when_upstream_gate_missing(trial, key, missing_gate, redirect):
    with pytest.raises(artifacts.ArtifactError) as exc:
        save(trial, key)
    assert missing_gate in exc.value.payload["unmet_gates"]
    assert exc.value.payload["redirect_to"] == redirect
    assert not artifact_path(trial, key).exists()
    assert artifacts.load_state(trial)["validation_log"][-1]["ok"] is False


def test_workflow_plan_needs_both_profiles(trial):
    save(trial, "prediction_contract")
    save(trial, "dataset_profile")
    with pytest.raises(artifacts.ArtifactError) as exc:
        save(trial, "workflow_plan")
    assert exc.value.payload["unmet_gates"] == ["column_profile_complete"]
    assert exc.value.payload["redirect_to"] == "data-profiling"
    save(trial, "column_profile")
    assert save(trial, "workflow_plan")["ok"]


def test_schema_failure_rejected_and_not_written(trial):
    with pytest.raises(artifacts.ArtifactError) as exc:
        save(trial, "prediction_contract", valid("prediction_contract", target_kind="nope"))
    assert exc.value.payload["errors"]
    assert not artifact_path(trial, "prediction_contract").exists()


def test_wrong_source_skill_rejected(trial):
    with pytest.raises(artifacts.ArtifactError, match="may not write"):
        save(trial, "prediction_contract", skill="training")


def test_preprocessing_engine_must_match_skill(trial):
    save_through(trial, "workflow_plan")
    with pytest.raises(artifacts.ArtifactError, match="engine"):
        save(trial, "preprocessing_summary", skill="parity-ml-preprocessing")
    assert save(trial, "preprocessing_summary",
                valid("preprocessing_summary", engine="parity-ml"), "parity-ml-preprocessing")["ok"]


def test_full_chain_and_optional_artifacts(trial):
    save_through(trial, "evaluation_summary")
    for key in ("explainability_summary", "model_development_documentation", "onnx_deployment",
                "auto_research_summary"):
        assert save(trial, key)["ok"]
    assert all(artifacts.evaluate_gates(trial).values())


def test_resave_bumps_version(trial):
    save(trial, "prediction_contract")
    result = save(trial, "prediction_contract", valid("prediction_contract", notes="v2"))
    assert result["version"] == 2
    state = artifacts.load_state(trial)
    assert len([a for a in state["artifacts"] if a["key"] == "prediction_contract"]) == 1


def test_deleted_artifact_file_unsets_gate(trial):
    save(trial, "prediction_contract")
    artifact_path(trial, "prediction_contract").unlink()
    assert artifacts.evaluate_gates(trial)["prediction_contract_approved"] is False


def test_hand_placed_file_does_not_satisfy_gate(trial):
    path = artifact_path(trial, "prediction_contract")
    path.write_text(json.dumps(valid("prediction_contract")), encoding="utf-8")
    assert artifacts.evaluate_gates(trial)["prediction_contract_approved"] is False


def test_decision_feedback_diff_logged(trial):
    proposal = artifacts.decision_proposal_path(trial, "prediction_contract")
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(json.dumps(valid("prediction_contract")), encoding="utf-8")
    result = save(trial, "prediction_contract", valid("prediction_contract", primary_metric="pr_auc"))
    assert result["decision_feedback"]["n_changes"] == 1
    episodes = (trial / "work" / "feedback_episodes.jsonl").read_text(encoding="utf-8").splitlines()
    episode = json.loads(episodes[-1])
    assert episode["type"] == "decision"
    assert episode["changes"][0]["path"] == "$.primary_metric"
    assert not proposal.exists()
