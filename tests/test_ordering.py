from agentic_automl import artifacts
from conftest import plan_with_engine, save, save_through, valid


def test_begin_skill_allowed_for_first_step(trial):
    result = artifacts.begin_skill(trial, "prediction-contract")
    assert result["allowed"] is True and result["redirect_to"] is None
    assert artifacts.load_state(trial)["active_skill"] == "prediction-contract"


def test_begin_skill_redirects_to_earliest_unmet(trial):
    result = artifacts.begin_skill(trial, "evaluation")
    assert result["allowed"] is False
    assert result["redirect_to"] == "prediction-contract"
    assert result["unmet_gates"] == ["training_complete"]

    save_through(trial, "dataset_profile")
    assert artifacts.begin_skill(trial, "workflow-planning")["redirect_to"] == "data-profiling"

    save(trial, "column_profile")
    assert artifacts.begin_skill(trial, "training")["redirect_to"] == "workflow-planning"


def test_begin_skill_redirect_follows_preprocessing_engine(trial):
    save_through(trial, "column_profile")
    save(trial, "workflow_plan", plan_with_engine("parity-ml"))
    assert artifacts.begin_skill(trial, "training")["redirect_to"] == "parity-ml-preprocessing"


def test_preprocessing_engine_defaults_to_parity_ml(trial):
    save_through(trial, "column_profile")
    save(trial, "workflow_plan", plan_with_engine(None))
    assert artifacts.next_step(trial)["next_skill"] == "parity-ml-preprocessing"


def test_begin_skill_rerun_reports_downstream(trial):
    save_through(trial, "workflow_plan")
    result = artifacts.begin_skill(trial, "prediction-contract")
    assert result["allowed"] and result["rerun"]
    assert result["downstream_artifacts_present"] == ["column_profile", "dataset_profile", "workflow_plan"]


def test_begin_skill_unknown(trial):
    try:
        artifacts.begin_skill(trial, "nope")
    except artifacts.ArtifactError as exc:
        assert "known_skills" in exc.payload
    else:
        raise AssertionError("expected ArtifactError")


def test_next_step_walks_chain(trial):
    expected = [
        ("prediction_contract", "data-profiling"),
        ("dataset_profile", "data-profiling"),
        ("column_profile", "workflow-planning"),
        ("workflow_plan", "data-preprocessing"),
        ("preprocessing_summary", "training"),
        ("training_summary", "evaluation"),
    ]
    assert artifacts.next_step(trial)["next_skill"] == "prediction-contract"
    for key, nxt in expected:
        save(trial, key)
        assert artifacts.next_step(trial)["next_skill"] == nxt, key
    save(trial, "evaluation_summary")
    done = artifacts.next_step(trial)
    assert done["next_skill"] is None
    assert done["canonical_workflow_complete"] is True
    assert done["optional_next_skills"] == ["explainability", "model-development-documentation"]
    save(trial, "explainability_summary")
    assert artifacts.next_step(trial)["optional_next_skills"] == ["model-development-documentation"]


def test_next_step_injects_auto_research(trial):
    save(trial, "prediction_contract", valid("prediction_contract", model_preference="auto_research"))
    for key in ("dataset_profile", "column_profile", "workflow_plan", "preprocessing_summary", "training_summary"):
        save(trial, key)
    assert artifacts.next_step(trial)["next_skill"] == "auto-research"
    save(trial, "auto_research_summary")
    assert artifacts.next_step(trial)["next_skill"] == "evaluation"


def test_next_step_suggests_parity_export(trial):
    save_through(trial, "column_profile")
    save(trial, "workflow_plan", plan_with_engine("parity-ml"))
    assert artifacts.next_step(trial)["next_skill"] == "parity-ml-preprocessing"
    save(trial, "preprocessing_summary", valid("preprocessing_summary", engine="parity-ml"),
         "parity-ml-preprocessing")
    save(trial, "training_summary")
    assert artifacts.next_step(trial)["next_skill"] == "parity-ml-export"
    save(trial, "onnx_deployment")
    assert artifacts.next_step(trial)["next_skill"] == "evaluation"
