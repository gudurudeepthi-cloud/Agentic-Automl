from __future__ import annotations

import copy
from pathlib import Path

import pytest

from agentic_automl import artifacts

VALID_PAYLOADS: dict[str, dict] = {
    "prediction_contract": {
        "task_type": "customer churn classification",
        "target_kind": "binary",
        "business_context": "Prioritise retention outreach.",
        "deployment_intent": "exploratory_baseline",
        "dataset_path": "data/churn.csv",
        "dataset_location": "local",
        "data_source": "internal",
        "data_segments": [{"path": "data/churn.csv", "format": "csv", "role": "full", "file_count": 1}],
        "schema_files": [],
        "split_strategy": "random_stratified",
        "split_column": None,
        "split_train_value": None,
        "split_test_value": None,
        "time_column": None,
        "split_ratio": 0.2,
        "target_column": "churned",
        "target_labels": [0, 1],
        "entity_id_column": "customer_id",
        "weight_column": None,
        "primary_metric": "roc_auc",
        "report_metrics": ["roc_auc", "pr_auc", "f1"],
        "model_preference": "single_baseline",
        "model_space": None,
        "scale_assumption": "local_workstation",
    },
    "dataset_profile": {
        "dataset_path": "data/churn.csv",
        "n_rows": 1000,
        "n_columns": 12,
        "target_column": "churned",
        "target_summary": {"0": 800, "1": 200},
        "split_summary": {"train": 800, "test": 200},
        "missing_value_summary": {"tenure": 0.01},
        "duplicate_rows": 0,
        "data_quality_flags": [{"severity": "info", "message": "No issues"}],
        "profiling_script": "work/profile.py",
    },
    "column_profile": {
        "dataset_path": "data/churn.csv",
        "columns": [
            {"name": "tenure", "dtype": "int64", "semantic_type": "numeric", "role": "feature",
             "missing_fraction": 0.01, "n_unique": 72},
            {"name": "churned", "dtype": "int64", "semantic_type": "binary", "role": "target",
             "missing_fraction": 0.0, "n_unique": 2},
        ],
    },
    "workflow_plan": {
        "objective": "Baseline churn model",
        "modeling_plan": {
            "candidate_models": [{"name": "lightgbm", "rationale": "Strong tabular baseline"}],
            "validation_strategy": "5-fold stratified CV",
            "primary_metric": "roc_auc",
        },
        "preprocessing_plan": {"engine": "open_source", "drop_columns": ["customer_id"]},
        "steps": ["preprocess", "train", "evaluate"],
    },
    "preprocessing_summary": {
        "engine": "open_source",
        "pipeline_script": "work/preprocess.py",
        "pipeline_artifact": "outputs/pipeline/preprocessor.joblib",
        "input_features": ["tenure"],
        "output_feature_count": 1,
        "transformations": [{"step": "impute", "description": "median impute"}],
        "dropped_columns": ["customer_id"],
        "train_rows": 800,
        "test_rows": 200,
    },
    "training_summary": {
        "models": [{"name": "lightgbm", "library": "lightgbm", "hyperparameters": {}, "cv_score": 0.81}],
        "selected_model": "lightgbm",
        "selection_rationale": "Best CV AUC",
        "primary_metric": "roc_auc",
        "cv_strategy": "5-fold stratified",
        "training_script": "work/train.py",
        "model_artifact_path": "outputs/model/model.joblib",
    },
    "onnx_deployment": {
        "onnx_path": "outputs/model/model.onnx",
        "opset": 17,
        "input_signature": [{"name": "input", "dtype": "float32"}],
        "parity_check": {"passed": True, "max_abs_diff": 1e-6, "n_samples": 100},
        "export_script": "work/export.py",
    },
    "auto_research_summary": {
        "primary_metric": "roc_auc",
        "baseline_metric": 0.81,
        "best_metric": 0.83,
        "iterations": [{"iteration": 1, "hypothesis": "h", "change": "c", "metric_value": 0.83, "kept": True}],
        "best_model_artifact_path": "outputs/model/model_ar.joblib",
        "stop_reason": "budget exhausted",
    },
    "evaluation_summary": {
        "model_evaluated": "lightgbm",
        "split_evaluated": "test",
        "primary_metric": "roc_auc",
        "primary_metric_value": 0.8,
        "metrics": {"roc_auc": 0.8},
        "baseline_comparison": None,
        "trust_checks": [{"name": "leakage", "passed": True, "detail": "No target-derived features"}],
        "narrative": "The model ranks churners well enough to target outreach.",
        "plots": [],
        "findings": ["Model beats prior baseline"],
        "evaluation_script": "work/evaluate.py",
    },
    "explainability_summary": {
        "method": "shap",
        "global_importance": [{"feature": "tenure", "importance": 0.4}],
        "plots": [],
        "findings": ["Tenure dominates"],
        "explainability_script": "work/explain.py",
    },
    "model_development_documentation": {
        "document_path": "outputs/report/mdd.md",
        "summary": "Churn baseline",
        "sections": ["Overview"],
        "limitations": [],
    },
}

# A schema-violating mutation per artifact.
INVALID_MUTATIONS: dict[str, dict] = {
    "prediction_contract": {"target_kind": "regression"},
    "dataset_profile": {"n_rows": -1},
    "column_profile": {"columns": []},
    "workflow_plan": {"preprocessing_plan": {"engine": "spark", "drop_columns": []}},
    "preprocessing_summary": {"output_feature_count": 0},
    "training_summary": {"models": []},
    "onnx_deployment": {"opset": 0},
    "auto_research_summary": {"iterations": []},
    "evaluation_summary": {"primary_metric_value": "high"},
    "explainability_summary": {"global_importance": []},
    "model_development_documentation": {"sections": []},
}


def valid(key: str, **overrides) -> dict:
    payload = copy.deepcopy(VALID_PAYLOADS[key])
    payload.update(overrides)
    return payload


def plan_with_engine(engine: str | None) -> dict:
    payload = valid("workflow_plan")
    if engine is None:
        del payload["preprocessing_plan"]["engine"]
    else:
        payload["preprocessing_plan"]["engine"] = engine
    return payload


@pytest.fixture(autouse=True)
def _isolated_trials_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_AUTOML_TRIALS_DIR", str(tmp_path / "trials"))
    monkeypatch.delenv("AGENTIC_AUTOML_TRIAL", raising=False)
    monkeypatch.delenv("TRIAL_DIR", raising=False)
    monkeypatch.delenv("AGENTIC_AUTOML_STORAGE_DIR", raising=False)
    monkeypatch.delenv("AGENTIC_AUTOML_RESULTS_DIR", raising=False)
    monkeypatch.delenv("AGENTIC_AUTOML_TRUST", raising=False)
    monkeypatch.delenv("AGENTIC_AUTOML_PYTHON", raising=False)
    monkeypatch.setenv("AGENTIC_AUTOML_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.setenv("AGENTIC_AUTOML_CLAUDE", "claude-fake")


@pytest.fixture
def trial(tmp_path) -> Path:
    path = tmp_path / "trials" / "t001"
    artifacts.ensure_state(path, goal="predict churn", dataset_path="data/churn.csv")
    return path


SKILL_FOR = {
    "prediction_contract": "prediction-contract",
    "dataset_profile": "data-profiling",
    "column_profile": "data-profiling",
    "workflow_plan": "workflow-planning",
    "preprocessing_summary": "data-preprocessing",
    "training_summary": "training",
    "onnx_deployment": "parity-ml-export",
    "auto_research_summary": "auto-research",
    "evaluation_summary": "evaluation",
    "explainability_summary": "explainability",
    "model_development_documentation": "model-development-documentation",
}


def save(trial: Path, key: str, payload: dict | None = None, skill: str | None = None) -> dict:
    return artifacts.save_artifact(trial, key, payload or valid(key), skill or SKILL_FOR[key])


def save_through(trial: Path, last_key: str) -> None:
    order = [
        "prediction_contract", "dataset_profile", "column_profile", "workflow_plan",
        "preprocessing_summary", "training_summary", "evaluation_summary",
    ]
    for key in order[: order.index(last_key) + 1]:
        save(trial, key)
