import json

import pytest
from jsonschema import Draft7Validator

from agentic_automl import artifacts
from agentic_automl.contracts import ARTIFACT_REGISTRY, claude_template_dir, schema_path, schemas_dir
from conftest import INVALID_MUTATIONS, VALID_PAYLOADS, valid


def test_every_registered_artifact_has_schema_and_fixtures():
    assert set(VALID_PAYLOADS) == set(ARTIFACT_REGISTRY)
    assert set(INVALID_MUTATIONS) == set(ARTIFACT_REGISTRY)
    for key in ARTIFACT_REGISTRY:
        assert schema_path(key).is_file(), key


@pytest.mark.parametrize("path", sorted(p.name for p in schemas_dir().glob("*.schema.json")))
def test_schemas_are_valid_draft7(path):
    Draft7Validator.check_schema(json.loads((schemas_dir() / path).read_text(encoding="utf-8")))


@pytest.mark.parametrize("key", sorted(ARTIFACT_REGISTRY))
def test_good_payload_validates(key):
    assert artifacts.validate_payload(key, valid(key)) == []


@pytest.mark.parametrize("key", sorted(ARTIFACT_REGISTRY))
def test_bad_payload_rejected(key):
    assert artifacts.validate_payload(key, valid(key, **INVALID_MUTATIONS[key]))


@pytest.mark.parametrize("key", sorted(ARTIFACT_REGISTRY))
def test_closed_schemas_reject_unknown_fields(key):
    assert artifacts.validate_payload(key, valid(key, surprise_field=1))


def test_contract_missing_required_field():
    payload = valid("prediction_contract")
    del payload["scale_assumption"]
    assert any("scale_assumption" in e for e in artifacts.validate_payload("prediction_contract", payload))


def test_contract_task_type_is_free_string():
    assert artifacts.validate_payload("prediction_contract", valid("prediction_contract", task_type="anything")) == []
    assert artifacts.validate_payload("prediction_contract", valid("prediction_contract", task_type=""))


@pytest.mark.parametrize(
    "overrides,ok",
    [
        ({"dataset_location": "gcs", "dataset_path": "gs://bucket/x.parquet"}, True),
        ({"dataset_location": "gcs", "dataset_path": "data/x.csv"}, False),
        ({"dataset_location": "local", "dataset_path": "gs://bucket/x.parquet"}, False),
        # split_strategy ties
        ({"split_strategy": "predefined_column", "split_ratio": None}, False),
        ({"split_strategy": "predefined_column", "split_column": "fold", "split_ratio": None,
          "split_train_value": "train", "split_test_value": "test"}, True),
        ({"split_strategy": "predefined_column", "split_column": "fold",
          "split_train_value": "train", "split_test_value": "test"}, False),  # ratio must be null
        ({"split_strategy": "temporal"}, False),
        ({"split_strategy": "temporal", "time_column": "event_date"}, True),
        ({"split_strategy": "temporal", "time_column": "event_date", "split_column": "fold"}, False),
        ({"split_strategy": "random", "split_ratio": None}, False),
        ({"split_strategy": "random", "time_column": "event_date"}, False),
        ({"split_strategy": "pre_split", "split_ratio": None}, True),
        ({"split_strategy": "pre_split"}, False),  # ratio must be null
        # target_labels
        ({"target_kind": "binary", "target_labels": [0, 1, 2]}, False),
        ({"target_kind": "multiclass", "target_labels": None}, False),
        ({"target_kind": "multiclass", "target_labels": ["a", "b"]}, False),
        ({"target_kind": "multiclass", "target_labels": ["a", "b", "c"]}, True),
        ({"target_kind": "continuous", "target_labels": None}, True),
        ({"target_kind": "continuous", "target_labels": [0, 1]}, False),
        # data_segments
        ({"data_segments": []}, False),
        ({"data_segments": [{"path": "tr.csv", "format": "csv", "role": "train", "file_count": 1},
                            {"path": "te.csv", "format": "csv", "role": "test", "file_count": 1}]}, True),
        ({"data_segments": [{"path": "tr.csv", "format": "csv", "role": "train", "file_count": 1}]}, False),
        ({"data_segments": [{"path": "a.csv", "format": "csv", "role": "full", "file_count": 1},
                            {"path": "tr.csv", "format": "csv", "role": "train", "file_count": 1},
                            {"path": "te.csv", "format": "csv", "role": "test", "file_count": 1}]}, False),
        ({"data_segments": [{"path": "a.csv", "format": "csv", "role": "holdout", "file_count": 1}]}, False),
        # model_space vs model_preference x deployment_intent
        ({"model_space": ["lightgbm"]}, False),  # exploratory + single_baseline -> null
        ({"deployment_intent": "production_parity", "model_space": ["lightgbm"]}, True),
        ({"model_preference": "user_specified", "model_space": None}, False),
        ({"model_preference": "user_specified", "model_space": []}, False),
        ({"model_preference": "user_specified", "model_space": ["xgboost"]}, True),
        ({"model_preference": "auto_research", "model_space": None}, True),
        ({"model_preference": "auto_research", "model_space": ["lightgbm"]}, True),
        ({"model_preference": "gradient_boosting"}, False),
        # enums + optional fields
        ({"scale_assumption": "single_machine"}, False),
        ({"ml_domain": "credit_risk"}, True),
        ({"ml_domain": "fraud"}, False),
        ({"research_max_experiments": 15}, True),
        ({"research_max_experiments": 0}, False),
    ],
)
def test_contract_conditional_rules(overrides, ok):
    errors = artifacts.validate_payload("prediction_contract", valid("prediction_contract", **overrides))
    assert (errors == []) is ok, errors


def test_ml_domain_enums_match_registry():
    registry = json.loads((claude_template_dir() / "skills" / "domains" / "registry.json").read_text(encoding="utf-8"))
    domains = set(registry["domains"])
    for key in ("prediction_contract", "workflow_plan"):
        schema = json.loads(schema_path(key).read_text(encoding="utf-8"))
        assert set(schema["properties"]["ml_domain"]["enum"]) - {None} == domains, key


def test_run_state_validates(trial):
    assert artifacts.validate_payload("run_state", artifacts.load_state(trial)) == []
