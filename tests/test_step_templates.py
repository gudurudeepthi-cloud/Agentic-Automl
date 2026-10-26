"""Step templates (repo scripts/steps/, shipped into trials as templates/): present, compiled,
referenced by their skills, copied into trials without the contributor scripts, and (when a runtime Python
with the modeling stack is available) runnable end to end."""
import json
import py_compile
import random
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import jsonschema
import pytest

from agentic_automl import cli
from agentic_automl import state as st
from agentic_automl.contracts import claude_template_dir, schema_path, step_templates_dir

SCRIPTS = step_templates_dir()
SKILLS = claude_template_dir() / "skills"
TEMPLATES = {
    "inspect_dataset.py": "prediction-contract",
    "profile_dataset.py": "data-profiling",
    "model_family_screen.py": "workflow-planning",
    "vintage_analysis.py": "workflow-planning",
    "features_template.py": "workflow-planning",
    "preprocess.py": "data-preprocessing",
    "feature_selection.py": "training",
    "train.py": "training",
    "evaluate.py": "evaluation",
}
# Resolved at import, before the autouse fixture isolates AGENTIC_AUTOML_HOME.
RUNTIME = st.runtime_python()


@pytest.mark.parametrize("name", [*TEMPLATES, "aaml_common.py"])
def test_template_compiles(name, tmp_path):
    py_compile.compile(str(SCRIPTS / name), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.parametrize("name,skill", TEMPLATES.items())
def test_skill_points_at_its_template(name, skill):
    assert f"templates/{name}" in (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    assert f"`{name}`" in (SCRIPTS / "README.md").read_text(encoding="utf-8")


def test_templates_ship_into_new_trials(capsys):
    cli.main(["new", "--no-launch"])
    trial = Path(json.loads(capsys.readouterr().out)["trial_dir"])
    for name in [*TEMPLATES, "aaml_common.py", "README.md"]:
        assert (trial / "templates" / name).is_file(), name
    # contributor tools (the data generator knows the answer key) must never reach a trial
    assert not (trial / "templates" / "generate_synthetic_fraud.py").exists()
    assert not list((trial / "templates").glob("run_agent_evals*"))


# --------------------------------------------------------------------------- end to end

def _has_stack(python: str) -> bool:
    try:
        return subprocess.run([python, "-c", "import sklearn, pandas, pyarrow, matplotlib, joblib"],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_runtime = pytest.mark.skipif(not _has_stack(RUNTIME), reason=f"runtime python {RUNTIME!r} lacks the ML stack")


def _dataset(path: Path, rows: int = 3000) -> None:
    rng = random.Random(0)
    t0 = datetime(2025, 1, 1)
    lines = ["row_id,entity,event_time,x1,x2,x3,region,const,leak,label"]
    for i in range(rows):
        x1, x2, x3 = rng.gauss(0, 1), rng.gauss(0, 1), rng.expovariate(0.5)
        region = rng.choice(["n", "s", "e", "w"])
        score = 1.5 * x1 + 0.5 * x3 + (0.8 if region == "n" else 0) - 2.5 + rng.gauss(0, 1)
        y = int(score > 0)
        x2s = "" if rng.random() < 0.05 else f"{x2:.4f}"
        ts = (t0 + timedelta(minutes=37 * i)).isoformat(sep=" ")
        lines.append(f"{i},{rng.randrange(300)},{ts},{x1:.4f},{x2s},{x3:.4f},{region},K,{y},{y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _contract(data: str, split: str) -> dict:
    temporal = split == "temporal"
    return {
        "task_type": "t", "target_kind": "binary", "business_context": "test",
        "deployment_intent": "exploratory_baseline", "dataset_path": data, "dataset_location": "local",
        "data_source": "internal",
        "data_segments": [{"path": data, "format": "csv", "file_count": 1, "role": "full"}],
        "schema_files": [], "split_strategy": split, "split_column": None, "split_train_value": None,
        "split_test_value": None, "time_column": "event_time" if temporal else None, "split_ratio": 0.25,
        "target_column": "label", "target_labels": [0, 1], "entity_id_column": "entity", "weight_column": None,
        "primary_metric": "pr_auc", "report_metrics": ["roc_auc", "precision_at_top_5pct"],
        "model_preference": "single_baseline", "model_space": None, "scale_assumption": "local_workstation",
        # intake answers: x3 carries signal but the user ruled it out; x2 is noise the user wants kept
        "excluded_columns": [{"column": "x3", "reason": "recorded after the outcome"}],
        "required_features": ["x2"],
    }


def _run(trial: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([RUNTIME, f"work/{script}", *args], cwd=trial, capture_output=True, text=True,
                          timeout=600)


def _ok(trial, script, *args):
    r = _run(trial, script, *args)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def _promote(trial: Path, key: str, dest: str) -> None:
    """Stand-in for save-artifact: schema-check the draft and copy it to its registered path."""
    payload = json.loads((trial / "work" / f"{key}.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, json.loads(schema_path(key).read_text(encoding="utf-8")))
    (trial / dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(trial / "work" / f"{key}.json", trial / dest)


@needs_runtime
@pytest.mark.parametrize("split", ["temporal", "random_stratified"])
def test_templates_run_end_to_end(tmp_path, split):
    trial = tmp_path / "trial"
    for d in ("work", "state", "outputs/model", "outputs/pipeline", "outputs/report", "preprocessed_data"):
        (trial / d).mkdir(parents=True)
    for f in SCRIPTS.glob("*.py"):
        shutil.copy(f, trial / "work" / f.name)
    (trial / "work" / "features_template.py").unlink()  # no engineered features in this test
    data = tmp_path / "data.csv"
    _dataset(data)
    (trial / "prediction_contract.json").write_text(json.dumps(_contract(data.as_posix(), split)), encoding="utf-8")
    plan = {"objective": "o", "steps": ["s"],
            "modeling_plan": {"primary_metric": "pr_auc", "validation_strategy": "v", "candidate_models": [
                {"name": "logistic_regression", "rationale": "r", "search_space": {"C": [0.3, 3.0]}},
                {"name": "random_forest", "rationale": "r", "search_space": {"n_estimators": 60}}]},
            "preprocessing_plan": {"engine": "open_source",
                                   "drop_columns": ["row_id", "entity", "event_time", "const", "leak"]}}
    (trial / "state" / "workflow_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    out = _ok(trial, "profile_dataset.py")
    assert "[critical] leak alone nearly determines the target" in out
    _promote(trial, "dataset_profile", "dataset_profile.json")
    _promote(trial, "column_profile", "column_profile.json")
    roles = {c["name"]: c["role"] for c in json.loads((trial / "column_profile.json").read_text())["columns"]}
    assert roles["leak"] == "leakage_suspect" and roles["const"] == "drop" and roles["row_id"] == "id"
    assert roles["x3"] == "drop"  # excluded at intake

    advice = (_ok(trial, "model_family_screen.py", "--folds", "3", "--families", "logistic_regression,random_forest"),
              json.loads((trial / "work" / "model_family_advice.json").read_text()))[1]
    assert advice["recommended_family"] in ("logistic_regression", "random_forest")
    assert ("TimeSeriesSplit" in advice["cv"]) == (split == "temporal")
    assert split == "temporal" or "StratifiedGroupKFold" in advice["cv"]

    # vintage selection: analyse train weeks, then exclude the first week (train only)
    out = _ok(trial, "vintage_analysis.py", "--column", "event_time", "--period", "W", "--maturity-days", "3",
              "--holdout-periods", "1", "--drop", "row_id", "entity", "const", "leak", "--no-plot")
    va = json.loads((trial / "work" / "vintage_analysis.json").read_text())
    weeks = [v["vintage"] for v in va["train_vintages"]]
    assert len(weeks) >= 5 and va["window_backtest"]["windows"]["all"]["mean"] > 0
    assert any(f in v["flags"] for v in va["train_vintages"][-1:] for f in ("immature", "partly_immature"))
    first = va["train_vintages"][0]
    plan["vintage_selection"] = {"column": "event_time", "period": "W", "exclude_periods": [weeks[0]],
                                 "rationale": "test: drop the first week"}
    jsonschema.validate(plan, json.loads(schema_path("workflow_plan").read_text(encoding="utf-8")))
    (trial / "state" / "workflow_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    _ok(trial, "preprocess.py")
    pre_sum = json.loads((trial / "work" / "preprocessing_summary.json").read_text())
    prof_split = json.loads((trial / "dataset_profile.json").read_text())["split_summary"]
    assert pre_sum["train_rows"] == prof_split["train"]["rows"] - first["rows"]
    assert pre_sum["test_rows"] == prof_split["test"]["rows"]  # test is never filtered
    assert "Vintage selection" in pre_sum["notes"]
    _promote(trial, "preprocessing_summary", "outputs/pipeline/preprocessing_summary.json")
    assert "estimated" in _ok(trial, "feature_selection.py", "--folds", "3", "--smoke")
    out = _ok(trial, "feature_selection.py", "--folds", "3", "--filters-only",
              "--filters", "quasi_constant,bivariate,mutual_info,vif,l1")
    assert "feature_selection.json not written" in out
    assert not (trial / "outputs/model/feature_selection.json").exists()
    biv = json.loads((trial / "outputs/model/bivariate.json").read_text())
    strength = {b["feature"]: b["strength"] for b in biv["features"]}
    assert biv["features"][0]["statistic"] == "information_value" and strength["x1"] > strength["x2"]
    train_rows = pre_sum["train_rows"]  # after vintage selection
    assert all(sum(r["n"] for r in b["bins"]) == train_rows for b in biv["features"])  # every row in one bin
    x2_bins = next(b for b in biv["features"] if b["feature"] == "x2")["bins"]
    assert x2_bins[-1]["bin"] == "<missing>"  # 5% blanks get their own bin
    _ok(trial, "feature_selection.py", "--folds", "3")
    selection = json.loads((trial / "outputs/model/feature_selection.json").read_text())
    assert "x2" not in selection["filters"]["removed"]["bivariate"]  # noise, but required by the contract
    assert "x3" not in selection["candidate_pool"] and selection["required_by_contract"] == ["x2"]
    assert all("x2" in fam["features"] for fam in selection["families"].values())
    _ok(trial, "train.py", "--folds", "3")
    _promote(trial, "training_summary", "outputs/model/training_summary.json")
    assert (trial / "outputs/model/cv_predictions.parquet").is_file()

    _ok(trial, "evaluate.py", "--threshold-rule", "f1")
    _promote(trial, "evaluation_summary", "outputs/report/evaluation_summary.json")
    summary = json.loads((trial / "outputs/report/evaluation_summary.json").read_text())
    checks = {c["name"]: c["passed"] for c in summary["trust_checks"]}
    assert checks["leakage"] and checks["split_integrity"]
    assert summary["threshold"] is not None and summary["metrics"]["pr_auc_ci_low"] is not None

    again = _run(trial, "evaluate.py")  # evaluate-once guard
    assert again.returncode != 0 and "--rerun-reason" in (again.stdout + again.stderr)


@needs_runtime
def test_inspect_dataset_proposes_intake_answers(tmp_path):
    trial = tmp_path / "trial"
    (trial / "work").mkdir(parents=True)
    for name in ("inspect_dataset.py", "aaml_common.py"):
        shutil.copy(SCRIPTS / name, trial / "work" / name)
    data = tmp_path / "data.csv"
    _dataset(data)
    _ok(trial, "inspect_dataset.py", data.as_posix())
    cand = json.loads((trial / "work" / "intake_candidates.json").read_text())["candidates"]
    assert cand["time"] == ["event_time"] and cand["row_id"] == ["row_id"]
    assert "label" in {t["column"] for t in cand["target"]}  # a candidate, for the user to name or not


def test_contract_rejects_contradictory_intake_answers():
    from agentic_automl.artifacts import validate_payload

    c = _contract("d.csv", "random_stratified")
    assert validate_payload("prediction_contract", c) == []
    bad = {**c, "excluded_columns": [{"column": "label", "reason": "r"}, {"column": "x2", "reason": "r"}],
           "required_features": ["x2", "entity"]}
    errors = " ".join(validate_payload("prediction_contract", bad))
    assert "target 'label' cannot be excluded" in errors
    assert "['x2'] are also in excluded_columns" in errors and "['entity'] are target" in errors
