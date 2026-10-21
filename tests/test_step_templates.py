"""Step templates in .claude/scripts/: present, compiled, referenced by their skills, shipped into trials,
and (when a runtime Python with the modeling stack is available) runnable end to end."""
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
from agentic_automl.contracts import claude_template_dir, schema_path

SCRIPTS = claude_template_dir() / "scripts"
SKILLS = claude_template_dir() / "skills"
TEMPLATES = {
    "profile_dataset.py": "data-profiling",
    "model_family_screen.py": "workflow-planning",
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
    assert f".claude/scripts/{name}" in (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    assert f"`{name}`" in (SCRIPTS / "README.md").read_text(encoding="utf-8")


def test_templates_ship_into_new_trials(capsys):
    cli.main(["new", "--no-launch"])
    trial = Path(json.loads(capsys.readouterr().out)["trial_dir"])
    for name in [*TEMPLATES, "aaml_common.py", "README.md"]:
        assert (trial / ".claude" / "scripts" / name).is_file(), name


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

    advice = (_ok(trial, "model_family_screen.py", "--folds", "3", "--families", "logistic_regression,random_forest"),
              json.loads((trial / "work" / "model_family_advice.json").read_text()))[1]
    assert advice["recommended_family"] in ("logistic_regression", "random_forest")
    assert ("TimeSeriesSplit" in advice["cv"]) == (split == "temporal")
    assert split == "temporal" or "StratifiedGroupKFold" in advice["cv"]

    _ok(trial, "preprocess.py")
    _promote(trial, "preprocessing_summary", "outputs/pipeline/preprocessing_summary.json")
    assert "estimated" in _ok(trial, "feature_selection.py", "--folds", "3", "--smoke")
    _ok(trial, "feature_selection.py", "--folds", "3")
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
