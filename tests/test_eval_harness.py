"""Contributor tooling: eval harness, scorers, reports, recommender, benchmark registry, synthetic data,
seed roles, and the scripts/ wrappers (release helpers included)."""
import importlib.util
import json
import py_compile
import zipfile
from pathlib import Path

import pytest

from agentic_automl import benchmark_readiness as br
from agentic_automl import eval_harness as eh
from agentic_automl import eval_reports as er
from agentic_automl import eval_scorers as es
from agentic_automl import eval_seed_roles as roles
from agentic_automl.contracts import source_root
from agentic_automl.synthetic_benchmarks import SyntheticDatasetConfig, rows_for_config

ROOT = source_root()
SCRIPTS = ROOT / "scripts"
EXPECTED_SCRIPTS = [
    "audit_benchmark_datasets.py", "check_tool_distribution.sh", "download_datasets.py", "generate_answer_bank.py",
    "generate_dataset_seeds.py", "generate_seed_profiles.py", "generate_synthetic_benchmarks.py",
    "generate_training_seed.py", "inspect_benchmark_dataset.py", "inspect_tool_distribution.py",
    "prepare_benchmark_samples.py", "prepare_release.py", "publish_artifactory.sh", "recommend_evals.py",
    "run_agent_evals.py", "summarize_trials.py",
]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- scripts

@pytest.mark.parametrize("name", EXPECTED_SCRIPTS)
def test_script_present(name, tmp_path):
    path = SCRIPTS / name
    assert path.is_file(), name
    if name.endswith(".py"):
        py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)
    else:
        assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert f"`{name}`" in (SCRIPTS / "README.md").read_text(encoding="utf-8")


def test_contributor_scripts_never_reach_trials(capsys):
    from agentic_automl import cli

    cli.main(["new", "--no-launch"])
    trial = Path(json.loads(capsys.readouterr().out)["trial_dir"])
    shipped = {p.name for p in (trial / "templates").iterdir()}
    assert not shipped & {*EXPECTED_SCRIPTS, "generate_synthetic_fraud.py"}


# --------------------------------------------------------------------------- scenarios + deterministic tier

def test_committed_scenarios_validate_and_reference_real_seeds():
    scenarios = eh.load_scenarios()
    assert scenarios
    for s in scenarios:
        if s["seed"]["kind"] == "fixture":
            assert (ROOT / "benchmarks/evals/fixtures/seeds" / s["seed"]["name"] / "state/run_state.json").is_file()
        assert not es.validate_smoke_metric_policy(s), s["id"]
        assert s["answer_bank"], f"{s['id']}: run scripts/generate_answer_bank.py"


def test_deterministic_tier_passes_on_committed_seeds(tmp_path):
    code = eh.main(["--tier", "deterministic", "--output-dir", str(tmp_path), "--report-md", str(tmp_path / "r.md"),
                    "--results-tsv", str(tmp_path / "r.tsv")])
    results = [json.loads(p.read_text()) for p in (tmp_path / "results").glob("*.json")]
    assert code == 0, [(r["scenario_id"], r["status"], r["stop_reason"]) for r in results]
    assert results and all(r["status"] == "pass" for r in results)
    assert "Verdict: PASS" in (tmp_path / "r.md").read_text()
    header, *rows = (tmp_path / "r.tsv").read_text().splitlines()
    assert header.split("\t") == er.TSV_COLUMNS and len(rows) == len(results)
    row = dict(zip(er.TSV_COLUMNS, rows[0].split("\t")))
    assert row["status"] == "pass" and float(row["test_value"]) > 0 and row["selected_model"]


def test_paid_tier_needs_trust_runtime(tmp_path):
    assert eh.main(["--tier", "e2e", "--output-dir", str(tmp_path)]) == 2
    assert not (tmp_path / "results").exists()


def test_prompt_includes_answer_bank_with_repo_root():
    s = next(x for x in eh.load_scenarios() if x["tiers"] != ["deterministic"])
    prompt = eh.build_prompt(s, "data.csv")
    assert "headless" in prompt and s["entry_skill"] in prompt
    assert eh.REPO_PLACEHOLDER not in prompt


# --------------------------------------------------------------------------- scorers

def _trial(tmp_path, evaluation=None):
    t = tmp_path / "trial"
    (t / "outputs/report").mkdir(parents=True)
    (t / "state").mkdir()
    (t / "state/run_state.json").write_text(json.dumps({"validation_log": [
        {"ok": False, "key": "evaluation_summary"}, {"ok": True, "key": "evaluation_summary"}]}))
    if evaluation is not None:
        (t / "outputs/report/evaluation_summary.json").write_text(json.dumps(evaluation))
    return t


def test_scorers(tmp_path):
    t = _trial(tmp_path, {"primary_metric_value": 0.6, "trust_checks": [{"name": "leakage", "passed": True}]})
    scenario = {"expected_artifacts": ["evaluation_summary"], "invariants": [
        {"type": "metric_band", "artifact": "evaluation_summary", "metric": "primary_metric_value", "min": 0.5, "max": 0.9},
        {"type": "metric_band", "name": "too_high", "artifact": "evaluation_summary", "metric": "primary_metric_value",
         "min": 0.7},
        {"type": "file_contains", "path": "outputs/report/evaluation_summary.json",
         "patterns": [r'"name": "leakage", "passed": true'], "absent_patterns": ["chargeback"]},
        {"type": "max_validation_failures", "name": "fails", "max_count": 0},
        {"type": "file_absent", "path": "work/ANSWER_KEY.md"},
    ]}
    r = es.score_trial(scenario, t)
    by = {s.name: s.passed for s in r.scores}
    assert by["expected_artifact:evaluation_summary"] and by["metric_band:evaluation_summary"]
    assert not by["too_high"] and not by["fails"] and not r.passed
    assert r.process_metrics == {"validation_failures": 1, "retries": 1}


def test_credit_marker_and_band_policy():
    assert es.parse_credit_marker('x <!-- aaml-feedback-credit:v1 {"layer": "skill", "summary": "s"} -->')
    assert es.parse_credit_marker("<!-- aaml-feedback-credit:v1 {bad json} -->") is None
    smoke = {"data_size": "smoke", "invariants": [{"type": "metric_band", "metric": "m", "min": 0.8, "max": 0.9}]}
    assert es.validate_smoke_metric_policy(smoke)


def test_all_skipped_is_not_a_pass():
    skipped = [{"status": "environment-unavailable", "scores": [], "scenario_id": "a", "elapsed_seconds": 0}]
    assert er.verdict(skipped)[0] is False
    assert er.verdict([])[0] is False
    mixed = skipped + [{"status": "pass", "scores": [], "scenario_id": "b", "elapsed_seconds": 0}]
    assert er.verdict(mixed)[0] is True


# --------------------------------------------------------------------------- recommender

def test_recommender_maps_paths_to_tiers():
    recs = {(r["level"], r["tier"]): r for r in eh.recommend([
        ".claude/skills/evaluation/SKILL.md", "schemas/prediction_contract.schema.json", "scripts/steps/train.py"])}
    assert ("MUST", "deterministic") in recs and ("CHECK", "e2e") in recs
    comp = recs[("MUST", "component")]
    assert comp["scenarios"] and all(s.endswith("_evaluation") for s in comp["scenarios"])
    assert eh.recommend(["README.md", "tests/test_cli.py"]) == []


# --------------------------------------------------------------------------- benchmarks

def test_registry_audit(tmp_path):
    results = {r.id: r.status for r in br.audit_benchmark_datasets()}
    assert results["synthetic_fraud"] == br.ReadinessStatus.READY_FOR_SCORED_EVAL
    assert results["give_me_some_credit"] == br.ReadinessStatus.METADATA_DRAFT
    base = {"id": "x", "status": "active", "access": "gcs", "brief": "b", "dataset_format": "csv",
            "task_type": "binary", "target_variable": "y", "primary_metric": "roc_auc"}
    assert br.audit_entry(base, tmp_path).status == br.ReadinessStatus.BLOCKED_GCS
    assert br.audit_entry({**base, "owner": "team", "owner_confirmed": False}, tmp_path).status \
        == br.ReadinessStatus.BLOCKED_OWNER_CONFIRMATION
    assert br.audit_entry({**base, "task_type": None}, tmp_path).status == br.ReadinessStatus.BLOCKED_METADATA


def test_seed_roles_match_registry():
    entry = roles.registry_entry("synthetic_fraud")
    prof = json.loads((ROOT / "benchmarks/evals/fixtures/seeds/synthetic_fraud_after_profiling/column_profile.json")
                      .read_text(encoding="utf-8"))
    roles.assert_roles_match_golden(prof, entry)
    prof["columns"] = [{**c, "role": "feature"} if c["name"] == "transaction_id" else c for c in prof["columns"]]
    with pytest.raises(AssertionError, match="transaction_id"):
        roles.assert_roles_match_golden(prof, entry)


def test_synthetic_benchmarks_deterministic():
    cfg = SyntheticDatasetConfig.from_dict({
        "name": "t", "rows": 300, "seed": 3, "target": {"name": "y", "kind": "binary", "base_rate": 0.3},
        "columns": [{"name": "id", "kind": "integer", "role": "id"},
                    {"name": "x", "kind": "numeric", "weight": 2.0, "missing": 0.1},
                    {"name": "c", "kind": "categorical", "levels": ["a", "b"], "level_weights": [1, -1]}]})
    a, b = rows_for_config(cfg), rows_for_config(cfg)
    assert a == b and len(a) == 300 and {r["y"] for r in a} == {0, 1}
    assert any(r["x"] == "" for r in a) and a[0]["id"] == 1


# --------------------------------------------------------------------------- release helpers

def test_prepare_release_bumps():
    bump = _load_script("prepare_release.py").bump
    assert bump("0.1.0", "patch", None, False, False) == "0.1.1"
    assert bump("0.1.3", "minor", None, False, False) == "0.2.0"
    assert bump("0.4.2", "major", None, False, False) == "1.0.0"
    assert bump("0.1.0", "patch", "rc", False, False) == "0.1.1rc1"
    assert bump("0.1.1rc1", "patch", "rc", False, False) == "0.1.1rc2"
    assert bump("0.1.1rc2", "final", None, False, False) == "0.1.1"
    assert bump("0.1.1", "patch", None, True, False) == "0.1.1.post1"


def test_inspect_tool_distribution_flags_blockers(tmp_path):
    mod = _load_script("inspect_tool_distribution.py")
    wheel = tmp_path / "agentic_automl-9.9.9.dev1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for m in mod.REQUIRED_WHEEL_MEMBERS[:-1]:
            zf.writestr(m, "")
        zf.writestr("agentic_automl/_data/steps/generate_synthetic_fraud.py", "")
        zf.writestr("agentic_automl-9.9.9.dev1.dist-info/METADATA", "Version: 9.9.9.dev1\n")
        zf.writestr("agentic_automl-9.9.9.dev1.dist-info/entry_points.txt",
                    "[console_scripts]\nagentic-automl = agentic_automl.cli:main\n")
    r = mod.inspect(wheel, release_ready=True)
    text = " ".join(r["problems"])
    assert not r["ok"]
    assert "missing required member" in text and "generator" in text and "development version" in text
