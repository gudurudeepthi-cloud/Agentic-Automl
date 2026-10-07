import csv
import json
from pathlib import Path

from agentic_automl import artifacts, cli, results
from agentic_automl import state as st
from agentic_automl.contracts import results_dir, storage_dir, trials_dir
from conftest import save, save_through, valid


def test_storage_defaults_outside_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTIC_AUTOML_TRIALS_DIR")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    assert storage_dir() == tmp_path / "userhome" / "agentic-automl"
    assert trials_dir() == storage_dir() / "trials"
    assert results_dir() == storage_dir() / "results"


def test_storage_dir_from_config(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("AGENTIC_AUTOML_TRIALS_DIR")
    assert cli.main(["config", "set-storage-dir", str(tmp_path / "shared")]) == 0
    capsys.readouterr()
    assert trials_dir() == (tmp_path / "shared" / "trials").resolve()
    assert cli.main(["config", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["results_dir"] == str((tmp_path / "shared" / "results").resolve())


def test_record_partial_trial(trial):
    save(trial, "prediction_contract")
    record = results.record_trial(trial)
    assert record["status"] == "next: data-profiling"
    assert record["contract"]["target_column"] == "churned"
    assert record["evaluation"] == {}
    assert results.record_path("t001").is_file()


def test_record_complete_trial_and_leaderboard(trial):
    save_through(trial, "training_summary")
    save(trial, "evaluation_summary", valid("evaluation_summary", trust_checks=[
        {"name": "leakage", "passed": True, "detail": "ok"},
        {"name": "cv_vs_test_gap", "passed": False, "detail": "gap 0.08"},
    ]))
    record = results.record_trial(trial)
    assert record["status"] == "complete"
    assert record["evaluation"]["primary_metric_value"] == 0.8
    assert record["evaluation"]["trust_checks_failed"] == ["cv_vs_test_gap"]
    assert record["training"]["cv_score"] == 0.81
    assert record["plan"]["preprocessing_engine"] == "open_source"
    rows = list(csv.DictReader(open(results_dir() / "leaderboard.csv", encoding="utf-8")))
    assert rows[0]["trial_id"] == "t001" and rows[0]["trust_checks_failed"] == "cv_vs_test_gap"
    assert "| t001 |" in (results_dir() / "leaderboard.md").read_text(encoding="utf-8")


def test_result_survives_trial_deletion(trial):
    import shutil

    results.record_trial(trial)
    shutil.rmtree(trial)
    results.write_leaderboard()
    assert [r["trial_id"] for r in results.load_records()] == ["t001"]


def test_slug_rename_replaces_record(trial):
    results.record_trial(trial)
    artifacts.set_trial_slug(trial, "churn")
    new_dir = artifacts.apply_pending_slug(trial)
    results.record_trial(new_dir)
    assert [r["trial_id"] for r in results.load_records()] == ["t001--churn"]


def test_new_and_launch_update_ledger(monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: 0)
    cli.main(["new", "--goal", "g"])
    [record] = results.load_records()
    assert record["goal"] == "g"


def test_results_cli_sync_and_show(trial, capsys):
    assert cli.main(["results", "sync"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["recorded"] == ["t001"] and out["ok"]
    assert cli.main(["results"]) == 0
    assert "t001" in capsys.readouterr().out


def test_ledger_failure_does_not_break_launcher(monkeypatch, trial, capsys):
    monkeypatch.setattr(results, "record_trial", lambda t: (_ for _ in ()).throw(OSError("disk full")))
    cli.update_results(trial)
    assert "results ledger not updated" in capsys.readouterr().err
