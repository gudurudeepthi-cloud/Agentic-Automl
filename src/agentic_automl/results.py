"""Durable results ledger: `agentic-automl results [sync|show|path]`.

One JSON record per trial in <results_dir>/trials/<trial_id>.json, plus
leaderboard.csv and leaderboard.md across all of them. Records outlive their
trial directories, so deleting a trial keeps its result. The launcher refreshes
the ledger after every session; `results sync` rebuilds it from all trials.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agentic_automl import artifacts
from agentic_automl import state as st
from agentic_automl.contracts import ARTIFACT_REGISTRY, results_dir

LEADERBOARD_COLUMNS = [
    "trial_id", "created_at", "status", "dataset_path", "target_column", "primary_metric",
    "primary_metric_value", "cv_score", "selected_model", "preprocessing_engine",
    "trust_checks_failed", "forked_from",
]


def _pick(payload: dict[str, Any] | None, *fields: str) -> dict[str, Any]:
    return {f: payload.get(f) for f in fields} if payload else {}


def build_record(trial: Path) -> dict[str, Any]:
    state = artifacts.load_state(trial)
    load = lambda key: artifacts.load_artifact(trial, key)  # noqa: E731
    present = artifacts.recorded_artifacts(trial, state)
    contract, plan, training, evaluation = (load(k) for k in
                                            ("prediction_contract", "workflow_plan", "training_summary",
                                             "evaluation_summary"))
    step = artifacts.next_step(trial)
    selected = None
    if training:
        selected = next((m for m in training["models"] if m["name"] == training["selected_model"]), None)
    checks = (evaluation or {}).get("trust_checks") or []
    return {
        "trial_id": state["trial_id"],
        "trial_dir": str(trial),
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "goal": state.get("goal"),
        "dataset_path": (contract or {}).get("dataset_path") or state.get("dataset_path"),
        "forked_from": state.get("forked_from"),
        "previous_trial_ids": [Path(p).name for p in state.get("previous_trial_dirs", [])],
        "status": "complete" if step["canonical_workflow_complete"] else f"next: {step['next_skill']}",
        "artifacts": {e["key"]: e.get("version", 1) for e in state["artifacts"] if e["key"] in present},
        "contract": _pick(contract, "task_type", "target_column", "target_kind", "split_strategy",
                          "primary_metric", "report_metrics", "model_preference", "deployment_intent",
                          "ml_domain"),
        "plan": {
            "preprocessing_engine": ((plan or {}).get("preprocessing_plan") or {}).get("engine",
                                                                                      "parity-ml" if plan else None),
            "candidate_models": [m["name"] for m in ((plan or {}).get("modeling_plan") or {}).get("candidate_models", [])],
        } if plan else {},
        "training": {
            "selected_model": training["selected_model"],
            "cv_strategy": training["cv_strategy"],
            "cv_score": (selected or {}).get("cv_score"),
            "cv_score_std": (selected or {}).get("cv_score_std"),
        } if training else {},
        "evaluation": {
            **_pick(evaluation, "primary_metric", "primary_metric_value", "metrics", "threshold", "narrative"),
            "trust_checks": checks,
            "trust_checks_failed": [c["name"] for c in checks if not c["passed"]],
        } if evaluation else {},
        "deliverables": {k: ARTIFACT_REGISTRY[k]["path"] for k in sorted(present)},
    }


def record_path(trial_id: str) -> Path:
    return results_dir() / "trials" / f"{trial_id}.json"


def record_trial(trial: Path) -> dict[str, Any]:
    record = build_record(trial)
    path = record_path(record["trial_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for old in record["previous_trial_ids"]:  # renamed by a slug: drop the stale record
        stale = record_path(old)
        if stale != path and stale.exists():
            stale.unlink()
    write_leaderboard()
    return record


def load_records() -> list[dict[str, Any]]:
    folder = results_dir() / "trials"
    records = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda r: r.get("created_at") or "", reverse=True)


def leaderboard_row(record: dict[str, Any]) -> dict[str, Any]:
    ev, tr, ct = record.get("evaluation") or {}, record.get("training") or {}, record.get("contract") or {}
    return {
        "trial_id": record["trial_id"],
        "created_at": record.get("created_at"),
        "status": record.get("status"),
        "dataset_path": record.get("dataset_path"),
        "target_column": ct.get("target_column"),
        "primary_metric": ev.get("primary_metric") or ct.get("primary_metric"),
        "primary_metric_value": ev.get("primary_metric_value"),
        "cv_score": tr.get("cv_score"),
        "selected_model": tr.get("selected_model"),
        "preprocessing_engine": (record.get("plan") or {}).get("preprocessing_engine"),
        "trust_checks_failed": ";".join(ev.get("trust_checks_failed") or []),
        "forked_from": record.get("forked_from"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value).replace("|", "/")


def write_leaderboard() -> Path:
    rows = [leaderboard_row(r) for r in load_records()]
    root = results_dir()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "leaderboard.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Trial leaderboard",
        "",
        f"Generated {artifacts.now_iso()} from `{root / 'trials'}`. Newest first.",
        "",
        "| " + " | ".join(LEADERBOARD_COLUMNS) + " |",
        "|" + "---|" * len(LEADERBOARD_COLUMNS),
    ]
    lines += ["| " + " | ".join(_fmt(row[c]) for c in LEADERBOARD_COLUMNS) + " |" for row in rows]
    md = root / "leaderboard.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def sync_all() -> dict[str, Any]:
    recorded, errors = [], []
    for trial in st.list_trials():
        try:
            recorded.append(record_trial(trial)["trial_id"])
        except (artifacts.ArtifactError, KeyError, OSError) as exc:
            errors.append({"trial": trial.name, "error": str(exc)})
    write_leaderboard()
    return {"ok": not errors, "recorded": recorded, "errors": errors, "results_dir": str(results_dir())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-automl results", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", choices=["sync", "show", "path"], default="show")
    parser.add_argument("--json", action="store_true", help="show: print records as JSON")
    args = parser.parse_args(argv)
    if args.action == "sync":
        print(json.dumps(sync_all(), indent=2))
    elif args.action == "path":
        print(results_dir())
    elif args.json:
        print(json.dumps(load_records(), indent=2))
    else:
        md = results_dir() / "leaderboard.md"
        print(md.read_text(encoding="utf-8") if md.is_file() else "No results yet. Run: agentic-automl results sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
