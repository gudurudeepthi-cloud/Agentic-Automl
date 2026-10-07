"""One row per trial: `agentic-automl summarize-trials [--json]`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_automl import artifacts
from agentic_automl import state as st


def collect_trial(trial_dir: Path) -> dict[str, Any]:
    state = st.read_state(trial_dir) or {}
    row: dict[str, Any] = {
        "trial_id": trial_dir.name,
        "created_at": state.get("created_at"),
        "goal": state.get("goal"),
        "dataset_path": state.get("dataset_path"),
        "n_artifacts": 0,
        "next_skill": None,
        "primary_metric": None,
        "primary_metric_value": None,
        "forked_from": state.get("forked_from"),
    }
    if not state:
        row["error"] = "unreadable run_state.json"
        return row
    try:
        row["n_artifacts"] = len(artifacts.recorded_artifacts(trial_dir, state))
        step = artifacts.next_step(trial_dir)
        row["next_skill"] = "complete" if step["canonical_workflow_complete"] else step["next_skill"]
    except (artifacts.ArtifactError, KeyError) as exc:
        row["error"] = str(exc)
    evaluation = artifacts.load_artifact(trial_dir, "evaluation_summary")
    if evaluation:
        row["primary_metric"] = evaluation.get("primary_metric")
        row["primary_metric_value"] = evaluation.get("primary_metric_value")
    return row


def _cell(value: Any, width: int) -> str:
    text = "" if value is None else (f"{value:.4g}" if isinstance(value, float) else str(value))
    return (text[: width - 1] + "~") if len(text) > width else text.ljust(width)


COLUMNS = [("trial_id", 34), ("goal", 30), ("n_artifacts", 4), ("next_skill", 24),
           ("primary_metric", 12), ("primary_metric_value", 8)]


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = {"n_artifacts": "art", "primary_metric": "metric", "primary_metric_value": "value"}
    print("  ".join(_cell(headers.get(c, c), w) for c, w in COLUMNS).rstrip())
    for row in rows:
        print("  ".join(_cell(row.get(c), w) for c, w in COLUMNS).rstrip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-automl summarize-trials", description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON rows instead of a table")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    rows = [collect_trial(t) for t in st.recent_trials(args.limit)]
    if args.json:
        print(json.dumps(rows, indent=2))
    elif rows:
        print_table(rows)
    else:
        print("No trials yet. Start one with: agentic-automl new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
