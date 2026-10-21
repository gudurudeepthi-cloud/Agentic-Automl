"""Deterministic scorers: check a scenario's invariants against a trial directory (no Claude spend)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_automl.contracts import ARTIFACT_REGISTRY, RUN_STATE_PATH

CREDIT_MARKER = "aaml-feedback-credit:v1"
_CREDIT_RE = re.compile(rf"<!--\s*{re.escape(CREDIT_MARKER)}\s+(\{{.*?\}})\s*-->", re.S)


@dataclass
class EvalScore:
    name: str
    passed: bool
    detail: str


@dataclass
class TrialScoreResult:
    scores: list[EvalScore] = field(default_factory=list)
    artifact_snapshot: dict[str, bool] = field(default_factory=dict)
    process_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.scores) and all(s.passed for s in self.scores)

    def score_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.scores]


def artifact_snapshot(trial: Path) -> dict[str, bool]:
    return {key: (trial / spec["path"]).is_file() for key, spec in ARTIFACT_REGISTRY.items()}


def _run_state(trial: Path) -> dict:
    p = trial / RUN_STATE_PATH
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def count_validation_failures(trial: Path) -> int:
    return sum(1 for e in _run_state(trial).get("validation_log", []) if not e.get("ok", True))


def count_retries(trial: Path) -> int:
    """Saves of an artifact after a failed attempt at the same artifact."""
    failed, retries = set(), 0
    for e in _run_state(trial).get("validation_log", []):
        key = e.get("artifact") or e.get("key")
        if not e.get("ok", True):
            failed.add(key)
        elif key in failed:
            retries += 1
            failed.discard(key)
    return retries


def _lookup_metric_value(payload: Any, dotted: str) -> Any:
    cur = payload
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)] if part.isdigit() else next((x for x in cur if x.get("name") == part), None)
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def validate_metric_band_bounds(inv: dict) -> str | None:
    if "min" not in inv and "max" not in inv:
        return "metric_band needs min and/or max"
    if "min" in inv and "max" in inv and inv["min"] > inv["max"]:
        return f"metric_band min {inv['min']} > max {inv['max']}"
    return None


def validate_smoke_metric_policy(scenario: dict) -> list[str]:
    """Smoke-sized data can't support a tight band: require one at least 0.2 wide (or one-sided)."""
    errs = []
    if scenario.get("data_size") == "smoke":
        for inv in scenario["invariants"]:
            if inv["type"] == "metric_band" and "min" in inv and "max" in inv and inv["max"] - inv["min"] < 0.2:
                errs.append(f"{inv.get('name') or inv['metric']}: smoke band narrower than 0.2")
    return errs


def parse_credit_marker(text: str) -> dict | None:
    m = _CREDIT_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("layer") and data.get("summary") else None


def _score_invariant(inv: dict, trial: Path) -> EvalScore:
    kind = inv["type"]
    name = inv.get("name") or f"{kind}:{inv.get('path') or inv.get('artifact') or inv.get('metric') or ''}"
    path = trial / inv["path"] if inv.get("path") else None
    if kind == "file_present":
        return EvalScore(name, path.is_file(), f"{inv['path']} {'exists' if path.is_file() else 'missing'}")
    if kind == "file_absent":
        return EvalScore(name, not path.exists(), f"{inv['path']} {'present' if path.exists() else 'absent'}")
    if kind == "file_contains":
        if not path.is_file():
            return EvalScore(name, False, f"{inv['path']} missing")
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [p for p in inv.get("patterns", []) if not re.search(p, text, re.M)]
        present = [p for p in inv.get("absent_patterns", []) if re.search(p, text, re.M)]
        return EvalScore(name, not missing and not present,
                         f"missing patterns {missing}; forbidden present {present}" if missing or present else "ok")
    if kind == "artifact_present":
        rel = ARTIFACT_REGISTRY[inv["artifact"]]["path"]
        ok = (trial / rel).is_file()
        return EvalScore(name, ok, f"{rel} {'exists' if ok else 'missing'}")
    if kind == "metric_band":
        problem = validate_metric_band_bounds(inv)
        if problem:
            return EvalScore(name, False, problem)
        rel = ARTIFACT_REGISTRY[inv["artifact"]]["path"]
        if not (trial / rel).is_file():
            return EvalScore(name, False, f"{rel} missing")
        value = _lookup_metric_value(json.loads((trial / rel).read_text(encoding="utf-8")), inv["metric"])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return EvalScore(name, False, f"{inv['metric']} not numeric in {rel}: {value!r}")
        ok = inv.get("min", float("-inf")) <= value <= inv.get("max", float("inf"))
        return EvalScore(name, ok, f"{inv['metric']} = {value:.4f}, band [{inv.get('min')}, {inv.get('max')}]")
    if kind == "max_validation_failures":
        n = count_validation_failures(trial)
        return EvalScore(name, n <= inv.get("max_count", 0), f"{n} validation failures (max {inv.get('max_count', 0)})")
    if kind == "feedback_credit_marker_parseable":
        if not path or not path.is_file():
            return EvalScore(name, False, f"{inv.get('path')} missing")
        data = parse_credit_marker(path.read_text(encoding="utf-8", errors="replace"))
        return EvalScore(name, data is not None, f"marker {'parsed: ' + str(data) if data else 'missing or malformed'}")
    return EvalScore(name, False, f"unknown invariant type {kind!r}")


def score_trial(scenario: dict, trial_dir: str | Path) -> TrialScoreResult:
    trial = Path(trial_dir)
    snap = artifact_snapshot(trial)
    scores = [EvalScore(f"expected_artifact:{a}", snap.get(a, False),
                        f"{ARTIFACT_REGISTRY[a]['path']} {'exists' if snap.get(a) else 'missing'}")
              for a in scenario.get("expected_artifacts", [])]
    scores += [EvalScore(f"policy:{e}", False, e) for e in validate_smoke_metric_policy(scenario)]
    scores += [_score_invariant(inv, trial) for inv in scenario["invariants"]]
    return TrialScoreResult(scores, snap, {"validation_failures": count_validation_failures(trial),
                                           "retries": count_retries(trial)})
