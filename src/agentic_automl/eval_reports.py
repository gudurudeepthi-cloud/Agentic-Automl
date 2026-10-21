"""Eval result buckets, markdown report and the gate verdict. An all-skipped suite is NOT a pass."""
from __future__ import annotations

from collections import Counter

PASSING_STATUSES = {"pass"}
FAILING_STATUSES = {"fail", "harness-error", "harness-unavailable", "incomplete"}
SKIPPED_STATUSES = {"environment-unavailable"}


def summarize_results(results: list[dict]) -> dict:
    counts = Counter(r["status"] for r in results)
    return {
        "total": len(results),
        "passed": sum(counts[s] for s in PASSING_STATUSES),
        "failed": sum(counts[s] for s in FAILING_STATUSES),
        "skipped": sum(counts[s] for s in SKIPPED_STATUSES),
        "by_status": dict(counts),
    }


def verdict(results: list[dict]) -> tuple[bool, str]:
    s = summarize_results(results)
    if s["total"] == 0:
        return False, "no scenarios ran"
    if s["failed"]:
        return False, f"{s['failed']} of {s['total']} scenarios failed"
    if s["passed"] == 0:
        return False, f"all {s['total']} scenarios were skipped (environment unavailable): not a pass"
    note = f"; {s['skipped']} skipped" if s["skipped"] else ""
    return True, f"{s['passed']} of {s['total']} scenarios passed{note}"


def render_results_table(results: list[dict]) -> str:
    lines = ["| scenario | tier | status | failed checks | seconds |", "|---|---|---|---|---|"]
    for r in results:
        failed = [x["name"] for x in r["scores"] if not x["passed"]]
        lines.append(f"| {r['scenario_id']} | {r.get('tier', '')} | {r['status']} | "
                     f"{', '.join(failed) or '-'} | {r['elapsed_seconds']:.1f} |")
    return "\n".join(lines)


def render_summary(results: list[dict]) -> str:
    ok, why = verdict(results)
    s = summarize_results(results)
    return (f"**Verdict: {'PASS' if ok else 'FAIL'}** ({why}). "
            f"passed {s['passed']}, failed {s['failed']}, skipped {s['skipped']}.")


TSV_COLUMNS = ["run_id", "scenario_id", "dataset_id", "tier", "status", "primary_metric", "test_value",
               "cv_score", "cv_score_std", "selected_model", "trust_checks_failed", "failed_scores", "cost_usd",
               "elapsed_seconds", "trial_dir"]


def _model_metrics(trial_dir: str | None) -> dict:
    """Headline numbers from the scored trial's artifacts (blank when a stage didn't run)."""
    import json
    from pathlib import Path

    out = {}
    if not trial_dir:
        return out
    t = Path(trial_dir)
    ev, tr = t / "outputs/report/evaluation_summary.json", t / "outputs/model/training_summary.json"
    if ev.is_file():
        e = json.loads(ev.read_text(encoding="utf-8"))
        out.update(primary_metric=e.get("primary_metric"), test_value=e.get("primary_metric_value"),
                   trust_checks_failed=",".join(c["name"] for c in e.get("trust_checks", []) if not c["passed"]))
    if tr.is_file():
        s = json.loads(tr.read_text(encoding="utf-8"))
        m = next((m for m in s.get("models", []) if m["name"] == s.get("selected_model")), {})
        out.update(selected_model=s.get("selected_model"), cv_score=m.get("cv_score"), cv_score_std=m.get("cv_score_std"))
        out.setdefault("primary_metric", s.get("primary_metric"))
    return out


def render_results_tsv(results: list[dict]) -> str:
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("\t", " ").replace("\n", " ")

    rows = ["\t".join(TSV_COLUMNS)]
    for r in results:
        row = {**r, **_model_metrics(r.get("trial_dir")),
               "failed_scores": ",".join(x["name"] for x in r["scores"] if not x["passed"]),
               "cost_usd": (r.get("cost") or {}).get("usd")}
        rows.append("\t".join(cell(row.get(c)) for c in TSV_COLUMNS))
    return "\n".join(rows) + "\n"


def render_results_markdown(results: list[dict], title: str = "Agent eval report") -> str:
    parts = [f"# {title}", "", render_summary(results), "", render_results_table(results), ""]
    for r in results:
        bad = [x for x in r["scores"] if not x["passed"]]
        if bad or r["status"] not in PASSING_STATUSES:
            parts += [f"## {r['scenario_id']} ({r['status']})", ""]
            if r.get("stop_reason"):
                parts.append(f"Stop reason: {r['stop_reason']}")
            parts += [f"- **{x['name']}**: {x['detail']}" for x in bad] + [""]
    return "\n".join(parts)
