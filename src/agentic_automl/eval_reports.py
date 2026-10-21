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
