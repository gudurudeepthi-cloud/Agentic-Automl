"""Print one benchmark dataset's readiness entry: `python scripts/inspect_benchmark_dataset.py --id telco_churn [--json]`.

Offline (no GCS): registry metadata, readiness status, issues, smoke fixture, scenarios and seeds.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_automl.benchmark_readiness import audit_benchmark_datasets, load_dataset_registry  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    entries = {e["id"]: e for e in load_dataset_registry()}
    if args.id not in entries:
        print(f"unknown dataset {args.id!r}; known: {', '.join(sorted(entries))}", file=sys.stderr)
        return 1
    audit = next(r for r in audit_benchmark_datasets() if r.id == args.id)
    seeds = sorted(p.name for p in (ROOT / "benchmarks/evals/fixtures/seeds").glob(f"{args.id}_*") if p.is_dir())
    out = {"entry": entries[args.id], "readiness": audit.to_dict(), "seeds": seeds}
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    e = entries[args.id]
    print(f"{args.id}: {audit.status.value}")
    print(f"  task {e.get('task_type')} target {e.get('target_variable')!r} metric {e.get('primary_metric')} "
          f"access {e['access']} status {e['status']}")
    print(f"  smoke fixture: {audit.smoke_csv or 'none'}")
    print(f"  scenarios: {', '.join(audit.scenarios) or 'none'}")
    print(f"  seeds: {', '.join(seeds) or 'none'}")
    for i in audit.issues:
        print(f"  {'BLOCKING' if i.blocking else 'note'}: {i.field}: {i.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
