"""Validate (or --write) benchmark sample manifests: benchmarks/datasets/<id>/samples.json.

Default: load every manifest, validate it against schemas/benchmark_samples.schema.json, and
check each sample file's row count and sha256. --write rebuilds the manifest for --id (or all
datasets) from the committed smoke fixture benchmarks/evals/fixtures/datasets/<id>.csv.

    python scripts/prepare_benchmark_samples.py
    python scripts/prepare_benchmark_samples.py --write --id telco_churn
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsonschema import Draft7Validator  # noqa: E402

from agentic_automl.benchmark_readiness import load_dataset_registry, load_sample_manifests  # noqa: E402


def rows_and_hash(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return max(0, data.count(b"\n") - 1), hashlib.sha256(data).hexdigest()


def write_manifest(ds_id: str) -> Path | None:
    csv_path = ROOT / "benchmarks/evals/fixtures/datasets" / f"{ds_id}.csv"
    if not csv_path.is_file():
        print(f"{ds_id}: no smoke fixture (run scripts/generate_dataset_seeds.py --dataset {ds_id})")
        return None
    rows, digest = rows_and_hash(csv_path)
    manifest = {"dataset_id": ds_id, "samples": [{
        "name": "smoke", "path": csv_path.relative_to(ROOT).as_posix(), "rows": rows, "sha256": digest,
        "seed": 0, "method": "seeded row sample, original order kept (agentic_automl.eval_seeds.make_smoke_csv)"}]}
    out = ROOT / "benchmarks/datasets" / ds_id / "samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{ds_id}: wrote {out.relative_to(ROOT).as_posix()} ({rows} rows)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    if args.write:
        ids = [args.id] if args.id else [e["id"] for e in load_dataset_registry()]
        for ds_id in ids:
            write_manifest(ds_id)
    validator = Draft7Validator(json.loads((ROOT / "schemas/benchmark_samples.schema.json").read_text("utf-8")))
    bad = 0
    for ds_id, manifest in load_sample_manifests().items():
        if args.id and ds_id != args.id:
            continue
        problems = [e.message for e in validator.iter_errors(manifest)]
        for s in manifest.get("samples", []):
            p = ROOT / s["path"]
            if not p.is_file():
                problems.append(f"{s['path']} missing")
                continue
            rows, digest = rows_and_hash(p)
            if rows != s["rows"]:
                problems.append(f"{s['path']}: {rows} rows, manifest says {s['rows']}")
            if digest != s["sha256"]:
                problems.append(f"{s['path']}: sha256 changed (re-run with --write if intended)")
        bad += bool(problems)
        print(f"{ds_id}: {'OK' if not problems else 'INVALID: ' + '; '.join(problems)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
