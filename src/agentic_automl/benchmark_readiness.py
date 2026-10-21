"""Audit benchmarks/datasets.json for eval-readiness (offline: no downloads, no GCS calls).

Each registry entry gets one ReadinessStatus, worst blocker first:
  blocked_metadata            schema-invalid, or a CRITICAL_FIELD is missing
  blocked_owner_confirmation  an owner is named but has not confirmed (owner_confirmed false)
  blocked_gcs                 access=gcs without gcs_paths.staging
  metadata_draft              status=draft
  ready_for_smoke             a committed smoke CSV exists (benchmarks/evals/fixtures/datasets/<id>.csv)
  ready_for_scored_eval       smoke CSV + a metric band (performance.expected_band or automl_best)
                              + at least one scenario under benchmarks/evals/scenarios/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from jsonschema import Draft7Validator

from agentic_automl.contracts import schemas_dir, source_root

CRITICAL_FIELDS = ("task_type", "target_variable", "primary_metric", "dataset_format")


class ReadinessStatus(str, Enum):
    METADATA_DRAFT = "metadata_draft"
    READY_FOR_SMOKE = "ready_for_smoke"
    READY_FOR_SCORED_EVAL = "ready_for_scored_eval"
    BLOCKED_METADATA = "blocked_metadata"
    BLOCKED_GCS = "blocked_gcs"
    BLOCKED_OWNER_CONFIRMATION = "blocked_owner_confirmation"


@dataclass
class ReadinessIssue:
    field: str
    message: str
    blocking: bool = True


@dataclass
class DatasetReadiness:
    id: str
    status: ReadinessStatus
    issues: list[ReadinessIssue] = field(default_factory=list)
    smoke_csv: str | None = None
    scenarios: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def repo_root() -> Path:
    return source_root() or Path.cwd()


def benchmarks_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "benchmarks"


def load_dataset_registry(path: Path | None = None) -> list[dict]:
    path = path or benchmarks_dir() / "datasets.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sample_manifests(root: Path | None = None) -> dict[str, dict]:
    """{dataset_id: manifest} from benchmarks/datasets/<id>/samples.json."""
    out = {}
    for p in sorted((benchmarks_dir(root) / "datasets").glob("*/samples.json")):
        manifest = json.loads(p.read_text(encoding="utf-8"))
        out[manifest.get("dataset_id", p.parent.name)] = manifest
    return out


def _scenarios_for(dataset_id: str, root: Path) -> list[str]:
    out = []
    for p in sorted((benchmarks_dir(root) / "evals" / "scenarios").glob("*.json")):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("dataset_id") == dataset_id:
                out.append(p.name)
        except json.JSONDecodeError:
            continue
    return out


def audit_entry(entry: dict, root: Path | None = None) -> DatasetReadiness:
    root = root or repo_root()
    validator = Draft7Validator(json.loads((schemas_dir() / "benchmark_dataset.schema.json").read_text("utf-8")))
    ds_id = str(entry.get("id", "<missing id>"))
    issues = [ReadinessIssue("/".join(map(str, e.path)) or "<root>", e.message)
              for e in validator.iter_errors(entry)]
    issues += [ReadinessIssue(f, "missing critical field") for f in CRITICAL_FIELDS if not entry.get(f)]
    smoke = benchmarks_dir(root) / "evals" / "fixtures" / "datasets" / f"{ds_id}.csv"
    scenarios = _scenarios_for(ds_id, root)
    perf = entry.get("performance") or {}
    has_band = bool(perf.get("expected_band") or perf.get("automl_best"))

    if issues:
        status = ReadinessStatus.BLOCKED_METADATA
    elif entry.get("owner") and not entry.get("owner_confirmed"):
        status = ReadinessStatus.BLOCKED_OWNER_CONFIRMATION
        issues.append(ReadinessIssue("owner_confirmed", f"owner {entry['owner']!r} has not confirmed the entry"))
    elif entry.get("access") == "gcs" and not (entry.get("gcs_paths") or {}).get("staging"):
        status = ReadinessStatus.BLOCKED_GCS
        issues.append(ReadinessIssue("gcs_paths.staging", "access=gcs needs a staging path"))
    elif entry.get("status") == "draft":
        status = ReadinessStatus.METADATA_DRAFT
    elif smoke.is_file() and has_band and scenarios:
        status = ReadinessStatus.READY_FOR_SCORED_EVAL
    elif smoke.is_file():
        status = ReadinessStatus.READY_FOR_SMOKE
        if not has_band:
            issues.append(ReadinessIssue("performance", "no expected_band/automl_best: cannot score", False))
        if not scenarios:
            issues.append(ReadinessIssue("scenarios", "no scenario references this dataset", False))
    else:
        status = ReadinessStatus.METADATA_DRAFT
        issues.append(ReadinessIssue("smoke_csv", f"no smoke fixture at {smoke.relative_to(root).as_posix()} "
                                                  "(run scripts/generate_dataset_seeds.py)", False))
    return DatasetReadiness(ds_id, status, issues, smoke.relative_to(root).as_posix() if smoke.is_file() else None,
                            scenarios)


def audit_benchmark_datasets(registry: list[dict] | None = None, root: Path | None = None) -> list[DatasetReadiness]:
    root = root or repo_root()
    registry = registry if registry is not None else load_dataset_registry(benchmarks_dir(root) / "datasets.json")
    seen, out = set(), []
    for entry in registry:
        r = audit_entry(entry, root)
        if r.id in seen:
            r.status = ReadinessStatus.BLOCKED_METADATA
            r.issues.append(ReadinessIssue("id", "duplicate id in registry"))
        seen.add(r.id)
        out.append(r)
    return out


def render_markdown(results: list[DatasetReadiness]) -> str:
    lines = ["# Benchmark dataset readiness", "", "| dataset | status | scenarios | issues |", "|---|---|---|---|"]
    for r in results:
        issues = "; ".join(f"{'**' if i.blocking else ''}{i.field}{'**' if i.blocking else ''}: {i.message}"
                           for i in r.issues) or "none"
        lines.append(f"| {r.id} | {r.status.value} | {len(r.scenarios)} | {issues} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audit_benchmark_datasets", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", type=Path, default=None, help="also write the markdown report here")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any dataset is blocked")
    args = ap.parse_args(argv)
    results = audit_benchmark_datasets(load_dataset_registry(args.registry) if args.registry else None)
    md = render_markdown(results)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
    print(json.dumps([r.to_dict() for r in results], indent=2) if args.json else md)
    blocked = [r for r in results if r.status.value.startswith("blocked")]
    return 1 if args.strict and blocked else 0
