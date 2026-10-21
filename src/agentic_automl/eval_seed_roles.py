"""Keep generated seed column roles honest.

The registry entry (benchmarks/datasets.json) is the golden source for which columns are
the target, ids and time. A seed's column_profile was produced by running the profiling
template; if it disagrees on those roles, the seed is wrong (or the registry is), and a
scenario scored against it would test the wrong thing.
"""
from __future__ import annotations

from pathlib import Path

from agentic_automl.benchmark_readiness import load_dataset_registry, repo_root


def registry_entry(dataset_id: str, root: Path | None = None) -> dict:
    root = root or repo_root()
    for entry in load_dataset_registry(root / "benchmarks" / "datasets.json"):
        if entry["id"] == dataset_id:
            return entry
    raise KeyError(f"dataset {dataset_id!r} not in benchmarks/datasets.json")


def derive_roles(smoke_config: dict) -> dict[str, str]:
    """Golden {column: role} for the columns whose role is fixed by the registry."""
    roles = {smoke_config["target_variable"]: "target"}
    for col in smoke_config.get("id_columns", []):
        roles[col] = "id"
    if smoke_config.get("entity_id_column"):
        roles[smoke_config["entity_id_column"]] = "id"
    time_col = (smoke_config.get("split") or {}).get("time_column")
    if time_col:
        roles[time_col] = "time"
    return roles


def golden_id_columns(dataset_id: str, root: Path | None = None) -> list[str]:
    entry = registry_entry(dataset_id, root)
    return sorted({*entry.get("id_columns", []), *([entry["entity_id_column"]] if entry.get("entity_id_column") else [])})


def assert_roles_match_golden(column_profile: dict, smoke_config: dict) -> None:
    actual = {c["name"]: c["role"] for c in column_profile["columns"]}
    golden = derive_roles(smoke_config)
    wrong = {col: {"golden": role, "seed": actual.get(col, "<missing>")}
             for col, role in golden.items() if actual.get(col) != role}
    if wrong:
        raise AssertionError(f"seed column roles disagree with the registry: {wrong}")
