"""Tiny deterministic datasets for credential-free smoke tests (stdlib only).

A config names columns of a few kinds plus a target whose log-odds (or mean) is a
weighted sum of the features, so a model has real signal to find. The same config and
seed always produce byte-identical CSV.

    {"name": "toy_churn", "rows": 2000, "seed": 7,
     "target": {"name": "churned", "kind": "binary", "base_rate": 0.2, "noise": 1.0},
     "columns": [
       {"name": "customer_id", "kind": "integer", "role": "id"},
       {"name": "tenure", "kind": "numeric", "mean": 30, "std": 15, "min": 0, "weight": -0.8},
       {"name": "plan", "kind": "categorical", "levels": ["basic", "plus", "pro"], "level_weights": [0.4, -0.2, -0.5]},
       {"name": "signup", "kind": "date", "start": "2024-01-01", "days": 365},
       {"name": "autopay", "kind": "boolean", "p": 0.6, "weight": -0.5, "missing": 0.05}]}
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

ColumnKind = Literal["numeric", "integer", "categorical", "date", "boolean"]
KINDS = ("numeric", "integer", "categorical", "date", "boolean")


@dataclass
class SyntheticColumn:
    name: str
    kind: ColumnKind
    role: str = "feature"            # feature | id | time (golden roles, see eval_seed_roles)
    mean: float = 0.0
    std: float = 1.0
    min: float | None = None
    max: float | None = None
    levels: list[str] = field(default_factory=list)
    level_weights: list[float] = field(default_factory=list)
    p: float = 0.5
    start: str = "2024-01-01"
    days: int = 365
    weight: float = 0.0              # contribution of the standardised value to the target
    missing: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"column {self.name}: kind {self.kind!r} not in {KINDS}")
        if self.kind == "categorical" and not self.levels:
            raise ValueError(f"column {self.name}: categorical needs levels")
        if self.level_weights and len(self.level_weights) != len(self.levels):
            raise ValueError(f"column {self.name}: level_weights must match levels")


@dataclass
class SyntheticDatasetConfig:
    name: str
    rows: int
    columns: list[SyntheticColumn]
    target: dict[str, Any]
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticDatasetConfig":
        return cls(name=d["name"], rows=int(d["rows"]), seed=int(d.get("seed", 0)), target=d["target"],
                   columns=[SyntheticColumn(**c) for c in d["columns"]])

    @classmethod
    def from_json(cls, path: str | Path) -> "SyntheticDatasetConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _value(col: SyntheticColumn, i: int, rng: random.Random) -> tuple[Any, float]:
    """(cell value, standardised contribution to the target signal)."""
    if col.role == "id":
        return i + 1, 0.0
    if col.kind in ("numeric", "integer"):
        x = rng.gauss(col.mean, col.std)
        if col.min is not None:
            x = max(col.min, x)
        if col.max is not None:
            x = min(col.max, x)
        z = (x - col.mean) / col.std if col.std else 0.0
        return (int(round(x)) if col.kind == "integer" else round(x, 4)), col.weight * z
    if col.kind == "categorical":
        k = rng.randrange(len(col.levels))
        return col.levels[k], (col.level_weights[k] if col.level_weights else 0.0)
    if col.kind == "boolean":
        b = rng.random() < col.p
        return int(b), col.weight * (1.0 if b else -1.0)
    d = date.fromisoformat(col.start) + timedelta(days=rng.randrange(col.days))
    return d.isoformat(), 0.0


def rows_for_config(cfg: SyntheticDatasetConfig) -> list[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    t = cfg.target
    kind, noise = t.get("kind", "binary"), float(t.get("noise", 1.0))
    base = float(t.get("base_rate", 0.2))
    intercept = math.log(base / (1 - base)) if kind == "binary" else float(t.get("mean", 0.0))
    out = []
    for i in range(cfg.rows):
        row, signal = {}, intercept
        for col in cfg.columns:
            v, s = _value(col, i, rng)
            signal += s
            row[col.name] = "" if col.role != "id" and col.missing and rng.random() < col.missing else v
        signal += rng.gauss(0, noise)
        if kind == "binary":
            row[t["name"]] = int(rng.random() < 1 / (1 + math.exp(-signal)))
        elif kind == "multiclass":
            cuts = t.get("cuts", [-0.5, 0.5])
            labels = t.get("labels") or [f"class_{k}" for k in range(len(cuts) + 1)]
            row[t["name"]] = labels[sum(signal > c for c in cuts)]
        else:
            row[t["name"]] = round(signal * float(t.get("scale", 1.0)), 4)
        out.append(row)
    return out


def write_csv(cfg: SyntheticDatasetConfig, path: str | Path) -> int:
    rows = rows_for_config(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="generate_synthetic_benchmarks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON config (see module docstring)")
    ap.add_argument("--output", required=True, help="CSV to write")
    args = ap.parse_args(argv)
    cfg = SyntheticDatasetConfig.from_json(args.config)
    n = write_csv(cfg, args.output)
    print(json.dumps({"ok": True, "dataset": cfg.name, "rows": n, "output": str(args.output)}))
    return 0
