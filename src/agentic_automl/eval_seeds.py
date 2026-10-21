"""Build per-stage seed trials for a benchmark dataset by really running the step templates.

Stages (each one a directory under benchmarks/evals/fixtures/seeds/<id>_<stage>/):
  after_contract   approved prediction_contract only
  after_profiling  + dataset_profile, column_profile (roles checked against the registry)
  after_planning   + workflow_plan (logistic_regression reference + LightGBM, registry drops)
  after_training   + preprocessing, feature selection, models, training_summary
  evaluated        + evaluation_summary

Everything goes through the real validator (begin-skill / save-artifact), so seeds are
schema-valid and gate-consistent. The step templates run under the trial runtime Python.
Absolute checkout paths are replaced by {{REPO_ROOT}} so the committed seeds are portable.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from agentic_automl import artifacts
from agentic_automl.benchmark_readiness import benchmarks_dir, repo_root
from agentic_automl.contracts import step_templates_dir
from agentic_automl.eval_seed_roles import assert_roles_match_golden, registry_entry

STAGES = ["after_contract", "after_profiling", "after_planning", "after_training", "evaluated"]
REPO_PLACEHOLDER = "{{REPO_ROOT}}"


def data_dir(dataset_id: str, root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "benchmarks" / dataset_id


def source_file(entry: dict, root: Path | None = None) -> Path:
    """Where download_datasets.py puts (or the user placed) the full dataset."""
    d = data_dir(entry["id"], root)
    if entry["access"] == "generated":
        return d / entry["generator"]["output"]
    if entry.get("download_url"):
        return d / entry["download_url"].rsplit("/", 1)[-1]
    found = sorted(d.glob(f"*.{entry['dataset_format']}"))
    if not found:
        raise FileNotFoundError(f"no {entry['dataset_format']} file in {d} (access={entry['access']}: see the brief)")
    return found[0]


def fixtures_dir(root: Path | None = None) -> Path:
    return benchmarks_dir(root) / "evals" / "fixtures"


def make_smoke_csv(entry: dict, root: Path | None = None) -> Path:
    """Seeded row sample of the full data, original row order kept (temporal splits stay meaningful)."""
    src = source_file(entry, root)
    if not src.is_file():
        raise FileNotFoundError(f"{src} missing: run scripts/download_datasets.py --id {entry['id']}")
    with src.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    n = min(entry.get("smoke_rows", 5000), len(body))
    keep = sorted(random.Random(0).sample(range(len(body)), n))
    out = fixtures_dir(root) / "datasets" / f"{entry['id']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(body[i] for i in keep)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_for(entry: dict, data_path: str) -> dict:
    split = {"split_strategy": "random_stratified", "split_ratio": 0.2, "time_column": None, "split_column": None,
             "split_train_value": None, "split_test_value": None, **(entry.get("split") or {})}
    labels = None
    if entry["task_type"] == "binary":
        pos = entry.get("positive_label", 1)
        labels = [("No" if pos == "Yes" else 0), pos] if not isinstance(pos, bool) else [False, True]
    return {
        "task_type": f"{entry['id']} {entry['task_type']} prediction",
        "target_kind": entry["task_type"],
        "business_context": entry.get("business_context") or entry["id"],
        "deployment_intent": "exploratory_baseline",
        "dataset_path": data_path,
        "dataset_location": "local",
        "data_source": "external" if entry["access"] == "public_download" else "internal",
        "data_segments": [{"path": data_path, "format": entry["dataset_format"], "file_count": 1, "role": "full"}],
        "schema_files": [],
        **split,
        "target_column": entry["target_variable"],
        "target_labels": labels,
        "entity_id_column": entry.get("entity_id_column"),
        "weight_column": None,
        "primary_metric": entry["primary_metric"],
        "report_metrics": [m for m in entry.get("secondary_metrics", []) if m != entry["primary_metric"]],
        "model_preference": "single_baseline",
        "model_space": None,
        "scale_assumption": "local_workstation",
        "notes": f"Seed contract generated from benchmarks/datasets.json entry {entry['id']}.",
    }


def plan_for(entry: dict, column_profile: dict, contract: dict) -> dict:
    roles = {c["name"]: c for c in column_profile["columns"]}
    drops = list(dict.fromkeys(
        entry.get("drop_columns", [])
        + [n for n, c in roles.items() if c["role"] in ("id", "time", "drop", "leakage_suspect")]
        + [n for n, c in roles.items() if c["semantic_type"] in ("high_cardinality_categorical", "date_string")]))
    drops = [d for d in drops if d != contract["target_column"]]
    linear = "logistic_regression" if entry["task_type"] != "continuous" else "ridge"
    grid = {"C": [0.1, 1.0]} if linear == "logistic_regression" else {"alpha": [0.3, 3.0]}
    cv = ("5-fold expanding-window TimeSeriesSplit" if contract["split_strategy"] == "temporal"
          else "5-fold stratified (grouped when the entity repeats) CV on train")
    return {
        "objective": f"Seed baseline for {entry['id']}: {entry['primary_metric']} on the held-out test split.",
        "modeling_plan": {
            "candidate_models": [
                {"name": linear, "library": "scikit-learn", "rationale": "Simple, interpretable reference.",
                 "search_space": grid},
                {"name": "lightgbm", "library": "lightgbm", "rationale": "Gradient-boosting contender.",
                 "search_space": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15,
                                  "min_child_samples": 50}},
            ],
            "validation_strategy": cv,
            "primary_metric": entry["primary_metric"],
            "selection_criteria": "Permutation feature selection, then paired per-fold family comparison.",
        },
        "preprocessing_plan": {"engine": "open_source", "drop_columns": drops,
                               "imputation": "Median + missing indicators, fit on train",
                               "categorical_encoding": "One-hot, unknown ignored"},
        "steps": ["data-preprocessing", "training", "evaluation"],
        "risks": ["Smoke-sized data: wide CV error bars."],
        "notes": "Generated by scripts/generate_dataset_seeds.py.",
    }


class SeedBuilder:
    """snapshot=True (seed generation): resample the smoke CSV from the full data, write seed
    directories and dataset docs into benchmarks/. snapshot=False (the eval `pipeline` tier):
    build in `trial_dir` from the committed smoke CSV and write nothing into benchmarks/."""

    def __init__(self, dataset_id: str, runtime_python: str, root: Path | None = None, log=print,
                 trial_dir: Path | None = None, snapshot: bool = True):
        self.root = root or repo_root()
        self.entry = registry_entry(dataset_id, self.root)
        self.py = runtime_python
        self.log = log
        self.write_fixtures = snapshot
        self.tmp = Path(tempfile.mkdtemp(prefix=f"aaml-seed-{dataset_id}-")) if trial_dir is None else None
        self.trial = trial_dir or self.tmp / "trial"

    # -- helpers ----------------------------------------------------------------
    def _run(self, script: str, *args: str) -> str:
        p = subprocess.run([self.py, f"work/{script}", *args], cwd=self.trial, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{script} failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
        return p.stdout

    def _save(self, key: str, skill: str, payload: dict | None = None) -> None:
        payload = payload if payload is not None else json.loads((self.trial / "work" / f"{key}.json").read_text("utf-8"))
        artifacts.save_artifact(self.trial, key, payload, skill)

    def _begin(self, skill: str) -> None:
        r = artifacts.begin_skill(self.trial, skill)
        if not r["allowed"]:
            raise RuntimeError(f"begin-skill {skill} blocked: {r.get('message')}")

    def snapshot(self, stage: str) -> Path:
        if not self.write_fixtures:
            return self.trial
        dest = fixtures_dir(self.root) / "seeds" / f"{self.entry['id']}_{stage}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.trial, dest, ignore=shutil.ignore_patterns(".claude", "__pycache__", "*.pyc"))
        posix, native = self.root.as_posix(), json.dumps(str(self.root))[1:-1]
        for p in dest.rglob("*"):
            if p.suffix in (".json", ".md", ".jsonl") and p.is_file():
                text = p.read_text(encoding="utf-8")
                new = text.replace(posix, REPO_PLACEHOLDER).replace(native, REPO_PLACEHOLDER)
                if new != text:
                    p.write_text(new, encoding="utf-8")
        self.log(f"  seed -> {dest.relative_to(self.root).as_posix()}")
        return dest

    # -- stages -----------------------------------------------------------------
    def build(self, until: str = "evaluated") -> list[Path]:
        from agentic_automl import cli

        stop = STAGES.index(until)
        out = []
        committed = fixtures_dir(self.root) / "datasets" / f"{self.entry['id']}.csv"
        if self.write_fixtures or not committed.is_file():
            smoke = make_smoke_csv(self.entry, self.root)
        else:
            smoke = committed
        self.log(f"smoke CSV: {smoke.relative_to(self.root).as_posix()} ({self.entry.get('smoke_rows')} rows)")
        artifacts.ensure_state(self.trial, goal=self.entry.get("business_context"), dataset_path=smoke.as_posix())
        cli.sync_runtime_files(self.trial)
        for f in step_templates_dir().glob("*.py"):
            if f.name != "features_template.py":
                shutil.copy(f, self.trial / "work" / f.name)

        self._begin("prediction-contract")
        contract = contract_for(self.entry, smoke.as_posix())
        self._save("prediction_contract", "prediction-contract", contract)
        out.append(self.snapshot("after_contract"))
        if stop == 0:
            return out

        self._begin("data-profiling")
        self._run("profile_dataset.py")
        self._save("dataset_profile", "data-profiling")
        self._save("column_profile", "data-profiling")
        cprof = json.loads((self.trial / "column_profile.json").read_text("utf-8"))
        assert_roles_match_golden(cprof, self.entry)
        if self.write_fixtures:
            self._write_dataset_docs(cprof)
        out.append(self.snapshot("after_profiling"))
        if stop == 1:
            return out

        self._begin("workflow-planning")
        self._save("workflow_plan", "workflow-planning", plan_for(self.entry, cprof, contract))
        out.append(self.snapshot("after_planning"))
        if stop == 2:
            return out

        self._begin("data-preprocessing")
        self._run("preprocess.py")
        self._save("preprocessing_summary", "data-preprocessing")
        self._begin("training")
        self._run("feature_selection.py")
        self._run("train.py")
        self._save("training_summary", "training")
        out.append(self.snapshot("after_training"))
        if stop == 3:
            return out

        self._begin("evaluation")
        self._run("evaluate.py")
        self._save("evaluation_summary", "evaluation")
        out.append(self.snapshot("evaluated"))
        return out

    def _write_dataset_docs(self, cprof: dict) -> None:
        d = benchmarks_dir(self.root) / "datasets" / self.entry["id"]
        d.mkdir(parents=True, exist_ok=True)
        features = {"dataset_id": self.entry["id"], "generated_by": "scripts/generate_dataset_seeds.py (smoke sample)",
                    "columns": [{"name": c["name"], "role": c["role"], "type": c["semantic_type"],
                                 "missing_fraction": c["missing_fraction"]} for c in cprof["columns"]]}
        (d / "features.json").write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
        dprof = json.loads((self.trial / "dataset_profile.json").read_text("utf-8"))
        src = source_file(self.entry, self.root)
        with src.open(encoding="utf-8") as fh:
            full_rows = sum(1 for _ in fh) - 1
        sizing = {"dataset_id": self.entry["id"], "rows": full_rows, "columns": dprof["n_columns"],
                  "file_size_mb": round(src.stat().st_size / 1e6, 2),
                  "positive_rate": dprof["target_summary"].get("positive_rate"),
                  "smoke_rows": dprof["n_rows"]}
        (d / "sizing.json").write_text(json.dumps(sizing, indent=2) + "\n", encoding="utf-8")

    def cleanup(self) -> None:
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)


def seed_cli(argv: list[str] | None, *, default_dataset: str | None, until: str, prog: str, doc: str) -> int:
    import argparse

    from agentic_automl.state import runtime_python

    ap = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=default_dataset, required=default_dataset is None)
    ap.add_argument("--until", choices=STAGES, default=until)
    ap.add_argument("--python", default=None, help="runtime python with the ML stack (default: configured)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary build trial")
    args = ap.parse_args(argv)
    b = SeedBuilder(args.dataset, args.python or runtime_python())
    try:
        seeds = b.build(args.until)
    except (FileNotFoundError, RuntimeError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            b.cleanup()
    print(json.dumps({"ok": True, "dataset": args.dataset, "seeds": [s.name for s in seeds]}))
    return 0
