"""Full seed set for one benchmark dataset: `python scripts/generate_dataset_seeds.py --dataset synthetic_fraud`.

Writes the smoke CSV (benchmarks/evals/fixtures/datasets/<id>.csv) and seed trials
<id>_after_contract / _after_profiling / _after_planning / _after_training / _evaluated under
benchmarks/evals/fixtures/seeds/, by really running the step templates (scripts/steps/) with
the trial runtime Python and saving through the validator. Also refreshes the dataset's
features.json and sizing.json. Heavy: needs pandas, sklearn, lightgbm, joblib in that Python.

Get the full data first: `python scripts/download_datasets.py --id <id>`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_automl.eval_seeds import seed_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(seed_cli(None, default_dataset=None, until="evaluated", prog="generate_dataset_seeds",
                              doc=__doc__))
