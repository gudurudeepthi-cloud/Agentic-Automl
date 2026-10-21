"""Real after-training seed (preprocessed parquet, feature selection, model.joblib, training_summary):
`python scripts/generate_training_seed.py [--dataset telco_churn]`.

Same pipeline as generate_dataset_seeds.py, stopped after training (writes every stage up to
<id>_after_training). Lets an evaluation scenario start mid-chain without a real upstream run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_automl.eval_seeds import seed_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(seed_cli(None, default_dataset="telco_churn", until="after_training",
                              prog="generate_training_seed", doc=__doc__))
