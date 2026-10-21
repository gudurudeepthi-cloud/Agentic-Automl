"""Profiling seeds only (contract + schema-valid dataset_profile / column_profile):
`python scripts/generate_seed_profiles.py [--dataset telco_churn]`.

Same pipeline as generate_dataset_seeds.py, stopped after profiling: writes
<id>_after_contract and <id>_after_profiling. Cheap (no training).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_automl.eval_seeds import seed_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(seed_cli(None, default_dataset="telco_churn", until="after_profiling",
                              prog="generate_seed_profiles", doc=__doc__))
