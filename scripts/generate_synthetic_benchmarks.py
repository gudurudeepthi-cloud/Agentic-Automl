"""Deterministic synthetic CSV from a JSON config:
`python scripts/generate_synthetic_benchmarks.py --config cfg.json --output out.csv`.

Thin wrapper over agentic_automl.synthetic_benchmarks (stdlib only; see its docstring for the config).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_automl.synthetic_benchmarks import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
