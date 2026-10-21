"""Audit benchmarks/datasets.json for eval-readiness: `python scripts/audit_benchmark_datasets.py [--json] [--strict]`.

Thin wrapper over agentic_automl.benchmark_readiness.main. Offline: no downloads or GCS calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_automl.benchmark_readiness import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
