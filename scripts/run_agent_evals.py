"""Run agent evals: `python scripts/run_agent_evals.py --tier deterministic --output-dir <dir> --report-md <file>`.

Thin wrapper over agentic_automl.eval_harness.main (see --help for tiers). Paid tiers
(component / e2e / real) need --trust-runtime.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_automl.eval_harness import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
