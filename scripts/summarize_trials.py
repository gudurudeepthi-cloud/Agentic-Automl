"""One row per trial: `python scripts/summarize_trials.py [--json]` (same as `agentic-automl summarize-trials`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_automl.trial_summary import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
