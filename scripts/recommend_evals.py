"""Which eval tiers does a diff need? `python scripts/recommend_evals.py --base origin/main --head HEAD`.

Recommend-only: prints MUST / CHECK tiers for the changed paths; never runs a paid tier.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from agentic_automl.eval_harness import recommend_evals_cli  # noqa: E402
except ImportError as exc:  # the branch predates the harness
    raise SystemExit(f"agentic_automl.eval_harness.recommend_evals_cli not found ({exc}). "
                     "Your branch predates the eval harness: rebase on origin/main and retry.")

if __name__ == "__main__":
    raise SystemExit(recommend_evals_cli())
