"""Scenario answer_bank for a dataset: `python scripts/generate_answer_bank.py --dataset telco_churn [--print]`.

Builds [{match, answer}] from the benchmarks/datasets.json entry and, when present, the seeded
prediction_contract.json (benchmarks/evals/fixtures/seeds/<id>_after_contract/). A headless
agent run answers its clarifying questions from it. Without --print, rewrites the answer_bank
of every scenario for that dataset in benchmarks/evals/scenarios/.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_automl.eval_seed_roles import registry_entry  # noqa: E402


def answer_bank(ds_id: str) -> list[dict]:
    e = registry_entry(ds_id)
    seed = ROOT / "benchmarks/evals/fixtures/seeds" / f"{ds_id}_after_contract" / "prediction_contract.json"
    c = json.loads(seed.read_text(encoding="utf-8")) if seed.is_file() else {}
    split = c.get("split_strategy") or (e.get("split") or {}).get("split_strategy")
    split_desc = split + (f" on {c.get('time_column') or (e.get('split') or {}).get('time_column')}"
                          if split == "temporal" else "")
    ratio = c.get("split_ratio") or (e.get("split") or {}).get("split_ratio")
    bank = [
        {"match": "target column", "answer": f"The target column is {e['target_variable']}"
                                             + (f" (positive class {e['positive_label']!r})." if "positive_label" in e else ".")},
        {"match": "dataset path", "answer": f"Use {{{{REPO_ROOT}}}}/benchmarks/evals/fixtures/datasets/{ds_id}.csv."},
        {"match": "business context", "answer": e.get("business_context") or e["brief"]},
        {"match": "split", "answer": f"Use a {split_desc} split with {ratio:.0%} held out for test." if ratio else
                                     f"Use a {split_desc} split."},
        {"match": "metric", "answer": f"Optimise {e['primary_metric']}; also report "
                                      f"{', '.join(e.get('secondary_metrics', [])) or 'nothing else'}."},
        {"match": "model preference", "answer": "A single baseline model is fine (single_baseline)."},
        {"match": "deployment", "answer": "Exploratory baseline; no production parity needed (open_source engine is fine)."},
        {"match": "approve", "answer": "Yes, approved. Proceed."},
        {"match": "compute", "answer": "Yes, go ahead; keep each run under 10 minutes."},
    ]
    if e.get("entity_id_column"):
        bank.append({"match": "entity", "answer": f"{e['entity_id_column']} identifies the entity; rows repeat per entity."})
    return bank


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--print", action="store_true", dest="print_only")
    args = ap.parse_args(argv)
    bank = answer_bank(args.dataset)
    if args.print_only:
        print(json.dumps(bank, indent=2))
        return 0
    updated = 0
    for p in sorted((ROOT / "benchmarks/evals/scenarios").glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("dataset_id") == args.dataset:
            s["answer_bank"] = bank
            p.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
            updated += 1
            print(f"updated {p.relative_to(ROOT).as_posix()}")
    if not updated:
        print(f"no scenarios for {args.dataset}; use --print to see the bank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
