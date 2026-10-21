"""Training: read feature_selection.json, choose the family, refit on full train, persist.

Template (copy to work/ with aaml_common.py). Run work/feature_selection.py first.
Train split only; test is never read.

Family choice (paired per fold, same folds for every family): start from the simplest
candidate (aaml_common.COMPLEXITY order) and move to a more complex one only when its
mean per-fold gain is larger than the std of that gain.

Writes outputs/model/model.joblib (winner), model_<family>.joblib (others),
outputs/model/cv_predictions.parquet (winner's validation-fold predictions, used by
evaluation to pick a threshold without touching test), work/training_summary.json.

    <runtime python> work/train.py [--folds 5] [--no-cv-predictions]
"""
import argparse
import json
import time

import joblib

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--folds", type=int, default=5, help="must match the feature_selection.py run")
ap.add_argument("--selection", default="outputs/model/feature_selection.json")
ap.add_argument("--no-cv-predictions", action="store_true")
args = ap.parse_args()

d = A.Data.load(n_splits=args.folds)
METRIC = d.metric
sel = A.read_json(args.selection)
assert sel["primary_metric"] == METRIC, "feature_selection.json was computed for a different metric"
fam = sel["families"]
chosen = {f: (i["reduced"] if i["use_reduced"] else i["full"]) for f, i in fam.items()}
assert all(len(r["fold_scores"]) == len(d.folds) for r in chosen.values()), "fold count differs from selection run"

order = sorted(fam, key=lambda f: A.COMPLEXITY.index(f) if f in A.COMPLEXITY else len(A.COMPLEXITY))
winner, steps = order[0], []
for challenger in order[1:]:
    gap, std, per_fold = A.paired_gap(chosen[challenger]["fold_scores"], chosen[winner]["fold_scores"], METRIC)
    won = gap > std
    steps.append(f"{challenger} vs {winner}: {gap:+.4f} +/- {std:.4f} per fold {per_fold} -> "
                 f"{'switch' if won else 'keep ' + winner}")
    if won:
        winner = challenger
why = ("Only one candidate." if len(order) == 1 else
       "Paired per-fold comparison, simplest first; a more complex family must beat the incumbent by more than "
       "the std of the per-fold gap. " + "; ".join(steps) + f". Selected {winner}.")
print(why, flush=True)

models = []
for family, info in fam.items():
    cols = d.cols_for(info["features"])
    t0 = time.time()
    model = d.fit(d.make(family, info["params"], cols), d.train[cols], d.y, d.weight)
    path = "outputs/model/model.joblib" if family == winner else f"outputs/model/model_{family}.joblib"
    joblib.dump({"family": family, "model": model, "features": cols, "raw_features": info["features"],
                 "params": info["params"], "task": d.task, "target_labels": d.contract.get("target_labels")}, path)
    models.append({"name": family, "library": A.model_library(family), "hyperparameters": info["params"],
                   "cv_score": round(chosen[family]["score"], 5), "cv_score_std": round(chosen[family]["score_std"], 5),
                   "train_time_seconds": round(time.time() - t0, 1), "artifact_path": path})
    print(f"{family}: {len(info['features'])} raw features / {len(cols)} columns -> {path}", flush=True)

notes = [f"Feature selection: work/feature_selection.py -> {args.selection} (rule '{sel['rule']}', "
         f"redundancy removed {list(sel['redundancy_filter'])})."]
if not args.no_cv_predictions:
    info = fam[winner]
    r = d.cv(winner, info["features"], info["params"], keep_predictions=True)
    r["predictions"].to_parquet("outputs/model/cv_predictions.parquet", index=False)
    notes.append("Winner's validation-fold predictions: outputs/model/cv_predictions.parquet "
                 "(for choosing a threshold without test).")
other = chosen[winner].get("other_metrics") or {}
if other:
    notes.append("Winner CV: " + ", ".join(f"{k} {v:.4f}" for k, v in other.items()) + ".")
notes.append("Test split not used.")

sel["family_decision"] = {"winner": winner, "steps": steps, "rationale": why}
A.write_json(args.selection.replace(".json", "_decision.json"), sel["family_decision"])
summary = {
    "models": models,
    "selected_model": winner,
    "selection_rationale": why + f" Features: {len(fam[winner]['features'])} of {len(d.raw)} raw after selection.",
    "primary_metric": METRIC,
    "cv_strategy": d.cv_desc,
    "training_script": "work/train.py",
    "model_artifact_path": "outputs/model/model.joblib",
    "random_seed": A.SEED,
    "notes": " ".join(notes),
}
A.write_json("work/training_summary.json", summary)
print(json.dumps({m["name"]: [m["cv_score"], m["cv_score_std"]] for m in models}))
