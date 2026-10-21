"""Planning evidence: characterise the data, probe engineered features, screen model families.

Template (copy to work/ with aaml_common.py). TRAIN split only, encoded with the same
shared encoder and the same CV folds that training will use, so the screen and training
agree. Writes work/model_family_advice.json.

    <runtime python> work/model_family_screen.py [--families logistic_regression,random_forest,lightgbm]
                                                 [--folds 5] [--drop col1,col2] [--max-rows 300000]

--drop lists the columns the plan will drop (ids, leakage, high-cardinality); profile
roles id/time/target/drop/leakage_suspect are excluded automatically.
"""
import argparse
import importlib.util
import json

import numpy as np

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--families", default=None, help="comma list (default: linear, random_forest, boosting)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--drop", default="", help="extra columns to exclude (comma list)")
ap.add_argument("--max-onehot-levels", type=int, default=100)
ap.add_argument("--max-rows", type=int, default=None, help="screen on the most recent / a seeded sample of rows")
args = ap.parse_args()

c, cprof = A.contract(), A.column_profile()
T = A.task(c)
METRIC = c["primary_metric"]
train, _, _ = A.split(c)  # test is discarded here
if args.max_rows and len(train) > args.max_rows:
    train = (train.tail(args.max_rows) if c["split_strategy"] == "temporal"
             else train.sample(args.max_rows, random_state=A.SEED).sort_index()).reset_index(drop=True)
F = A.load_features_module()
engineered = list(getattr(F, "ENGINEERED", []) or [])

roles = {col["name"]: col["role"] for col in cprof["columns"]}
excluded = set(filter(None, args.drop.split(","))) | {c["target_column"]} | {
    k for k in (c.get("time_column"), c.get("entity_id_column"), c.get("weight_column"), c.get("split_column")) if k}
features = [col for col in train.columns if col not in excluded and roles.get(col, "feature") == "feature"]
numeric, cats = A.split_columns(train, features, cprof)
wide = [f for f in cats if train[f].nunique() > args.max_onehot_levels]
features = [f for f in features if f not in wide]
print(f"screening {len(features)} features (skipped high-cardinality {wide})", flush=True)

d = A.Data.from_raw(c, train, features, cprof, engineered, n_splits=args.folds)

# --- 1. data characteristics -------------------------------------------------
numeric = [f for f in features if f not in d.cats]
skew = train[numeric].skew(numeric_only=True)
character = {
    "train_rows": int(len(train)), "n_features": len(features), "n_numeric": len(numeric),
    "n_categorical": len(d.cats), "encoded_columns": int(sum(map(len, d.groups.values()))),
    "max_cardinality": int(max((train[f].nunique() for f in d.cats), default=0)),
    "features_with_missing": {f: round(float(v), 4) for f, v in train[features].isna().mean().items() if v > 0},
    "highly_skewed_numeric": sorted(skew[skew.abs() > 2].index.tolist()),
}
if T == "binary":
    character.update(positives=int(d.y.sum()), positive_rate=round(float(d.y.mean()), 5))


# --- 2. univariate probe (train, orientation-free) ---------------------------
def probe(f):
    from sklearn.metrics import roc_auc_score

    x = train[f]
    if f in d.cats or x.nunique() < 2:
        return None
    x = x.fillna(x.median())
    if T == "binary":
        a = roc_auc_score(d.y, x)
        return round(float(max(a, 1 - a)), 4)
    if T == "regression":
        return round(float(abs(np.corrcoef(x, d.y)[0, 1])), 4)
    return None


engineered_probe = {f: probe(f) for f in engineered if f in features}
base_probe = {f: probe(f) for f in numeric if f not in engineered}

# --- 3. family screen ------------------------------------------------------------
linear = "logistic_regression" if T != "regression" else "ridge"
boosting = "lightgbm" if importlib.util.find_spec("lightgbm") else "hist_gradient_boosting"
families = args.families.split(",") if args.families else [linear, "random_forest", boosting]
screen = {}
for fam in families:
    r = d.cv(fam, features, {})
    screen[fam] = {"score": round(r["score"], 5), "score_std": round(r["score_std"], 5),
                   "fold_scores": r["fold_scores"], "seconds": r["seconds"]}
    A.show(fam, r, METRIC)

# --- 4. recommendation (paired per fold) ---------------------------------------------
sign = 1 if A.greater_is_better(METRIC) else -1
ranked = sorted(screen, key=lambda k: -sign * screen[k]["score"])
best = ranked[0]
ref = linear if linear in screen else ranked[-1]
if best == ref and len(ranked) > 1:  # the simple reference won outright: report the gap to the runner-up
    best = ranked[1]
gap, gap_std, per_fold = A.paired_gap(screen[best]["fold_scores"], screen[ref]["fold_scores"], METRIC)
if best == ref or gap <= gap_std:
    rec = ref
    verdict = "worse than" if gap < -gap_std else "not distinguishable from"
    why = (f"Paired per-fold gain of {best} over {ref} = {gap:+.4f} +/- {gap_std:.4f} (folds {per_fold}; "
           f"positive = {best} better): {best} is {verdict} {ref}, so prefer the simpler, more interpretable "
           f"{ref}; keep {best} as a contender for training to confirm.")
else:
    rec = best
    why = (f"{best} beats {ref} by {gap:+.4f} per fold (> std {gap_std:.4f}; folds {per_fold}): the signal "
           "looks non-linear or interaction-driven.")
    if len(ranked) > 1:
        g2, s2, _ = A.paired_gap(screen[best]["fold_scores"], screen[ranked[1]]["fold_scores"], METRIC)
        why += (f" Runner-up {ranked[1]} trails by {g2:+.4f} +/- {s2:.4f}"
                + (" (within noise: a viable alternative)." if g2 <= s2 else "."))
advice = {
    "cv": d.cv_desc,
    "primary_metric": METRIC,
    "data_characteristics": character,
    "engineered_feature_probe": engineered_probe,
    "base_feature_probe": base_probe,
    "family_screen": screen,
    "ranking": ranked,
    "paired_gap_best_minus_reference": {"best": best, "reference": ref, "mean": round(gap, 5),
                                        "std": round(gap_std, 5), "per_fold": per_fold},
    "recommended_family": rec,
    "rationale": why,
    "reference_baseline": ref,
    "notes": "Families run with library defaults (aaml_common.make_model); tune in training.",
}
A.write_json("work/model_family_advice.json", advice)
print("\nengineered probe:", json.dumps(engineered_probe))
print("ranking:", ranked)
print("RECOMMEND:", rec, "-", why)
