"""Feature selection on CV (train only; test is never read) -> outputs/model/feature_selection.json.

Template (copy to work/ with aaml_common.py). Run it before train.py, which reads its output.

Stage 0  record what the plan already removed (drop_columns + their profile roles)
Stage 1  redundancy: |Spearman| > --rho between numeric features; keep the original over an
         engineered transform, else the stronger univariate signal
Stage 2  per candidate family: tune list-valued search_space entries on CV (small grid), then
         permutation importance of every raw feature on every validation fold (a feature's
         encoded columns are shuffled together)
Stage 3  drop per --rule, re-score, keep the reduced set only if within 1 std of the full set

--rule mean    drop if mean importance <= 0 (default)
--rule stable  drop unless mean > 0, positive in >= 75% of folds, and mean > its fold std

    <runtime python> work/feature_selection.py --smoke      # one fold per family + runtime estimate
    <runtime python> work/feature_selection.py              # full run (background it if the estimate is long)
"""
import argparse
import time

import numpy as np

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--rule", choices=["mean", "stable"], default="mean")
ap.add_argument("--rho", type=float, default=0.95)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--repeats", type=int, default=3, help="permutation repeats per feature per fold")
ap.add_argument("--families", default=None, help="comma list (default: plan candidate_models)")
ap.add_argument("--smoke", action="store_true", help="time one fold per family, print an estimate, write nothing")
ap.add_argument("--out", default="outputs/model/feature_selection.json")
args = ap.parse_args()

rng = np.random.default_rng(A.SEED)
d = A.Data.load(n_splits=args.folds)
METRIC = d.metric
families = args.families.split(",") if args.families else list(d.candidates)
assert families, "no candidate_models in the plan"

# --- smoke: one fold, one fit per family, then extrapolate -------------------------------
if args.smoke:
    total = 0.0
    for fam in families:
        grid = A.param_grid(d.candidates.get(fam, {}).get("search_space"))
        t0 = time.time()
        r = d.cv(fam, d.raw, grid[0], folds=[d.folds[-1]])
        fit = time.time() - t0
        n_fits = len(d.folds) * (len(grid) + 2)  # grid + full-with-permutation + reduced
        perm = fit * 0.15 * len(d.raw) * args.repeats * len(d.folds)  # predict-only, rough
        est = fit * n_fits + perm
        total += est
        print(f"{fam}: one fold {fit:.1f}s ({METRIC} {r['score']:.4f}); grid {len(grid)}; "
              f"estimated full run ~{est / 60:.1f} min", flush=True)
    print(f"TOTAL estimated ~{total / 60:.1f} min for {len(families)} families x {len(d.folds)} folds")
    raise SystemExit(0)

# --- stage 0 --------------------------------------------------------------------------
prof_roles = {col["name"]: f"{col['role']} ({col['semantic_type']})" for col in A.column_profile()["columns"]}
stage0 = {col: prof_roles.get(col, "plan") for col in d.plan["preprocessing_plan"]["drop_columns"]}
print(f"stage 0 (plan) removed: {stage0}")
print(f"candidates: {len(d.raw)} raw features ({len(d.engineered)} engineered); CV: {d.cv_desc}", flush=True)

# --- stage 1 --------------------------------------------------------------------------
numeric = [f for f in d.raw if f not in d.cats]


def univariate(f):
    from sklearn.metrics import roc_auc_score

    x = d.train[d.groups[f][0]]
    if x.nunique() < 2:
        return 0.0
    if d.task == "binary":
        a = roc_auc_score(d.y, x)
        return max(a, 1 - a)
    return abs(float(np.corrcoef(x, d.y if d.task == "regression" else d.y.astype(float))[0, 1]))


uni = {f: univariate(f) for f in numeric}
cols0 = {f: d.groups[f][0] for f in numeric}  # the value column (not the missing indicator)
rho = d.train[[cols0[f] for f in numeric]].corr(method="spearman").abs() if numeric else None
redundant = {}
for i, a in enumerate(numeric):
    for b in numeric[i + 1:]:
        if a in redundant or b in redundant:
            continue
        r = rho.loc[cols0[a], cols0[b]]
        if r > args.rho:
            ea, eb = a in d.engineered, b in d.engineered
            drop = b if eb and not ea else a if ea and not eb else (a if uni[a] < uni[b] else b)
            redundant[drop] = {"duplicate_of": b if drop == a else a, "spearman": round(float(r), 4)}
pool = [f for f in d.raw if f not in redundant]
print(f"stage 1 (redundancy) removed: {redundant or 'nothing'} -> {len(pool)} features", flush=True)


def removable(imp):
    if args.rule == "mean":
        return [f for f in pool if imp[f]["mean"] <= 0]
    out = []
    for f in pool:
        folds = np.array(imp[f]["per_fold"])
        if not (imp[f]["mean"] > 0 and (folds > 0).mean() >= 0.75 and imp[f]["mean"] > folds.std()):
            out.append(f)
    return out


# --- stages 2-3 -------------------------------------------------------------------------
sign = 1 if A.greater_is_better(METRIC) else -1
out_families = {}
for fam in families:
    print(f"stage 2-3: {fam}", flush=True)
    grid = A.param_grid(d.candidates.get(fam, {}).get("search_space"))
    tuning = None
    if len(grid) > 1:  # tune here so importance is measured on the model we keep
        tuning = []
        for params in grid:
            r = d.cv(fam, pool, params)
            tuning.append({"params": params, "score": r["score"], "score_std": r["score_std"]})
            A.show(str({k: v for k, v in params.items() if isinstance(d.candidates[fam]["search_space"].get(k), list)}),
                   r, METRIC)
        params = max(tuning, key=lambda t: sign * t["score"])["params"]
    else:
        params = grid[0]
    full = d.cv(fam, pool, params, permute=True, rng=rng, repeats=args.repeats)
    A.show(f"all {len(pool)} features (+permutation)", full, METRIC)
    drop = removable(full["permutation_importance"])
    selected = [f for f in pool if f not in drop]
    reduced = d.cv(fam, selected, params) if drop and selected else full
    A.show(f"selected {len(selected)} features", reduced, METRIC)
    keep = bool(drop) and bool(selected) and sign * (reduced["score"] - full["score"]) >= -full["score_std"]
    out_families[fam] = {"params": params, "tuning": tuning, "full": full, "removed": drop, "reduced": reduced,
                         "use_reduced": keep, "features": selected if keep else pool}
    print(f"   -> {'reduced' if keep else 'full'} set ({len(out_families[fam]['features'])}); "
          f"removed {drop if keep else []}", flush=True)

A.write_json(args.out, {
    "rule": args.rule,
    "primary_metric": METRIC,
    "cv_strategy": f"{d.cv_desc}; permutation repeats={args.repeats}, encoded columns permuted jointly",
    "stage0_plan_removed": stage0,
    "redundancy_filter": redundant,
    "candidate_pool": pool,
    "families": out_families,
    "note": "Selection, tuning and family choice share these folds (mild optimism); test is the honest check.",
})

print(f"\n{'feature':<30}" + "".join(f"{f[:10]:>12}" for f in families))
first = out_families[families[0]]["full"]["permutation_importance"]
for f in sorted(pool, key=lambda f: -first[f]["mean"]):
    row = "".join(f"{out_families[m]['full']['permutation_importance'][f]['mean']:+11.4f}"
                  f"{'' if f in out_families[m]['features'] else '*'}" for m in families)
    print(f"{f:<30}{row}")
print("(* = dropped)")
for f, r in redundant.items():
    print(f"{f:<30} redundant with {r['duplicate_of']} (rho {r['spearman']})")
print(f"\nwrote {args.out}")
