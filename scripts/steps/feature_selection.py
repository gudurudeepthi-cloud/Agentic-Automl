"""Feature selection on CV (train only; test is never read) -> outputs/model/feature_selection.json.

Template (copy to work/ with aaml_common.py). Run it before train.py, which reads its output.

Stage 0  record what the plan already removed (drop_columns + their profile roles)
Stage 1  redundancy: |Spearman| > --rho between numeric features; keep the original over an
         engineered transform, else the stronger univariate signal
Stage 1b filter methods (--filters, comma list, run in this order; '' = none):
           quasi_constant  one value covers >= --max-dominant of train rows
           bivariate       feature vs target per bin (quantile bins / category levels / missing):
                           binary -> event rate, WoE and Information Value (drop IV < --min-iv);
                           multiclass -> Cramer's V, regression -> correlation ratio eta
                           (drop below --min-signal). Full tables -> outputs/model/bivariate.json,
                           IV > --iv-flag (1.0) flagged as a leakage question
           mutual_info     sklearn mutual information with the target (drop < --min-mi)
           vif             iterative variance inflation on numerics (drop the max while > --vif);
                           mostly matters for linear models
           l1              L1 logistic / Lasso per CV fold on scaled features (keep a feature
                           non-zero in >= --l1-min-folds of folds)
         The filtered pool is kept only if CV (first family) is within 1 std of the unfiltered pool.
Stage 2  per candidate family: tune list-valued search_space entries on CV (small grid), then
         permutation importance of every raw feature on every validation fold (a feature's
         encoded columns are shuffled together)
Stage 3  drop per --rule, re-score, keep the reduced set only if within 1 std of the full set

--rule mean    drop if mean importance <= 0 (default)
--rule stable  drop unless mean > 0, positive in >= 75% of folds, and mean > its fold std

    <runtime python> work/feature_selection.py --smoke      # one fold per family + runtime estimate
    <runtime python> work/feature_selection.py              # full run (background it if the estimate is long)
    <runtime python> work/feature_selection.py --filters-only   # stages 0-1b + bivariate report, no models

Required features (contract `required_features`) are exempt from every stage.
"""
import argparse
import time

import numpy as np
import pandas as pd

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--rule", choices=["mean", "stable"], default="mean")
ap.add_argument("--rho", type=float, default=0.95)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--repeats", type=int, default=3, help="permutation repeats per feature per fold")
ap.add_argument("--families", default=None, help="comma list (default: plan candidate_models)")
ap.add_argument("--smoke", action="store_true", help="time one fold per family, print an estimate, write nothing")
ap.add_argument("--filters", default="quasi_constant,bivariate",
                help="comma list: quasi_constant, bivariate, mutual_info, vif, l1 ('' = none)")
ap.add_argument("--filters-only", action="store_true",
                help="run stages 0-1b, write the bivariate report, print what would be removed; no models")
ap.add_argument("--max-dominant", type=float, default=0.995)
ap.add_argument("--bins", type=int, default=10, help="bivariate: quantile bins for numerics")
ap.add_argument("--min-iv", type=float, default=0.02, help="bivariate, binary: drop below this Information Value")
ap.add_argument("--iv-flag", type=float, default=1.0,
                help="bivariate, binary: flag IV above this as a leakage question (the profile's probe is the hard check)")
ap.add_argument("--min-signal", type=float, default=0.05,
                help="bivariate, multiclass/regression: drop below this Cramer's V / eta")
ap.add_argument("--min-mi", type=float, default=0.001)
ap.add_argument("--vif", type=float, default=10.0)
ap.add_argument("--l1-c", type=float, default=0.1, help="l1: inverse regularisation (Lasso alpha = 1/(C*n))")
ap.add_argument("--l1-min-folds", type=float, default=0.5)
ap.add_argument("--out", default="outputs/model/feature_selection.json")
ap.add_argument("--bivariate-out", default="outputs/model/bivariate.json")
args = ap.parse_args()
FILTERS = [f.strip() for f in args.filters.split(",") if f.strip()]
KNOWN = ["quasi_constant", "bivariate", "mutual_info", "vif", "l1"]
assert set(FILTERS) <= set(KNOWN), f"unknown --filters {sorted(set(FILTERS) - set(KNOWN))}; choose from {KNOWN}"

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
REQUIRED = [f for f in d.contract.get("required_features") or [] if f in d.raw]  # user asked to keep these
gone = sorted(set(d.contract.get("required_features") or []) - set(REQUIRED))
assert not gone, f"required_features missing before selection (dropped in the plan or profile?): {gone}"
for f in REQUIRED:
    redundant.pop(f, None)
pool = [f for f in d.raw if f not in redundant]
print(f"stage 1 (redundancy) removed: {redundant or 'nothing'} -> {len(pool)} features"
      + (f"; always kept (contract): {REQUIRED}" if REQUIRED else ""), flush=True)


# --- stage 1b: filter methods -----------------------------------------------------------
def binned(f):
    """(bin label per train row, bin order, ordered?) for a raw feature, rebuilt from its encoded columns."""
    g = d.groups[f]
    if f in d.cats:
        X = d.train[g].to_numpy()
        levels = np.array([col[len(f) + 1:] for col in g])
        b = pd.Series(np.where(X.max(1) > 0, levels[X.argmax(1)], "<unseen>"), index=d.train.index)
        share = b.map(b.value_counts(normalize=True))
        b = b.where(share >= 0.01, "<rare>")
        return b, list(b.value_counts().index), False
    x = d.train[g[0]]
    miss = d.train[g[1]].astype(bool) if len(g) > 1 else pd.Series(False, index=x.index)
    present = x[~miss]
    if present.nunique() <= args.bins:
        order = [str(v) for v in sorted(present.unique())]
        b = x.astype(str)
    else:  # quantile edges on values: heavily tied values share a bin instead of splitting arbitrarily
        edges = np.unique(np.quantile(present, np.linspace(0, 1, args.bins + 1)))
        cut = pd.cut(present, edges, include_lowest=True, duplicates="drop")
        labels = {iv: f"({iv.left:.4g}, {iv.right:.4g}]" for iv in cut.cat.categories}
        b = pd.Series("<missing>", index=x.index, dtype=object)
        b[~miss] = cut.map(labels).astype(str)
        order = list(labels.values())
    b = b.where(~miss, "<missing>")
    return b, order + (["<missing>"] if miss.any() else []), True


def bivariate(f):
    b, order, ordered = binned(f)
    tab = pd.DataFrame({"bin": b.to_numpy(), "y": d.y})
    sizes = tab.groupby("bin")["y"].size().reindex(order).dropna()
    rows, out = [], {"feature": f, "type": "numeric" if ordered else "categorical", "n_bins": int(len(sizes))}
    if d.task == "binary":
        ev = tab.groupby("bin")["y"].sum().reindex(sizes.index)
        pos, neg = d.y.sum(), len(d.y) - d.y.sum()
        de, dn = (ev + 0.5) / (pos + 0.5 * len(sizes)), (sizes - ev + 0.5) / (neg + 0.5 * len(sizes))
        woe = np.log(de / dn)
        iv = (de - dn) * woe
        rows = [{"bin": k, "n": int(sizes[k]), "events": int(ev[k]), "event_rate": round(float(ev[k] / sizes[k]), 5),
                 "woe": round(float(woe[k]), 4), "iv": round(float(iv[k]), 5)} for k in sizes.index]
        out.update(statistic="information_value", strength=round(float(iv.sum()), 5), trend_key="event_rate")
    elif d.task == "regression":
        means = tab.groupby("bin")["y"].mean().reindex(sizes.index)
        total = ((d.y - d.y.mean()) ** 2).sum()
        eta = np.sqrt(float((sizes * (means - d.y.mean()) ** 2).sum() / total)) if total > 0 else 0.0
        rows = [{"bin": k, "n": int(sizes[k]), "target_mean": round(float(means[k]), 5)} for k in sizes.index]
        out.update(statistic="correlation_ratio_eta", strength=round(eta, 5), trend_key="target_mean")
    else:
        from scipy.stats import chi2_contingency

        ct = pd.crosstab(tab["bin"], tab["y"]).reindex(sizes.index)
        chi2 = chi2_contingency(ct)[0] if min(ct.shape) > 1 else 0.0
        v = np.sqrt(chi2 / (len(tab) * max(1, min(ct.shape) - 1)))
        rates = ct.div(ct.sum(1), axis=0)
        rows = [{"bin": k, "n": int(sizes[k]), "class_rates": {str(c): round(float(r), 4) for c, r in rates.loc[k].items()}}
                for k in sizes.index]
        out.update(statistic="cramers_v", strength=round(float(v), 5), trend_key=None)
    if ordered and out["trend_key"]:
        trend = [r[out["trend_key"]] for r in rows if r["bin"] != "<missing>"]
        if len(trend) > 2:
            rho_t = pd.Series(trend).corr(pd.Series(range(len(trend))), method="spearman")
            out["monotonic_trend"] = None if pd.isna(rho_t) else round(float(rho_t), 3)
    out["bins"] = rows
    return out


def sample_idx(n_max=200_000):
    n = len(d.train)
    return np.sort(rng.choice(n, n_max, replace=False)) if n > n_max else np.arange(n)


def run_filter(name, cand):
    """{feature: reason} to remove from `cand` (required features never appear)."""
    free = [f for f in cand if f not in REQUIRED]
    if name == "quasi_constant":
        out = {}
        for f in free:
            top = (binned(f)[0] if f in d.cats else d.train[d.groups[f][0]]).value_counts(normalize=True).iloc[0]
            if top >= args.max_dominant:
                out[f] = f"one value covers {top:.2%} of train rows"
        return out
    if name == "bivariate":
        cut = args.min_iv if d.task == "binary" else args.min_signal
        return {f: f"{BIV[f]['statistic']} {BIV[f]['strength']} < {cut}" for f in free if BIV[f]["strength"] < cut}
    if name == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        idx = sample_idx()
        X = pd.DataFrame({f: (pd.factorize(binned(f)[0])[0] if f in d.cats else d.train[d.groups[f][0]].to_numpy())
                          for f in free}).iloc[idx]
        fn = mutual_info_regression if d.task == "regression" else mutual_info_classif
        mi = fn(X, d.y[idx], discrete_features=[f in d.cats for f in free], random_state=A.SEED)
        MI.update({f: round(float(v), 5) for f, v in zip(free, mi)})
        return {f: f"mutual information {MI[f]} < {args.min_mi}" for f in free if MI[f] < args.min_mi}
    if name == "vif":
        num = [f for f in cand if f not in d.cats]
        Z = d.train[[d.groups[f][0] for f in num]].to_numpy(dtype=float)
        keep = [i for i in range(len(num)) if Z[:, i].std() > 0]
        out = {}
        while len(keep) > 2:
            R = np.corrcoef(Z[:, keep], rowvar=False)
            vif = np.diag(np.linalg.pinv(R))
            order = [j for j in np.argsort(-vif) if num[keep[j]] not in REQUIRED]
            if not order or vif[order[0]] <= args.vif:
                break
            out[num[keep[order[0]]]] = f"VIF {vif[order[0]]:.1f} > {args.vif}"
            keep.pop(order[0])
        return out
    if name == "l1":
        import sklearn
        from sklearn.linear_model import Lasso, LogisticRegression
        from sklearn.preprocessing import StandardScaler

        cols = d.cols_for(cand)
        nonzero = {f: 0 for f in cand}
        for fit_idx, _ in d.folds:
            X = StandardScaler().fit_transform(d.train[cols].iloc[fit_idx])
            y = d.y[fit_idx]
            if d.task == "regression":
                m = Lasso(alpha=1.0 / (args.l1_c * len(y)), max_iter=5000).fit(X, (y - y.mean()) / (y.std() or 1))
                coef = np.abs(m.coef_).reshape(1, -1)
            else:  # sklearn >= 1.8 spells L1 as l1_ratio=1; older versions need penalty="l1"
                l1 = {"l1_ratio": 1.0} if tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 8) \
                    else {"penalty": "l1"}
                m = LogisticRegression(C=args.l1_c, solver="liblinear" if d.task == "binary" else "saga",
                                       max_iter=5000, random_state=A.SEED, **l1).fit(X, y)
                coef = np.abs(m.coef_)
            alive = dict(zip(cols, coef.max(0) > 1e-8))
            for f in cand:
                nonzero[f] += any(alive[col] for col in d.groups[f])
        share = {f: nonzero[f] / len(d.folds) for f in cand}
        L1.update({f: round(s, 3) for f, s in share.items()})
        return {f: f"non-zero L1 coefficient in {share[f]:.0%} of folds < {args.l1_min_folds:.0%}"
                for f in free if share[f] < args.l1_min_folds}
    raise ValueError(name)


BIV = {f: bivariate(f) for f in d.raw} if ("bivariate" in FILTERS or args.filters_only) else {}
MI, L1 = {}, {}
filter_removed, before_filters = {}, list(pool)
for name in FILTERS:
    gone_f = run_filter(name, pool)
    filter_removed[name] = gone_f
    pool = [f for f in pool if f not in gone_f]
    print(f"stage 1b {name:<15} removed {len(gone_f)}: {list(gone_f) or '-'} -> {len(pool)} features", flush=True)
if BIV:
    biv_flags = [{"feature": f, "message": f"IV {b['strength']} > {args.iv_flag}: very strong alone; worth confirming "
                                          "it is known before the outcome"} for f, b in BIV.items()
                 if b["statistic"] == "information_value" and b["strength"] > args.iv_flag]
    A.write_json(args.bivariate_out, {"target": d.target, "task": d.task, "flags": biv_flags,
                                      "features": sorted(BIV.values(), key=lambda b: -b["strength"])})
    print(f"\nbivariate ({next(iter(BIV.values()))['statistic']}; wrote {args.bivariate_out}):")
    for b in sorted(BIV.values(), key=lambda b: -b["strength"]):
        mono = f"  trend {b['monotonic_trend']:+.2f}" if b.get("monotonic_trend") is not None else ""
        print(f"  {b['feature']:<30} {b['strength']:>9.4f}  bins {b['n_bins']:<3}{mono}")
    for fl in biv_flags:
        print(f"[info] {fl['feature']}: {fl['message']}")
filters_check = None
if args.filters_only:
    print(f"\n--filters-only: would keep {len(pool)} of {len(before_filters)}; feature_selection.json not written")
    raise SystemExit(0)
if len(pool) < len(before_filters):
    fam0 = families[0]
    p0 = A.param_grid(d.candidates.get(fam0, {}).get("search_space"))[0]
    r_all, r_f = d.cv(fam0, before_filters, p0), d.cv(fam0, pool, p0)
    ok = bool(pool) and (1 if A.greater_is_better(METRIC) else -1) * (r_f["score"] - r_all["score"]) \
        >= -r_all["score_std"]
    filters_check = {"family": fam0, "unfiltered": {"n": len(before_filters), "score": r_all["score"],
                                                     "score_std": r_all["score_std"]},
                     "filtered": {"n": len(pool), "score": r_f["score"], "score_std": r_f["score_std"]},
                     "kept_filtered": ok}
    print(f"stage 1b check ({fam0}): unfiltered {r_all['score']:.4f} +/- {r_all['score_std']:.4f} vs filtered "
          f"{r_f['score']:.4f} -> {'keep filtered' if ok else 'REVERT to unfiltered'}", flush=True)
    if not ok:
        pool = before_filters


def removable(imp):
    if args.rule == "mean":
        return [f for f in pool if imp[f]["mean"] <= 0 and f not in REQUIRED]
    out = []
    for f in pool:
        if f in REQUIRED:
            continue
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
    "filters": {"methods": FILTERS, "removed": filter_removed, "check": filters_check,
                "mutual_information": MI or None, "l1_nonzero_share": L1 or None,
                "bivariate_report": args.bivariate_out if BIV else None,
                "bivariate_strength": {f: b["strength"] for f, b in BIV.items()} or None},
    "candidate_pool": pool,
    "required_by_contract": REQUIRED,
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
