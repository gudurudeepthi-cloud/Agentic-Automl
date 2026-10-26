"""Vintage analysis and selection -> work/vintage_analysis.json, outputs/report/vintage_analysis.png.

Template (copy to work/ with aaml_common.py). Run during workflow-planning, after profiling.
A vintage is the period (month by default) of each row's date. Every decision uses the TRAIN
split only. Test vintages are listed for context and never used to choose anything.

Methods
  volume        rows and events per vintage; flags vintages below --min-rows / --min-events
  event rate    fraud/default rate per vintage with a Wilson 95% interval, and a p-chart:
                pooled rate +/- 3 sigma for that vintage's size. Outside = unusual vintage
                (an attack, a policy change, a data fault); review it with the user, never drop
                it silently. Regression uses the target mean (+/- 3 standard errors).
  trend/break   Spearman trend of the rate over time, plus the single best split point
                (likelihood-ratio test, binary; Welch t, regression). The p-value is optimistic
                because the split point was searched.
  maturity      --maturity-days N: outcomes arrive late (e.g. chargebacks within 90 days).
                Train rows newer than (latest train date - N) are immature: their labels can
                still change. The last vintages' rates are also checked against the p-chart.
  drift (PSI)   population stability of each feature per vintage against the most recent
                --reference-periods train vintages (deciles / top categories).
                PSI > 0.25 major, 0.10-0.25 moderate
  window        backtest of training-window length: rolling origins inside train. For each of the
                last --origins blocks of --holdout-periods vintages, fit on the previous k
                vintages and score that block. Done for k in --windows (plus all). A shorter
                window is proposed only if it beats `all` by more than the std of its paired gain.

Output: per-vintage table, flags, and `proposed_vintage_selection` in the shape of the plan's
`vintage_selection`. Show it to the user, then put the approved version in the plan.
aaml_common.split() applies it to train in every later step.

    <runtime python> work/vintage_analysis.py [--period M] [--maturity-days 90] [--drop col ...]
"""
import argparse
import math

import numpy as np
import pandas as pd

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--column", default=None, help="date column (default: contract time_column)")
ap.add_argument("--period", choices=["D", "W", "M", "Q"], default="M")
ap.add_argument("--min-rows", type=int, default=None, help="default: max(200, 10%% of the median vintage)")
ap.add_argument("--min-events", type=int, default=10, help="binary: minimum positives per vintage")
ap.add_argument("--maturity-days", type=int, default=None)
ap.add_argument("--reference-periods", type=int, default=2, help="PSI reference = latest N train vintages")
ap.add_argument("--psi-features", type=int, default=15, help="max features checked for drift")
ap.add_argument("--holdout-periods", type=int, default=1, help="backtest: vintages per validation block")
ap.add_argument("--origins", type=int, default=3, help="backtest: number of rolling validation blocks")
ap.add_argument("--windows", default="auto",
                help="backtest: comma list of training-window lengths in periods (plus all); "
                     "auto = 3/4, 1/2 and 1/4 of the history before the first validation block")
ap.add_argument("--family", default="logistic_regression")
ap.add_argument("--drop", nargs="*", default=[], help="columns excluded from the backtest/drift features")
ap.add_argument("--max-onehot-levels", type=int, default=100)
ap.add_argument("--no-plot", action="store_true")
ap.add_argument("--out", default="work/vintage_analysis.json")
ap.add_argument("--plot", default="outputs/report/vintage_analysis.png")
args = ap.parse_args()

c, cprof = A.contract(), A.column_profile()
T = A.task(c)
assert T in ("binary", "regression"), \
    "vintage rates support binary and regression targets; for multiclass, adapt it to one class vs rest"
COL = args.column or c.get("time_column")
assert COL, "no date column: pass --column (the contract has no time_column)"
train, test, info = A.split(c, vintage=False)  # the full train split: this script decides the vintages
assert COL in train, f"{COL} not in the data"
train = train.copy()
train[COL] = pd.to_datetime(train[COL])
test = test.copy()
if COL in test:
    test[COL] = pd.to_datetime(test[COL])
y = A.encode_target(train[c["target_column"]], c).astype(float)
train["__v"] = A.vintages(train[COL], args.period)
order = sorted(train["__v"].unique())
assert len(order) >= 3, f"only {len(order)} {args.period} vintages in train; use a finer --period"


def label(p):
    return str(p.start_time.date()) if args.period == "W" else str(p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(mid - half, 6), round(mid + half, 6))


# --- volume, rate, p-chart --------------------------------------------------------------
g = pd.DataFrame({"v": train["__v"], "y": y}).groupby("v")["y"]
size, ysum, ymean, ystd = g.size(), g.sum(), g.mean(), g.std(ddof=1).fillna(0)
min_rows = args.min_rows or max(200, int(0.1 * size.median()))
pooled = float(y.mean())
sd_all = float(y.std(ddof=1)) if len(y) > 1 else 0.0
latest = train[COL].max()
rows = []
for p in order:
    n = int(size[p])
    r = {"vintage": label(p), "rows": n, "rate" if T != "regression" else "target_mean": round(float(ymean[p]), 6)}
    flags = []
    if n < min_rows:
        flags.append("low_volume")
    if T == "binary":
        k = int(ysum[p])
        lo, hi = wilson(k, n)
        sig = math.sqrt(pooled * (1 - pooled) / n)
        r.update(events=k, rate_ci95=[lo, hi], control=[round(max(0.0, pooled - 3 * sig), 6), round(pooled + 3 * sig, 6)])
        if k < args.min_events:
            flags.append("low_events")
    elif T == "regression":
        se = sd_all / math.sqrt(n)
        r.update(control=[round(pooled - 3 * se, 6), round(pooled + 3 * se, 6)])
    if "control" in r:
        val = r.get("rate", r.get("target_mean"))
        if val > r["control"][1]:
            flags.append("above_control")
        elif val < r["control"][0]:
            flags.append("below_control")
    if args.maturity_days:
        cutoff = latest - pd.Timedelta(days=args.maturity_days)
        if p.end_time > cutoff:
            flags.append("immature" if p.start_time > cutoff else "partly_immature")
    r["flags"] = flags
    rows.append(r)

# --- trend and single break -------------------------------------------------------------
vals = np.array([ymean[p] for p in order], dtype=float)
trend = pd.Series(vals).corr(pd.Series(range(len(vals))), method="spearman")
brk = None
if T in ("binary", "regression") and len(order) >= 4:
    best = None
    for k in range(2, len(order) - 1):
        a, b = order[:k], order[k:]
        na, nb = float(size[a].sum()), float(size[b].sum())
        ma, mb = float(ysum[a].sum() / na), float(ysum[b].sum() / nb)
        if T == "binary":
            def ll(s, n):
                p_ = s / n
                return 0.0 if p_ in (0.0, 1.0) else s * math.log(p_) + (n - s) * math.log(1 - p_)
            stat = 2 * (ll(ysum[a].sum(), na) + ll(ysum[b].sum(), nb) - ll(ysum.sum(), na + nb))
        else:
            va = float((ystd[a] ** 2 * (size[a] - 1)).sum() / max(na - 1, 1))
            vb = float((ystd[b] ** 2 * (size[b] - 1)).sum() / max(nb - 1, 1))
            stat = (ma - mb) ** 2 / (va / na + vb / nb) if va + vb > 0 else 0.0  # Welch t^2
        if best is None or stat > best[0]:
            best = (stat, k, ma, mb)
    from scipy.stats import chi2

    stat, k, ma, mb = best
    brk = {"between": [label(order[k - 1]), label(order[k])], "before": round(ma, 6), "after": round(mb, 6),
           "statistic": round(float(stat), 3), "p_value_optimistic": float(chi2.sf(stat, 1))}

# --- drift (PSI) --------------------------------------------------------------------------
roles = {col["name"]: col["role"] for col in cprof.get("columns", [])}
non_feat = {c["target_column"], COL, *(c.get(k) for k in ("time_column", "entity_id_column", "weight_column",
                                                          "split_column") if c.get(k)), *args.drop}
excluded = {e["column"] for e in c.get("excluded_columns") or []}
features = [f for f in train.columns if f not in non_feat | excluded and f != "__v"
            and roles.get(f, "feature") == "feature"]
numeric, cats = A.split_columns(train, features, cprof)
cats = [f for f in cats if train[f].nunique() <= args.max_onehot_levels]
features = numeric + cats
ref_mask = train["__v"].isin(order[-args.reference_periods:])
signal = {}
for f in features:  # rank drift candidates by univariate signal from the profile when available
    s = next((col.get("stats", {}).get("univariate_signal_train") for col in cprof.get("columns", [])
              if col["name"] == f), None)
    signal[f] = s or 0
drift_feats = sorted(features, key=lambda f: -signal[f])[:args.psi_features]


def psi(ref, cur, is_cat):
    if is_cat:
        top = ref.astype(str).value_counts().index[:20]
        r = ref.astype(str).where(ref.astype(str).isin(top), "<other>").value_counts(normalize=True)
        q = cur.astype(str).where(cur.astype(str).isin(top), "<other>").value_counts(normalize=True)
    else:
        edges = np.unique(np.nanquantile(ref.astype(float), np.linspace(0, 1, 11)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        r = pd.cut(ref.astype(float), edges).value_counts(normalize=True, dropna=False)
        q = pd.cut(cur.astype(float), edges).value_counts(normalize=True, dropna=False)
    keys = r.index.union(q.index)
    r, q = r.reindex(keys, fill_value=0) + 1e-4, q.reindex(keys, fill_value=0) + 1e-4
    return float(((q - r) * np.log(q / r)).sum())


for r, p in zip(rows, order):
    cur = train["__v"] == p
    scores = {f: round(psi(train.loc[ref_mask, f], train.loc[cur, f], f in cats), 4) for f in drift_feats}
    worst = max(scores, key=scores.get) if scores else None
    r["psi_max"] = scores.get(worst) if worst else None
    r["psi_max_feature"] = worst
    if worst and scores[worst] > 0.25:
        r["flags"].append("major_drift")
    elif worst and scores[worst] > 0.10:
        r["flags"].append("moderate_drift")

# --- training-window backtest -------------------------------------------------------------
h, n_orig = args.holdout_periods, args.origins
while n_orig > 1 and len(order) - n_orig * h < 2:
    n_orig -= 1
history = len(order) - n_orig * h  # vintages available before the first validation block
if args.windows == "auto":
    windows = sorted({w for w in (round(history * f) for f in (0.75, 0.5, 0.25)) if 2 <= w < history}, reverse=True)
else:
    windows = sorted({int(w) for w in args.windows.split(",") if w.strip()}, reverse=True)
backtest = None
if len(order) - n_orig * h >= 2:
    pre = A.build_encoder(numeric, cats, [f for f in numeric if train[f].isna().any()])
    for f in cats:
        train[f] = train[f].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    params = next((m.get("search_space") for m in A.read_json("work/workflow_plan.json", default={})
                   .get("modeling_plan", {}).get("candidate_models", []) if m["name"] == args.family), None)
    params = A.param_grid(params)[0] if params else {}
    per = {w: [] for w in ["all", *windows]}
    origins = []
    for o in range(n_orig, 0, -1):
        val_v = order[len(order) - o * h: len(order) - (o - 1) * h]
        before = order[:len(order) - o * h]
        val = train["__v"].isin(val_v)
        if A.task(c) == "binary" and len(np.unique(y[val.to_numpy()])) < 2:
            continue
        origins.append([label(v) for v in val_v])
        for w in per:
            fit_v = before if w == "all" else before[-w:]
            if w != "all" and w >= len(before):
                per[w].append(None)  # same as `all`
                continue
            fit = train["__v"].isin(fit_v)
            if args.maturity_days:  # labels of the last N days before the block would not be known yet
                fit &= train[COL] <= train.loc[val, COL].min() - pd.Timedelta(days=args.maturity_days)
            if fit.sum() < 50 or (A.task(c) != "regression" and len(np.unique(y[fit.to_numpy()])) < 2):
                per[w].append(None)
                continue
            Xf = pre.fit_transform(train.loc[fit, features])
            Xv = pre.transform(train.loc[val, features])
            m = A.make_model(args.family, params, T, Xf, {col for col in Xf.columns if col not in numeric})
            m.fit(Xf, y[fit.to_numpy()] if T == "regression" else y[fit.to_numpy()].astype(int))
            per[w].append(A.score(c["primary_metric"], y[val.to_numpy()] if T == "regression"
                                  else y[val.to_numpy()].astype(int), A.predict_scores(m, Xv, T), T))
    sign = 1 if A.greater_is_better(c["primary_metric"]) else -1
    res, skipped = {}, {}
    for w, scores in per.items():
        if not scores or any(s is None for s in scores):
            skipped[str(w)] = ("not enough history before some validation block, or under 50 rows / one class "
                               "left after the maturity gap")
            continue
        entry = {"periods": w, "mean": round(float(np.mean(scores)), 5), "std": round(float(np.std(scores)), 5),
                 "per_origin": [round(float(s), 5) for s in scores]}
        if w != "all" and "all" in res:
            gain = sign * (np.array(scores) - np.array(per["all"]))
            entry.update(gain_vs_all=round(float(gain.mean()), 5), gain_std=round(float(gain.std()), 5))
        res[str(w)] = entry
    winners = [e for k, e in res.items() if k != "all" and e["gain_vs_all"] > e["gain_std"]]
    chosen = max(winners, key=lambda e: e["gain_vs_all"]) if winners else res.get("all")
    backtest = {"family": args.family, "params": params, "metric": c["primary_metric"], "origins": origins,
                "holdout_periods": h, "windows": res, "skipped_windows": skipped,
                "chosen": chosen["periods"] if chosen else "all",
                "rule": "shorter window only if its mean paired gain over `all` exceeds the gain's std"}

# --- proposal ------------------------------------------------------------------------------
# Only thin *edge* vintages (partial first/last periods) are proposed for exclusion; a low event
# count alone is noise, not a reason to drop a period.
edge_low = [r["vintage"] for r in rows[:1] + rows[-1:] if "low_volume" in r["flags"]]
start = None
if backtest and backtest["chosen"] != "all":
    start = label(order[-int(backtest["chosen"])])
review = [r["vintage"] for r in rows if {"above_control", "below_control", "major_drift"} & set(r["flags"])]
proposal = {"column": COL, "period": args.period, "train_start": start, "train_end": None,
            "exclude_periods": [v for v in edge_low if v != start],
            "maturity_days": args.maturity_days, "analysis_script": "work/vintage_analysis.py",
            "rationale": "; ".join(filter(None, [
                f"window: last {backtest['chosen']} periods beat all by more than noise" if start else
                "window: all train vintages (no shorter window beat it by more than noise)",
                f"edge vintages with too little data: {edge_low}" if edge_low else None,
                f"maturity: drop the last {args.maturity_days} days of train" if args.maturity_days else None,
                f"review with the user (not excluded automatically): {review}" if review else None]))}
nothing = not (start or proposal["exclude_periods"] or args.maturity_days)

test_rows = []
if COL in test and len(test):
    tv = A.vintages(test[COL], args.period)
    ty = A.encode_target(test[c["target_column"]], c).astype(float)
    for p, grp in pd.DataFrame({"v": tv, "y": ty}).groupby("v")["y"]:
        test_rows.append({"vintage": label(p), "rows": int(len(grp)), "mean": round(float(grp.mean()), 6)})

A.write_json(args.out, {
    "column": COL, "period": args.period, "task": T, "pooled_rate_or_mean": round(pooled, 6),
    "min_rows": min_rows, "train_vintages": rows,
    "trend_spearman": None if pd.isna(trend) else round(float(trend), 3), "single_break": brk,
    "drift": {"reference": [label(p) for p in order[-args.reference_periods:]], "features": drift_feats},
    "window_backtest": backtest,
    "test_vintages_for_context_only": test_rows,
    "proposed_vintage_selection": None if nothing else proposal,
    "proposal_note": ("all train vintages kept: nothing to exclude" if nothing else
                      "proposal only: confirm with the user, then copy into workflow_plan.vintage_selection"),
})

# --- report ----------------------------------------------------------------------------------
key = "rate" if T != "regression" else "target_mean"
print(f"{COL} by {args.period}: {len(order)} train vintages, pooled {key} {pooled:.5f}, min_rows {min_rows}")
print(f"{'vintage':<12}{'rows':>8}{'events':>8}{key:>12}  {'95% CI':<22}{'psi_max':>9}  flags")
for r in rows:
    ci = f"[{r['rate_ci95'][0]:.4f}, {r['rate_ci95'][1]:.4f}]" if "rate_ci95" in r else ""
    psi_s = f"{r['psi_max']:.3f}" if r.get("psi_max") is not None else "-"
    print(f"{r['vintage']:<12}{r['rows']:>8}{r.get('events', ''):>8}{r[key]:>12.5f}  {ci:<22}{psi_s:>9}  "
          f"{','.join(r['flags']) or '-'}")
if test_rows:
    print("test (context only): " + ", ".join(f"{t['vintage']} n={t['rows']} {t['mean']:.4f}" for t in test_rows))
print(f"trend (Spearman over vintages): {trend:+.2f}" if not pd.isna(trend) else "trend: n/a")
if brk:
    print(f"largest shift: {brk['before']:.5f} -> {brk['after']:.5f} between {brk['between'][0]} and "
          f"{brk['between'][1]} (optimistic p={brk['p_value_optimistic']:.3g})")
if backtest:
    print(f"window backtest ({args.family}, {c['primary_metric']}, validation blocks {backtest['origins']}):")
    for k, e in backtest["windows"].items():
        g_ = f"  gain {e['gain_vs_all']:+.4f} +/- {e['gain_std']:.4f}" if "gain_vs_all" in e else ""
        print(f"  {k:>4} periods  {e['mean']:.4f} +/- {e['std']:.4f}{g_}")
    for k, why in backtest["skipped_windows"].items():
        print(f"  {k:>4} periods  skipped: {why}")
    print(f"  -> {backtest['chosen']}")
print(f"\nproposal: {proposal if not nothing else 'keep all train vintages'}")
print(f"wrote {args.out}")

if not args.no_plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(7, 0.45 * len(rows) + 3), 6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 2]})
    a1.bar(x, [r["rows"] for r in rows], color=["#c0392b" if "low_volume" in r["flags"] else "#7f8fa6" for r in rows])
    a1.set_ylabel("rows")
    vals = [r[key] for r in rows]
    if "control" in rows[0]:
        a2.fill_between(x, [r["control"][0] for r in rows], [r["control"][1] for r in rows], color="#dfe4ea",
                        label="p-chart +/- 3 sigma")
    if "rate_ci95" in rows[0]:
        a2.errorbar(x, vals, yerr=[[v - r["rate_ci95"][0] for v, r in zip(vals, rows)],
                                   [r["rate_ci95"][1] - v for v, r in zip(vals, rows)]],
                    fmt="o-", color="#2b45a8", capsize=2, label=f"{key} (Wilson 95%)")
    else:
        a2.plot(x, vals, "o-", color="#2b45a8", label=key)
    a2.axhline(pooled, color="#57606f", ls="--", lw=1, label="pooled")
    for i, r in enumerate(rows):
        if {"above_control", "below_control"} & set(r["flags"]):
            a2.plot(i, vals[i], "s", color="#c0392b", ms=9, mfc="none")
        if "immature" in r["flags"] or "partly_immature" in r["flags"]:
            a2.axvspan(i - 0.5, i + 0.5, color="#f6e58d", alpha=0.4, lw=0)
    if proposal["train_start"]:
        a2.axvline(next(i for i, r in enumerate(rows) if r["vintage"] == proposal["train_start"]) - 0.5,
                   color="#c0392b", lw=1.5, label="proposed train start")
    a2.set_xticks(x, [r["vintage"] for r in rows], rotation=60, ha="right", fontsize=8)
    a2.set_ylabel(key)
    a2.legend(fontsize=8, loc="best")
    a1.set_title(f"Train vintages by {args.period} ({COL}); squares = outside control, yellow = immature")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"wrote {args.plot}")
