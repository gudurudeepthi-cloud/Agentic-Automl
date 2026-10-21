"""Profile the dataset as the contract defines it -> work/dataset_profile.json, work/column_profile.json.

Template (copy to work/ with aaml_common.py, then adapt). The split is the one in
aaml_common.split(), the same one preprocessing uses, so the counts match later steps.
Leakage probes use the train split only, and categoricals are target-encoded out of fold.

    <runtime python> work/profile_dataset.py [--probe-sample 200000]

Everything written here is computed. Any domain explanation (e.g. "missing only for
online transactions") goes in a column's `notes` after you have checked it.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--probe-sample", type=int, default=200_000, help="max train rows used by the leakage probe")
ap.add_argument("--leak-threshold", type=float, default=0.95)
args = ap.parse_args()

c = A.contract()
T = A.task(c)
TARGET, TIME = c["target_column"], c.get("time_column")
train, test, info = A.split(c, engineer=False)
df = pd.concat([train, test], ignore_index=True)
raw_n = info["raw_rows"]
y_train = A.encode_target(train[TARGET], c)


def split_stats(part):
    y = A.encode_target(part[TARGET], c)
    out = {"rows": int(len(part))}
    if T == "binary":
        out.update(positives=int(y.sum()), positive_rate=round(float(y.mean()), 5))
    elif T == "multiclass":
        out["class_rates"] = {str(k): round(float(v), 5) for k, v in part[TARGET].value_counts(normalize=True).items()}
    else:
        out.update(mean=round(float(np.mean(y)), 5), std=round(float(np.std(y)), 5))
    return out


def oof_encode(x: pd.Series, y: np.ndarray, folds=5, smoothing=20) -> pd.Series:
    """Out-of-fold smoothed target mean. In-sample rates memorise labels of high-cardinality columns."""
    from sklearn.model_selection import KFold

    yser = pd.Series(y, index=x.index, dtype=float)
    prior = yser.mean()
    out = pd.Series(prior, index=x.index, dtype=float)
    for fit_idx, score_idx in KFold(folds, shuffle=True, random_state=0).split(x):
        stats = yser.iloc[fit_idx].groupby(x.iloc[fit_idx]).agg(["sum", "count"])
        enc = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        out.iloc[score_idx] = x.iloc[score_idx].map(enc).fillna(prior).to_numpy()
    return out


probe_idx = (np.random.default_rng(A.SEED).choice(len(train), args.probe_sample, replace=False)
             if len(train) > args.probe_sample else np.arange(len(train)))
probe, y_probe = train.iloc[probe_idx], y_train[probe_idx]


def probe_strength(col, is_cat):
    """Single-feature signal on train: orientation-free AUC (classification; best one-vs-rest
    for multiclass) or |Spearman| (regression). None when undefined."""
    from sklearn.metrics import roc_auc_score

    x = probe[col]
    if is_cat:
        x = x.astype(str).fillna("<missing>")
        targets = [y_probe] if T != "multiclass" else [(y_probe == k).astype(int) for k in np.unique(y_probe)]
        vals = []
        for yt in targets:
            enc = oof_encode(x, yt)
            if enc.nunique() > 1 and len(np.unique(yt)) > 1:
                a = roc_auc_score(yt, enc) if T != "regression" else abs(pd.Series(enc).corr(pd.Series(yt), "spearman"))
                vals.append(max(a, 1 - a) if T != "regression" else a)
        return round(float(max(vals)), 4) if vals else None
    x = pd.to_numeric(x, errors="coerce")
    if x.nunique() < 2:
        return None
    if T == "regression":
        return round(float(abs(x.corr(pd.Series(y_probe, index=x.index), method="spearman"))), 4)
    x = x.fillna(x.min() - 1)
    targets = [y_probe] if T == "binary" else [(y_probe == k).astype(int) for k in np.unique(y_probe)]
    aucs = [roc_auc_score(yt, x) for yt in targets if len(np.unique(yt)) > 1]
    return round(float(max(max(a, 1 - a) for a in aucs)), 4) if aucs else None


flags, columns = [], []
ent, wcol, scol = c.get("entity_id_column"), c.get("weight_column"), c.get("split_column")
for col in df.columns:
    s = df[col]
    n_unique = int(s.nunique(dropna=True))
    missing = round(float(s.isna().mean()), 4)
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    stats, notes = {}, None
    if col == TARGET:
        role, sem = "target", T
    elif col == TIME:
        role, sem = "time", "datetime"
        stats = {"min": str(s.min()), "max": str(s.max()), **({"test_cutoff": info["cutoff"]} if "cutoff" in info else {})}
    elif col == ent:
        role, sem = "id", "entity_identifier"
    elif col == wcol:
        role, sem = "weight", "numeric"
    elif col == scol:
        role, sem = "split", "split_label"
    elif n_unique <= 1:
        role, sem = "drop", "constant"
        flags.append({"severity": "warning", "columns": [col], "message": f"{col} is constant ({s.dropna().iloc[0]!r})"
                      if s.notna().any() else f"{col} is entirely missing"})
    elif not numeric and A.numeric_text_share(s) > 0.95:
        # Numbers stored as text (e.g. "29.85" with " " for blanks): a numeric feature, not an id/categorical.
        role, sem = "feature", "numeric_as_text"
        conv = A.coerce_numeric_text(s)
        q = conv.quantile([0, 0.5, 1])
        stats = {"min": round(float(q.iloc[0]), 4), "median": round(float(q.iloc[1]), 4),
                 "max": round(float(q.iloc[2]), 4), "unparseable_or_blank": int(conv.isna().sum() - s.isna().sum())}
        flags.append({"severity": "info", "columns": [col],
                      "message": f"{col} is numeric stored as text ({stats['unparseable_or_blank']} blank/unparseable); "
                                 "aaml_common.split() coerces it to numbers (blanks -> NaN)"})
        stats["univariate_signal_train"] = probe_strength(col, is_cat=False)
    elif n_unique >= 0.95 * len(df) and (not numeric or pd.api.types.is_integer_dtype(s)):
        role, sem = "id", "row_identifier"
        flags.append({"severity": "warning", "columns": [col],
                      "message": f"{col} is unique per row ({n_unique} values): an identifier, not a feature"})
    else:
        role = "feature"
        if not numeric:
            parsed = pd.to_datetime(s.dropna().astype(str).head(1000), errors="coerce", format="mixed")
            if len(parsed) and parsed.notna().mean() > 0.9 and not s.astype(str).str.fullmatch(r"\d+").all():
                sem = "date_string"
                flags.append({"severity": "info", "columns": [col],
                              "message": f"{col} parses as a date; derive features from it or drop it"})
            else:
                sem = "high_cardinality_categorical" if n_unique > 100 else "categorical"
            stats = {"top": {str(k): int(v) for k, v in s.value_counts().head(5).items()}}
        elif set(s.dropna().unique()) <= {0, 1}:
            sem = "boolean"
        else:
            sem = "numeric"
            q = s.quantile([0, 0.01, 0.5, 0.99, 1])
            stats = dict(zip(["min", "p01", "median", "p99", "max"], [round(float(v), 4) for v in q]))
        if sem != "date_string":
            strength = probe_strength(col, is_cat=not numeric)
            stats["univariate_signal_train"] = strength
            if strength is not None and strength > args.leak_threshold:
                role = "leakage_suspect"
                flags.append({"severity": "critical", "columns": [col],
                              "message": f"{col} alone nearly determines the target (train signal {strength}); "
                                         "check whether it is known before the outcome"})
        if sem == "high_cardinality_categorical":
            flags.append({"severity": "warning", "columns": [col],
                          "message": f"{col} has {n_unique} levels; do not one-hot encode (drop, or use it as a "
                                     "novelty/frequency feature)"})
    if missing > 0.5 and role == "feature":
        flags.append({"severity": "warning", "columns": [col], "message": f"{col} is {missing:.0%} missing"})
    if missing > 0 and role == "feature" and len(df) > 0:
        # Is missingness concentrated in one level of a categorical? (evidence for a 'structural' note)
        for other in df.columns:
            if other == col or df[other].dtype != object or df[other].nunique() > 20:
                continue
            rate = s.isna().groupby(df[other]).mean()
            if (rate > 0.99).any() and (rate < 0.01).any():
                stats["missing_pattern"] = {other: {str(k): round(float(v), 3) for k, v in rate.items()}}
                break
    entry = {"name": col, "dtype": str(s.dtype), "semantic_type": sem, "role": role,
             "missing_fraction": missing, "n_unique": n_unique}
    if stats:
        entry["stats"] = stats
    if notes:
        entry["notes"] = notes
    columns.append(entry)

# --- dataset-level flags ----------------------------------------------------------
tr, te = split_stats(train), split_stats(test)
if T == "binary":
    if tr["positive_rate"] < 0.01:
        flags.append({"severity": "warning", "columns": [TARGET],
                      "message": f"Severe class imbalance: {tr['positive_rate']:.3%} positives in train"})
    if tr["positive_rate"] > 0 and not 0.67 <= te["positive_rate"] / tr["positive_rate"] <= 1.5:
        flags.append({"severity": "warning", "columns": [TARGET],
                      "message": f"Test positive rate {te['positive_rate']:.4%} vs train {tr['positive_rate']:.4%}"})
    if te.get("positives", 0) < 100:
        flags.append({"severity": "warning", "columns": [TARGET],
                      "message": f"Only {te['positives']} test positives: test metrics will be noisy"})
elif T == "multiclass":
    rare = {k: v for k, v in tr["class_rates"].items() if v < 0.01}
    if rare:
        flags.append({"severity": "warning", "columns": [TARGET], "message": f"Rare classes (<1% of train): {rare}"})
if info.get("duplicate_rows"):
    flags.append({"severity": "warning", "columns": [],
                  "message": f"{info['duplicate_rows']} exact duplicate rows; dropped before splitting"})

segment_summary = {}
if TIME:
    by_month = df.groupby(df[TIME].dt.to_period("M"))[TARGET]
    if T == "binary":
        rates = df.assign(_y=A.encode_target(df[TARGET], c)).groupby(df[TIME].dt.to_period("M"))["_y"].mean()
        segment_summary["positive_rate_by_month"] = {str(k): round(float(v), 5) for k, v in rates.items()}
        if len(rates) > 2 and rates.iloc[0] > 0 and not 0.8 <= rates.iloc[-1] / rates.iloc[0] <= 1.25:
            flags.append({"severity": "info", "columns": [TARGET, TIME],
                          "message": f"Positive rate drifts over time ({rates.iloc[0]:.3%} -> {rates.iloc[-1]:.3%})"})
    segment_summary["rows_by_month"] = {str(k): int(v) for k, v in by_month.size().items()}
if ent:
    per = df.groupby(ent).size()
    shared = len(set(train[ent]) & set(test[ent]))
    segment_summary["rows_per_entity"] = {"mean": round(float(per.mean()), 2), "max": int(per.max())}
    if per.max() > 1:
        flags.append({"severity": "info", "columns": [ent],
                      "message": f"~{per.mean():.1f} rows per {ent}; {shared} entities appear in both train and test"})

now = datetime.now(timezone.utc).isoformat(timespec="seconds")
y_all = A.encode_target(df[TARGET], c)
target_summary = ({"counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
                   **({"positive_rate": round(float(y_all.mean()), 5)} if T == "binary" else {})}
                  if T != "regression" else
                  {"mean": float(np.mean(y_all)), "std": float(np.std(y_all)),
                   "quantiles": {str(q): float(np.quantile(y_all, q)) for q in (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1)}})
dataset_profile = {
    "dataset_path": c["dataset_path"],
    "n_rows": int(raw_n),
    "n_columns": int(df.shape[1]),
    "target_column": TARGET,
    "target_summary": target_summary,
    "split_summary": {**{k: v for k, v in info.items() if k not in ("raw_rows", "duplicate_rows")},
                      "train": tr, "test": te},
    "missing_value_summary": {col: round(float(v), 4) for col, v in df.isna().mean().items() if v > 0},
    "duplicate_rows": int(info.get("duplicate_rows") or 0),
    "data_quality_flags": flags,
    "profiling_script": "work/profile_dataset.py",
    "segment_summary": segment_summary,
    "generated_at": now,
    "notes": (f"n_rows counts raw rows ({raw_n}); target and split counts are after removing "
              f"{info.get('duplicate_rows', 0)} exact duplicates. Leakage probe on "
              f"{len(probe_idx)} train rows (seed {A.SEED})."),
}
A.write_json("work/dataset_profile.json", dataset_profile)
A.write_json("work/column_profile.json", {"dataset_path": c["dataset_path"], "columns": columns, "generated_at": now})

print(f"rows {raw_n} (dupes {info.get('duplicate_rows', 0)}); train {tr}; test {te}")
for col in columns:
    sig = col.get("stats", {}).get("univariate_signal_train")
    print(f"  {col['name']:<26} {col['role']:<16} {col['semantic_type']:<30} miss={col['missing_fraction']:<7} "
          f"uniq={col['n_unique']:<8} signal={sig}")
for f in flags:
    print(f"[{f['severity']}] {f['message']}")
