"""Open-source preprocessing: split -> (work/features.py) -> drop -> impute/one-hot (fit on train).

Template (copy to work/ with aaml_common.py, then adapt to the approved plan). Writes:
  preprocessed_data/{train,test}.parquet   encoded features + target + carried `_<col>` columns
                                           (ids, time, weight, and `_seg_<col>` raw segments)
  outputs/pipeline/preprocessor.joblib     {preprocessor, raw_features, categorical, raw_to_columns, ...}
  work/preprocessing_summary.json          draft for save-artifact

The shared matrix is model-neutral. Model-specific transforms (log/scale for linear
models) are fitted inside each model's pipeline per CV fold (aaml_common.make_model).

    <runtime python> work/preprocess.py
"""
import argparse

import joblib
import pandas as pd

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--max-onehot-levels", type=int, default=100,
                help="refuse to one-hot a categorical with more train levels than this (drop it in the plan)")
args = ap.parse_args()

c, p, cprof, dprof = A.contract(), A.plan(), A.column_profile(), A.dataset_profile()
pp = p["preprocessing_plan"]
TARGET = c["target_column"]
DROP = list(pp["drop_columns"])
F = A.load_features_module()
engineered = list(getattr(F, "ENGINEERED", []) or [])
assert sorted(pp.get("engineered_features") or []) == sorted(engineered), \
    f"plan engineered_features {pp.get('engineered_features')} != work/features.py ENGINEERED {engineered}"

train, test, info = A.split(c)
split = dprof.get("split_summary", {})
assert len(train) == split.get("train", {}).get("rows", len(train)), "train rows differ from dataset_profile"
assert len(test) == split.get("test", {}).get("rows", len(test)), "test rows differ from dataset_profile"
if c["split_strategy"] == "temporal":
    assert train[c["time_column"]].max() < test[c["time_column"]].min(), "temporal split overlaps"

roles = {col["name"]: col["role"] for col in cprof["columns"]}
non_features = {TARGET, *(k for k in (c.get("time_column"), c.get("entity_id_column"), c.get("weight_column"),
                                       c.get("split_column")) if k)}
leaky_kept = [col for col, r in roles.items() if r == "leakage_suspect" and col not in DROP]
assert not leaky_kept, f"leakage suspects not in drop_columns: {leaky_kept} (drop them or record why in the plan)"
features = [col for col in train.columns if col not in DROP and col not in non_features
            and (roles.get(col, "feature") == "feature" or col in engineered)]  # engineered cols have no role
numeric, cats = A.split_columns(train, features, cprof)
too_wide = {f: int(train[f].nunique()) for f in cats if train[f].nunique() > args.max_onehot_levels}
assert not too_wide, f"high-cardinality categoricals {too_wide}: drop them in the plan or engineer them"
with_missing = [f for f in numeric if train[f].isna().any()]
for part in (train, test):  # categoricals as strings (NaN kept) so the encoder never sees mixed types
    for f in cats:
        part[f] = part[f].map(lambda v: v if pd.isna(v) else str(v)).astype(object)

pre = A.build_encoder(numeric, cats, with_missing)
X_train = pre.fit_transform(train[features])
X_test = pre.transform(test[features])
assert list(X_train.columns) == list(X_test.columns)
assert not X_train.isna().any().any() and not X_test.isna().any().any(), "NaNs left after preprocessing"
mapping = A.raw_to_columns(pre, features, cats)

carry = [k for k in dict.fromkeys([*(col for col, r in roles.items() if r == "id"), c.get("time_column"),
                                   c.get("entity_id_column"), c.get("weight_column")]) if k and k in train]
segments = [f for f in cats if train[f].nunique() <= A.MAX_SEGMENT_LEVELS]


def frame(X, src):
    out = X.copy()
    out[TARGET] = src[TARGET].to_numpy()
    for k in carry:
        out[A.CARRY_PREFIX + k] = src[k].to_numpy()
    for k in segments:
        out[A.SEGMENT_PREFIX + k] = src[k].astype(str).to_numpy()
    return out


frame(X_train, train).to_parquet("preprocessed_data/train.parquet", index=False)
frame(X_test, test).to_parquet("preprocessed_data/test.parquet", index=False)
joblib.dump({"preprocessor": pre, "raw_features": features, "categorical": cats, "numeric": numeric,
             "raw_to_columns": mapping, "output_features": list(X_train.columns), "engineered": engineered,
             "carried": carry, "segments": segments},
            "outputs/pipeline/preprocessor.joblib")

leakage_checks = [
    f"Leakage suspects excluded: {[col for col, r in roles.items() if r == 'leakage_suspect']}",
    "Imputer and encoder fit on train only; test passes through transform",
    f"Row counts match dataset_profile (train {len(train)}, test {len(test)})",
]
if c["split_strategy"] == "temporal":
    leakage_checks.append("Every test row is later than every train row")
if engineered:
    leakage_checks.append("Engineered features are point-in-time (work/features.py runs on the time-sorted table, "
                          "using only each row and strictly earlier rows)")
transformations = [{"step": "dedupe", "description": f"Dropped {info.get('duplicate_rows', 0)} exact duplicate rows"}]
if engineered:
    transformations.append({"step": "feature_engineering", "columns": engineered,
                            "description": "work/features.py (point-in-time)"})
transformations += [
    {"step": "split", "description": f"{c['split_strategy']} split" +
     (f", train < {info['cutoff']} <= test" if "cutoff" in info else "")},
    {"step": "drop", "columns": DROP, "description": "Plan drop_columns (ids, time, constants, leakage, ...)"},
    {"step": "impute", "columns": with_missing, "description": "Median + missing indicators, fit on train"},
    {"step": "onehot", "columns": cats, "description": "Missing -> '__missing__'; unknown levels ignored"},
]
summary = {
    "engine": "open_source",
    "pipeline_script": "work/preprocess.py",
    "pipeline_artifact": "outputs/pipeline/preprocessor.joblib",
    "input_features": features,
    "output_feature_count": int(X_train.shape[1]),
    "transformations": transformations,
    "dropped_columns": DROP,
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "leakage_checks": leakage_checks,
    "notes": "Model-specific scaling/log transforms are applied inside each model pipeline, per CV fold.",
}
A.write_json("work/preprocessing_summary.json", summary)
print(f"{len(features)} raw features ({len(engineered)} engineered, {len(cats)} categorical) -> "
      f"{X_train.shape[1]} columns; train {len(train)}, test {len(test)}; imputed {with_missing}; "
      f"carried {carry}; segments {segments}")
