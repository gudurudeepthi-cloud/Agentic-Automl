"""Shared helpers for the step templates in templates/ (copy this file into work/ with them).

One definition of: loading the contract's data, the split, the CV folds, metrics, the
shared encoder and the model families. Profiling, the family screen, preprocessing,
feature selection, training and evaluation all import it, so they agree on rows,
folds and scores. Scripts run from the trial root: `<runtime python> work/<script>.py`.

Domain-specific feature engineering lives in an optional `work/features.py` with
`ENGINEERED: list[str]` and `engineer(df, contract) -> df` (see features_template.py).
"""
from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CARRY_PREFIX = "_"  # carried columns in the parquet files (ids, time, weight, raw segments)
SEGMENT_PREFIX = "_seg_"
MAX_SEGMENT_LEVELS = 20


# --------------------------------------------------------------------------- files

def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"{path} not found (run the earlier step first)")
    return json.loads(p.read_text(encoding="utf-8-sig"))  # Windows PowerShell 5.1 writes a UTF-8 BOM


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def contract():
    return read_json("prediction_contract.json")


def plan():
    return read_json("state/workflow_plan.json")


def column_profile():
    return read_json("column_profile.json", default={"columns": []})


def dataset_profile():
    return read_json("dataset_profile.json", default={})


def load_features_module():
    """work/features.py if present, else None (no engineered features)."""
    path = Path("work/features.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- data

def _read_one(path: str, fmt: str) -> pd.DataFrame:
    fmt = (fmt or Path(path).suffix.lstrip(".")).lower()
    p = Path(path)
    if not path.startswith("gs://") and p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lstrip(".").lower() == fmt)
        if not files:
            raise FileNotFoundError(f"no *.{fmt} files under {path}")
        return pd.concat([_read_one(str(f), fmt) for f in files], ignore_index=True)
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt in ("parquet", "pq"):
        return pd.read_parquet(path)
    if fmt in ("jsonl", "ndjson"):
        return pd.read_json(path, lines=True)
    if fmt == "json":
        return pd.read_json(path)
    raise ValueError(f"unsupported format {fmt!r} for {path}; extend _read_one")


def load_raw(c) -> dict[str, pd.DataFrame]:
    """{'full': df} or {'train': df, 'test': df} from contract.data_segments (role 'other' is ignored)."""
    out = {}
    for seg in c["data_segments"]:
        if seg["role"] == "other":
            continue
        df = _read_one(seg["path"], seg.get("format"))
        out[seg["role"]] = pd.concat([out[seg["role"]], df], ignore_index=True) if seg["role"] in out else df
    return out


def coerce_numeric_text(s: pd.Series) -> pd.Series:
    """'29.85' -> 29.85; blank / unparseable -> NaN."""
    return pd.to_numeric(s.astype("string").str.strip().replace("", pd.NA), errors="coerce")


def numeric_text_share(s: pd.Series) -> float:
    """Share of non-blank text values that parse as numbers (0 for numeric dtypes or no values)."""
    if pd.api.types.is_numeric_dtype(s):
        return 0.0
    text = s.dropna().astype(str).str.strip()
    text = text[text != ""]
    return float(pd.to_numeric(text, errors="coerce").notna().mean()) if len(text) else 0.0


def _coerce_profiled_numeric_text(df: pd.DataFrame) -> pd.DataFrame:
    """Columns the profile marked numeric_as_text become numbers in every later step."""
    cols = [col["name"] for col in column_profile().get("columns", [])
            if col.get("semantic_type") == "numeric_as_text" and col["name"] in df]
    if cols:
        df = df.copy()
        for col in cols:
            df[col] = coerce_numeric_text(df[col])
    return df


def _prepare(df: pd.DataFrame, c, dedupe: bool) -> tuple[pd.DataFrame, int]:
    n_dupes = int(df.duplicated().sum())
    if dedupe:
        df = df.drop_duplicates()
    df = _coerce_profiled_numeric_text(df)
    if c.get("time_column"):
        df = df.copy()
        df[c["time_column"]] = pd.to_datetime(df[c["time_column"]])
        sort = [c["time_column"]] + [k for k in (c.get("entity_id_column"),) if k]
        df = df.sort_values(sort, kind="mergesort")
    return df.reset_index(drop=True), n_dupes


def split(c, *, engineer: bool = True, dedupe: bool = True):
    """Apply the contract split exactly as every step must.

    Returns (train, test, info). Exact duplicate rows are dropped first. When
    work/features.py exists and `engineer` is True, its engineer() runs on the full
    time-sorted table before splitting (so features must be point-in-time).
    """
    raw = load_raw(c)
    feats = load_features_module() if engineer else None
    strategy, target = c["split_strategy"], c["target_column"]
    info = {"strategy": strategy, "after_dedup": dedupe}

    if "full" in raw:
        df, n_dupes = _prepare(raw["full"], c, dedupe)
        info.update(raw_rows=int(len(raw["full"])), duplicate_rows=n_dupes)
        if feats:
            df = feats.engineer(df, c)
        if strategy == "temporal":
            t = c["time_column"]
            cut = df[t].quantile(1 - c["split_ratio"])
            train, test = df[df[t] < cut], df[df[t] >= cut]
            info["cutoff"] = str(cut)
        elif strategy == "predefined_column":
            col = c["split_column"]
            train = df[df[col] == c["split_train_value"]]
            test = df[df[col] == c["split_test_value"]]
        elif strategy in ("random", "random_stratified"):
            from sklearn.model_selection import train_test_split
            strat = df[target] if strategy == "random_stratified" else None
            train, test = train_test_split(df, test_size=c["split_ratio"], random_state=SEED, stratify=strat)
            train, test = train.sort_index(), test.sort_index()
        else:
            raise ValueError(f"split_strategy {strategy!r} needs train/test segments, not a full segment")
    else:
        train, d1 = _prepare(raw["train"], c, dedupe)
        test, d2 = _prepare(raw["test"], c, dedupe)
        info.update(raw_rows=int(len(raw["train"]) + len(raw["test"])), duplicate_rows=d1 + d2)
        if feats:  # history features must see train before test
            both = pd.concat([train.assign(__part="train"), test.assign(__part="test")], ignore_index=True)
            both, _ = _prepare(both, c, dedupe=False)
            both = feats.engineer(both, c)
            train = both[both["__part"] == "train"].drop(columns="__part")
            test = both[both["__part"] == "test"].drop(columns="__part")
    return train.reset_index(drop=True), test.reset_index(drop=True), info


def task(c) -> str:
    kind = c["target_kind"]
    return {"binary": "binary", "multiclass": "multiclass", "continuous": "regression"}.get(kind, "regression")


def encode_target(y: pd.Series, c) -> np.ndarray:
    """Binary -> 0/1 with target_labels[1] as the positive class; multiclass -> label index."""
    labels = c.get("target_labels")
    t = task(c)
    if t == "binary":
        pos = labels[1] if labels else 1
        return (y == pos).astype(int).to_numpy()
    if t == "multiclass":
        order = labels or sorted(y.dropna().unique().tolist())
        return y.map({v: i for i, v in enumerate(order)}).to_numpy()
    return y.astype(float).to_numpy()


# --------------------------------------------------------------------------- metrics

_TOPK = re.compile(r"^(precision|recall)_at_top_(\d+(?:_\d+)?)pct$")
LOWER_IS_BETTER = {"log_loss", "brier", "rmse", "mae", "mape", "mse"}


def greater_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def score(metric: str, y, s, t: str) -> float | None:
    """`s` = positive-class score (binary), class-probability matrix (multiclass) or prediction."""
    from sklearn import metrics as M

    y = np.asarray(y)
    m = _TOPK.match(metric)
    if m:
        frac = float(m.group(2).replace("_", ".")) / 100
        k = max(1, int(len(s) * frac))
        hits = y[np.argsort(-np.asarray(s))[:k]].sum()
        return float(hits / k if m.group(1) == "precision" else hits / max(1, y.sum()))
    if t == "binary" and len(np.unique(y)) < 2 and metric in ("pr_auc", "roc_auc"):
        return None
    if metric == "pr_auc":
        return float(M.average_precision_score(y, s))
    if metric == "roc_auc":
        return float(M.roc_auc_score(y, s, multi_class="ovr") if t == "multiclass" else M.roc_auc_score(y, s))
    if metric == "log_loss":
        return float(M.log_loss(y, s, labels=list(range(s.shape[1])) if t == "multiclass" else [0, 1]))
    if metric == "brier":
        return float(M.brier_score_loss(y, s))
    if metric in ("accuracy", "f1", "f1_macro", "balanced_accuracy"):
        pred = np.argmax(s, axis=1) if t == "multiclass" else (np.asarray(s) >= 0.5).astype(int)
        if metric == "accuracy":
            return float(M.accuracy_score(y, pred))
        if metric == "balanced_accuracy":
            return float(M.balanced_accuracy_score(y, pred))
        return float(M.f1_score(y, pred, average="macro" if t == "multiclass" or metric == "f1_macro" else "binary"))
    if metric == "rmse":
        return float(np.sqrt(M.mean_squared_error(y, s)))
    if metric == "mse":
        return float(M.mean_squared_error(y, s))
    if metric == "mae":
        return float(M.mean_absolute_error(y, s))
    if metric == "r2":
        return float(M.r2_score(y, s))
    if metric == "mape":
        return float(M.mean_absolute_percentage_error(y, s))
    raise ValueError(f"metric {metric!r} not implemented in aaml_common.score; add it")


def safe_score(metric, y, s, t):
    """score() for report metrics: None (not a crash) when a metric isn't implemented or doesn't apply."""
    try:
        return score(metric, y, s, t)
    except ValueError:
        return None


def better(a: float, b: float, metric: str) -> bool:
    return a > b if greater_is_better(metric) else a < b


# --------------------------------------------------------------------------- CV

def cv_folds(train: pd.DataFrame, y: np.ndarray, c, n_splits: int = 5, groups=None):
    """Folds consistent with the contract split. Returns (folds, description).

    temporal        -> expanding-window TimeSeriesSplit (train must be time-sorted)
    entity repeats  -> StratifiedGroupKFold / GroupKFold on entity_id_column
    classification  -> StratifiedKFold;  regression -> KFold
    """
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

    t = task(c)
    if c["split_strategy"] == "temporal":
        folds = list(TimeSeriesSplit(n_splits=n_splits).split(train))
        desc = f"{n_splits}-fold expanding-window TimeSeriesSplit on time-sorted train"
        if groups is not None:
            desc += " (entities can appear on both sides of a fold boundary, as they do across train/test)"
        return folds, desc
    if groups is not None and pd.Series(groups).duplicated().any():
        if t == "regression":
            return list(GroupKFold(n_splits=n_splits).split(train, y, groups)), \
                f"{n_splits}-fold GroupKFold on {c['entity_id_column']}"
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(cv.split(train, y, groups)), f"{n_splits}-fold StratifiedGroupKFold on {c['entity_id_column']}"
    if t == "regression":
        return list(KFold(n_splits, shuffle=True, random_state=SEED).split(train)), f"{n_splits}-fold KFold"
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(train, y)), f"{n_splits}-fold StratifiedKFold"


# --------------------------------------------------------------------------- encoder

def build_encoder(numeric: list[str], categorical: list[str], with_missing: list[str]):
    """Model-neutral shared matrix: median + indicator for numerics with NaN, passthrough
    for the rest, one-hot categoricals (missing -> '__missing__', unknown ignored)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    plain = [n for n in numeric if n not in with_missing]
    pre = ColumnTransformer(
        [
            ("num_impute", SimpleImputer(strategy="median", add_indicator=True), with_missing),
            ("num", "passthrough", plain),
            ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="__missing__")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def raw_to_columns(pre, raw: list[str], categorical: list[str]) -> dict[str, list[str]]:
    """Map each raw feature to its encoded columns (one-hot levels, missing indicator)."""
    out_cols = list(pre.get_feature_names_out())
    mapping = {f: [] for f in raw}
    for col in out_cols:
        if col.startswith("missingindicator_"):
            mapping[col[len("missingindicator_"):]].append(col)
        elif col in mapping:
            mapping[col].append(col)
        else:
            owner = max((f for f in categorical if col.startswith(f + "_")), key=len)
            mapping[owner].append(col)
    assert all(mapping.values()) and sum(map(len, mapping.values())) == len(out_cols)
    return mapping


def split_columns(df: pd.DataFrame, features: list[str], cprof=None) -> tuple[list[str], list[str]]:
    """(numeric, categorical) using column_profile semantic types, falling back to dtype."""
    sem = {col["name"]: col["semantic_type"] for col in (cprof or {}).get("columns", [])}
    cats = [f for f in features if "categorical" in sem.get(f, "") or not pd.api.types.is_numeric_dtype(df[f])]
    return [f for f in features if f not in cats], cats


# --------------------------------------------------------------------------- models

# Simpler first: ties on the paired CV gap go to the earlier family.
COMPLEXITY = ["logistic_regression", "linear_regression", "ridge", "random_forest", "extra_trees",
              "hist_gradient_boosting", "lightgbm", "xgboost", "catboost"]


def model_library(name: str) -> str:
    return {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}.get(name, "scikit-learn")


def make_model(name: str, params: dict, t: str, X: pd.DataFrame, onehot_cols: set[str] | None = None):
    """Estimator for one family. Linear models get log1p on skewed non-negative numerics plus
    scaling, fitted inside the pipeline (so per CV fold)."""
    params = {k: v for k, v in (params or {}).items() if not isinstance(v, (list, dict, str)) or k in _STR_PARAMS}
    clf = t != "regression"
    if name in ("logistic_regression", "linear_regression", "ridge"):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        onehot_cols = onehot_cols or set()
        cand = [c for c in X.columns if c not in onehot_cols and not c.startswith("missingindicator_")]
        skewed = [c for c in cand if X[c].min() >= 0 and abs(X[c].skew()) > 2]
        rest = [c for c in X.columns if c not in skewed]
        pre = ColumnTransformer([
            ("log", Pipeline([("log1p", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), skewed),
            ("sc", StandardScaler(), rest),
        ])
        est = (LogisticRegression(max_iter=5000, random_state=SEED, **params) if clf
               else Ridge(random_state=SEED, **params))
        return Pipeline([("pre", pre), ("m", est)])
    if name in ("random_forest", "extra_trees"):
        from sklearn import ensemble as E
        cls = {("random_forest", True): E.RandomForestClassifier, ("random_forest", False): E.RandomForestRegressor,
               ("extra_trees", True): E.ExtraTreesClassifier, ("extra_trees", False): E.ExtraTreesRegressor}[(name, clf)]
        return cls(**{"n_estimators": 300, "min_samples_leaf": 20, "n_jobs": -1, "random_state": SEED, **params})
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(**{"random_state": SEED, **params})
    if name == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
        return cls(**{"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "random_state": SEED,
                      "n_jobs": -1, "verbose": -1, **params})
    if name == "xgboost":
        import xgboost as xgb
        cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
        return cls(**{"n_estimators": 500, "learning_rate": 0.05, "random_state": SEED, "n_jobs": -1, **params})
    raise ValueError(f"model family {name!r} not in make_model; add it")


_STR_PARAMS = {"penalty", "solver", "objective", "boosting_type", "max_features", "criterion", "class_weight"}


def param_grid(search_space: dict | None, cap: int = 12) -> list[dict]:
    """Expand list-valued entries of a plan search_space into a small grid (fixed values kept)."""
    import itertools

    space = search_space or {}
    fixed = {k: v for k, v in space.items() if not isinstance(v, list)}
    lists = {k: v for k, v in space.items() if isinstance(v, list)}
    combos = [dict(zip(lists, vals)) for vals in itertools.product(*lists.values())] or [{}]
    if len(combos) > cap:
        rng = np.random.default_rng(SEED)
        combos = [combos[i] for i in sorted(rng.choice(len(combos), cap, replace=False))]
    return [{**fixed, **c} for c in combos]


def predict_scores(model, X, t: str):
    if t == "binary":
        return model.predict_proba(X)[:, 1]
    if t == "multiclass":
        return model.predict_proba(X)
    return model.predict(X)


# --------------------------------------------------------------------------- modeling data

class Data:
    """Encoded train split + raw-feature -> encoded-column groups + CV folds. Test is never loaded.

    Data.load() reads preprocessing outputs (training steps); Data.from_raw() encodes the
    raw train split in memory (the planning-stage family screen, before preprocessing exists).
    """

    def __init__(self, c, train, raw, cats, groups, engineered=(), weight=None, entity=None, n_splits=5):
        self.contract = c
        self.task = task(c)
        self.metric = c["primary_metric"]
        self.target = c["target_column"]
        self.train, self.raw, self.cats, self.groups = train, raw, cats, groups
        self.y = encode_target(train[self.target], c)
        self.engineered = set(engineered)
        self.onehot = {col for f in cats for col in groups[f]}
        self.weight = weight
        self.folds, self.cv_desc = cv_folds(train, self.y, c, n_splits, entity)
        self.plan = read_json("state/workflow_plan.json", default={"modeling_plan": {"candidate_models": []}})
        self.candidates = {m["name"]: m for m in self.plan["modeling_plan"]["candidate_models"]}

    @classmethod
    def load(cls, n_splits: int = 5):
        import joblib

        c = contract()
        train = pd.read_parquet("preprocessed_data/train.parquet")
        meta = joblib.load("outputs/pipeline/preprocessor.joblib")
        wcol, ent = c.get("weight_column"), c.get("entity_id_column")
        return cls(c, train, meta["raw_features"], meta["categorical"], meta["raw_to_columns"],
                   meta.get("engineered", []),
                   weight=train[CARRY_PREFIX + wcol].to_numpy() if wcol else None,
                   entity=train[CARRY_PREFIX + ent].to_numpy() if ent and CARRY_PREFIX + ent in train else None,
                   n_splits=n_splits)

    @classmethod
    def from_raw(cls, c, train: pd.DataFrame, features: list[str], cprof=None, engineered=(), n_splits=5):
        numeric, cats = split_columns(train, features, cprof)
        train = train.copy()
        for f in cats:
            train[f] = train[f].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
        pre = build_encoder(numeric, cats, [f for f in numeric if train[f].isna().any()])
        X = pre.fit_transform(train[features])
        X[c["target_column"]] = train[c["target_column"]].to_numpy()
        wcol, ent = c.get("weight_column"), c.get("entity_id_column")
        return cls(c, X, features, cats, raw_to_columns(pre, features, cats), engineered,
                   weight=train[wcol].to_numpy() if wcol else None,
                   entity=train[ent].to_numpy() if ent else None, n_splits=n_splits)

    def cols_for(self, feats):
        return [col for f in feats for col in self.groups[f]]

    def make(self, family, params, cols):
        return make_model(family, params, self.task, self.train[cols], self.onehot)

    def fit(self, model, X, y, w):
        if w is None:
            return model.fit(X, y)
        if hasattr(model, "steps"):  # sklearn Pipeline: route the weight to the final step
            return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
        return model.fit(X, y, sample_weight=w)

    def cv(self, family, feats, params, *, permute=False, rng=None, repeats=3, folds=None, metrics=None,
           keep_predictions=False):
        """Mean/std of the primary metric (plus `metrics`) over the folds; optionally per-fold
        permutation importance per raw feature (all its encoded columns shuffled together)."""
        cols = self.cols_for(feats)
        X, y = self.train[cols], self.y
        metrics = [m for m in (self.contract.get("report_metrics", []) if metrics is None else metrics)
                   if m != self.metric]
        per = {m: [] for m in [self.metric, *metrics]}
        perm = {f: [] for f in feats}
        preds = []
        t0 = time.time()
        for fit_idx, val_idx in folds or self.folds:
            w = None if self.weight is None else self.weight[fit_idx]
            model = self.fit(self.make(family, params, cols), X.iloc[fit_idx], y[fit_idx], w)
            Xv, yv = X.iloc[val_idx], y[val_idx]
            s = predict_scores(model, Xv, self.task)
            for m in per:
                per[m].append(score(m, yv, s, self.task) if m == self.metric else safe_score(m, yv, s, self.task))
            if keep_predictions:
                preds.append(pd.DataFrame({"row": val_idx, "y": yv, **_score_cols(s)}))
            if permute:
                base = per[self.metric][-1]
                sign = 1 if greater_is_better(self.metric) else -1
                for f in feats:
                    drops = []
                    for _ in range(repeats):
                        Xp = Xv.copy()
                        g = self.groups[f]
                        Xp[g] = Xp[g].to_numpy()[rng.permutation(len(Xp))]
                        drops.append(sign * (base - score(self.metric, yv, predict_scores(model, Xp, self.task),
                                                          self.task)))
                    perm[f].append(float(np.mean(drops)))
        vals = [v for v in per[self.metric] if v is not None]
        res = {"score": float(np.mean(vals)), "score_std": float(np.std(vals)),
               "fold_scores": [None if v is None else round(float(v), 5) for v in per[self.metric]],
               "other_metrics": {m: float(np.mean([v for v in vs if v is not None])) for m, vs in per.items()
                                 if m != self.metric and any(v is not None for v in vs)},
               "seconds": round(time.time() - t0, 1)}
        if permute:
            res["permutation_importance"] = {f: {"mean": float(np.mean(v)), "per_fold": [round(x, 5) for x in v]}
                                             for f, v in perm.items()}
        if keep_predictions:
            res["predictions"] = pd.concat(preds, ignore_index=True)
        return res


def _score_cols(s):
    s = np.asarray(s)
    if s.ndim == 2:
        return {f"p{i}": s[:, i] for i in range(s.shape[1])}
    return {"score": s}


def paired_gap(a: list, b: list, metric: str) -> tuple[float, float, list]:
    """Per-fold (a - b), oriented so positive means `a` is better."""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    if not greater_is_better(metric):
        d = -d
    return float(np.nanmean(d)), float(np.nanstd(d)), np.round(d, 4).tolist()


def show(label, r, metric):
    print(f"   {label:<44} {metric} {r['score']:.4f} +/- {r['score_std']:.4f}  ({r['seconds']}s)", flush=True)
