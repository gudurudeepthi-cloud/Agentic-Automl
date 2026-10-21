"""Template for work/features.py: the trial's engineered features (copy, rename, fill in).

aaml_common.split() calls engineer(df, contract) on the de-duplicated, time-sorted full
table BEFORE splitting, so every feature must be point-in-time: a row may use only its
own values and strictly earlier rows. Never use statistics of the whole table (global
means, target rates, future rows). The helpers below are safe by construction.

List every new column in ENGINEERED. It must equal workflow_plan.preprocessing_plan
.engineered_features (preprocess.py asserts this).
"""
import numpy as np
import pandas as pd

ENGINEERED: list[str] = [
    # e.g. "entity_prior_rows", "secs_since_prev", "amount_vs_entity_history", "new_merchant_for_entity",
    #      "hour_sin", "hour_cos", "log_amount"
]


# --------------------------------------------------------------------------- helpers (point-in-time)

def prior_count(df, entity):
    """Number of earlier rows for the same entity."""
    return df.groupby(entity, sort=False).cumcount()


def secs_since_prev(df, entity, time_col):
    """Seconds since the entity's previous row (NaN for its first row)."""
    return df.groupby(entity, sort=False)[time_col].diff().dt.total_seconds()


def ratio_to_history(df, entity, col):
    """col / mean of the entity's strictly earlier values (NaN for its first row)."""
    prior_mean = df.groupby(entity, sort=False)[col].transform(lambda s: s.shift().expanding().mean())
    return df[col] / prior_mean


def rolling_count(df, entity, time_col, window="24h"):
    """Rows for the entity in the trailing window, excluding the current row."""
    s = pd.Series(1, index=df.index)
    out = (s.groupby(df[entity], sort=False)
            .transform(lambda g: g.set_axis(df.loc[g.index, time_col]).rolling(window, closed="left").sum()
                       .fillna(0).to_numpy()))
    return out.astype(float)


def is_new_value(df, entity, col):
    """1 the first time this entity is seen with this value of `col` (novelty flag)."""
    return (df.groupby([entity, col], sort=False).cumcount() == 0).astype(int)


def cyclical(values, period):
    """(sin, cos) encoding of an hour/weekday/month column."""
    angle = 2 * np.pi * np.asarray(values, dtype=float) / period
    return np.sin(angle), np.cos(angle)


# --------------------------------------------------------------------------- the trial's features

def engineer(df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    out = df.copy()
    ent, t = contract.get("entity_id_column"), contract.get("time_column")
    # Examples (delete what doesn't apply):
    # out["entity_prior_rows"] = prior_count(out, ent)
    # out["secs_since_prev"] = secs_since_prev(out, ent, t)
    # out["amount_vs_entity_history"] = ratio_to_history(out, ent, "amount")
    # out["new_merchant_for_entity"] = is_new_value(out, ent, "merchant_id")
    # out["hour_sin"], out["hour_cos"] = cyclical(out[t].dt.hour, 24)
    # out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
    missing = [c for c in ENGINEERED if c not in out.columns]
    assert not missing, f"ENGINEERED lists columns engineer() did not create: {missing}"
    return out
