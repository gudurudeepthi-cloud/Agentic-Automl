"""Generate a synthetic card-transaction fraud dataset for exercising the pipeline end to end.

Stdlib only and deterministic for a given --seed. Writes:

  <out-dir>/synthetic_fraud/transactions.csv        (or train.csv + test.csv with --split)
  <out-dir>/synthetic_fraud/data_dictionary.md      (pass as schema_files in the contract)
  <out-dir>/synthetic_fraud_ANSWER_KEY.md           (kept OUTSIDE the dataset dir so the agent can't peek)

The data has a known generating process plus deliberate traps the pipeline should
catch: rare positives, a post-outcome leakage column, a constant column, an ID-like
high-cardinality column, missing values, duplicate rows, repeated customers, and a
fraud-rate drift over time.

    python scripts/generate_synthetic_fraud.py --rows 250000 --fraud-rate 0.002 --seed 7 --out-dir data
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

MCC = {  # merchant category -> (share of traffic, fraud log-odds bump, typical amount)
    "grocery": (0.24, -0.6, 45),
    "restaurants": (0.16, -0.3, 35),
    "fuel": (0.10, 0.1, 50),
    "travel": (0.06, 0.6, 420),
    "electronics": (0.07, 1.1, 310),
    "fashion": (0.10, 0.2, 90),
    "digital_goods": (0.06, 1.4, 25),
    "gift_cards": (0.03, 1.8, 120),
    "pharmacy": (0.08, -0.4, 30),
    "utilities": (0.10, -0.8, 110),
}
CHANNELS = {"pos": (0.55, -0.5), "online": (0.38, 0.7), "atm": (0.07, 0.3)}
COUNTRIES = ["US"] * 85 + ["CA"] * 5 + ["GB"] * 3 + ["MX"] * 3 + ["NG", "BR", "RU", "IN"]
DEVICES = ["ios", "android", "web_desktop", "web_mobile"]

FIELDS = [
    "transaction_id", "customer_id", "event_time", "amount", "currency", "merchant_id",
    "merchant_category", "channel", "card_present", "country", "is_foreign", "hour_of_day",
    "customer_age", "account_tenure_days", "txn_count_24h", "avg_amount_30d",
    "amount_to_avg_ratio", "distance_from_home_km", "device_type", "failed_logins_7d",
    "chargeback_flag", "is_fraud",
]


def weighted(rng: random.Random, table: dict) -> str:
    return rng.choices(list(table), weights=[v[0] for v in table.values()])[0]


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def make_customers(rng: random.Random, n: int) -> list[dict]:
    customers = []
    for i in range(n):
        age = int(min(90, max(18, rng.gauss(44, 15))))
        customers.append({
            "customer_id": f"C{i:06d}",
            "age": age,
            "tenure": int(min(7000, rng.expovariate(1 / 900))) + 1,
            "spend": math.exp(rng.gauss(math.log(70), 0.6)),  # typical ticket size
            "device": rng.choice(DEVICES),
            "risky": rng.random() < 0.03,  # compromised accounts: latent, not a column
        })
    return customers


def make_row(rng: random.Random, i: int, cust: dict, start: datetime, days: int,
             signal: float) -> tuple[dict, float]:
    t = start + timedelta(seconds=rng.randrange(days * 86400))
    hour = t.hour
    mcc = weighted(rng, MCC)
    channel = weighted(rng, CHANNELS)
    if channel == "atm":
        mcc = "cash_withdrawal"
    base = MCC.get(mcc, (0, 0.5, 200))[2]
    amount = round(max(1.0, math.exp(rng.gauss(math.log(base * cust["spend"] / 70), 0.7))), 2)
    country = "US" if rng.random() < 0.9 else rng.choice(COUNTRIES)
    foreign = country != "US"
    velocity = min(40, int(rng.expovariate(1 / 2.5)) + (rng.randint(3, 15) if cust["risky"] else 0))
    avg30 = round(cust["spend"] * rng.uniform(0.8, 1.2), 2)
    ratio = round(amount / avg30, 3)
    distance = round(rng.expovariate(1 / 15), 1) if not foreign else round(rng.uniform(800, 9000), 1)
    failed = min(10, int(rng.expovariate(1.5))) + (rng.randint(1, 5) if cust["risky"] else 0)

    # Known generating process (log-odds); intercept is calibrated later.
    logit = (
        MCC.get(mcc, (0, 0.5, 0))[1]
        + CHANNELS[channel][1]
        + 1.2 * math.log1p(max(0.0, ratio - 1))
        + (0.9 if foreign else 0.0)
        + (0.8 if hour <= 4 else 0.0)
        + (1.0 if cust["tenure"] < 60 else 0.0)
        + 0.12 * velocity
        + 0.35 * failed
        + (1.5 if cust["risky"] else 0.0)
        - 0.01 * (cust["age"] - 44)
        + 0.6 * ((t - start).days / days)  # drift: fraud gets more common later
    ) * signal + rng.gauss(0, 0.6)
    row = {
        "transaction_id": f"T{i:08d}",
        "customer_id": cust["customer_id"],
        "event_time": t.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "currency": "USD",
        "merchant_id": f"M{rng.randrange(4000):05d}",
        "merchant_category": mcc,
        "channel": channel,
        "card_present": int(channel in ("pos", "atm")),
        "country": country,
        "is_foreign": int(foreign),
        "hour_of_day": hour,
        "customer_age": cust["age"],
        "account_tenure_days": cust["tenure"],
        "txn_count_24h": velocity,
        "avg_amount_30d": avg30,
        "amount_to_avg_ratio": ratio,
        "distance_from_home_km": distance,
        "device_type": cust["device"] if channel == "online" else "",
        "failed_logins_7d": failed,
    }
    return row, logit


def calibrate(logits: list[float], target_rate: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sum(sigmoid(z + mid) for z in logits) / len(logits)
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return (lo + hi) / 2


def generate(rows: int, seed: int, fraud_rate: float, days: int, signal: float = 1.0) -> list[dict]:
    rng = random.Random(seed)
    customers = make_customers(rng, max(50, rows // 12))
    start = datetime(2025, 1, 1)
    drafts = [make_row(rng, i, rng.choice(customers), start, days, signal) for i in range(rows)]
    shift = calibrate([z for _, z in drafts], fraud_rate)
    out = []
    for row, z in drafts:
        fraud = int(rng.random() < sigmoid(z + shift))
        # Leakage trap: chargebacks are filed AFTER the outcome is known.
        row["chargeback_flag"] = int(rng.random() < (0.93 if fraud else 0.0015))
        row["is_fraud"] = fraud
        # Missingness: distance unknown for some online card-not-present traffic.
        if row["channel"] == "online" and rng.random() < 0.18:
            row["distance_from_home_km"] = ""
        if rng.random() < 0.02:
            row["customer_age"] = ""
        out.append(row)
    out.sort(key=lambda r: r["event_time"])
    # Duplicate-row trap: ~0.3% exact re-sends from an upstream retry bug.
    for idx in rng.sample(range(len(out)), k=max(1, len(out) // 300)):
        out.insert(idx, dict(out[idx]))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


DICTIONARY = """# Synthetic card transactions: data dictionary

One row per card transaction. Target: `is_fraud` (1 = confirmed fraud).

| Column | Type | Description |
|---|---|---|
| transaction_id | string | Unique transaction identifier |
| customer_id | string | Cardholder identifier (customers have many transactions) |
| event_time | timestamp | Authorisation time, `YYYY-MM-DD HH:MM:SS` |
| amount | float | Transaction amount |
| currency | string | ISO currency code |
| merchant_id | string | Merchant identifier |
| merchant_category | string | Merchant category (grocery, travel, gift_cards, ...) |
| channel | string | pos / online / atm |
| card_present | int | 1 if the physical card was used |
| country | string | Merchant country |
| is_foreign | int | 1 if merchant country differs from home country |
| hour_of_day | int | Local hour of authorisation (0-23) |
| customer_age | int | Cardholder age in years (may be missing) |
| account_tenure_days | int | Days since account opening |
| txn_count_24h | int | Cardholder transactions in the prior 24 hours |
| avg_amount_30d | float | Cardholder average ticket over the prior 30 days |
| amount_to_avg_ratio | float | amount / avg_amount_30d |
| distance_from_home_km | float | Distance from billing address (may be missing) |
| device_type | string | Device for online transactions; empty otherwise |
| failed_logins_7d | int | Failed online-banking logins in the prior 7 days |
| chargeback_flag | int | 1 if a chargeback was filed for this transaction |
| is_fraud | int | **Target.** 1 if the transaction was confirmed fraudulent |
"""

ANSWER_KEY = """# ANSWER KEY: synthetic fraud dataset (do not give this to the agent)

Generated with seed={seed}, rows={rows} (+ duplicates), target fraud rate={rate:.2%},
{days} days starting 2025-01-01. Actual: {n} rows, {pos} frauds ({actual:.2%}).

What a correct pipeline run should notice:

| Trap | Where | Expected behaviour |
|---|---|---|
| Severe class imbalance (~{rate:.2%}, about 1 in {one_in}) | `is_fraud` | `pr_auc` (not accuracy) as primary metric; stratified or temporal split; class-imbalance strategy in plan |
| **Target leakage** | `chargeback_flag` | Flagged `leakage_suspect` in column_profile (single-feature AUC ~0.96); dropped in plan. If it's kept, evaluation's too-good-to-be-true check should fire |
| Constant column | `currency` (always USD) | Flagged and dropped |
| ID columns | `transaction_id`, `customer_id` | Role `id`; not features |
| High-cardinality categorical | `merchant_id` (~4000 values) | Dropped, or target/frequency-encoded out-of-fold; not one-hot |
| Repeated entities | `customer_id` (~12 txns each) | `entity_id_column`; grouped CV, or a note on leakage across folds |
| Temporal drift | fraud rate rises over time | `temporal` split on `event_time` preferred; CV-vs-test gap explained |
| Missing values | `distance_from_home_km` (~7%, online only), `customer_age` (~2%), `device_type` (empty for non-online) | Imputation in plan; device_type missingness is structural, not random |
| Duplicate rows | ~0.3% exact duplicates | Reported in `dataset_profile.duplicate_rows`; de-duplicated before splitting |

True signal (log-odds drivers, strongest first): latent compromised account (not observed;
shows up via `failed_logins_7d` and `txn_count_24h`), merchant category (gift_cards,
digital_goods, electronics), `amount_to_avg_ratio`, new accounts (`account_tenure_days` < 60),
`is_foreign`, night hours (`hour_of_day` <= 4), online channel.

Reference numbers (defaults: seed 7, 250k rows, 0.2% fraud, signal 2.0; de-duplicated,
last 20% of time as test, HistGradientBoosting with no early stopping):
  without `chargeback_flag`: ROC-AUC ~0.99, PR-AUC ~0.47
  with `chargeback_flag`:    ROC-AUC ~1.00, PR-AUC ~0.96
At 0.2% prevalence ROC-AUC is uninformative. Compare PR-AUC (base rate 0.002). PR-AUC
far above ~0.6 on this data means `chargeback_flag` leaked in. Reasonable results sit at
PR-AUC 0.35-0.55. The test split holds only ~100 frauds, so expect +/-0.05 noise.
Early stopping on a random validation slice is unstable with so few positives.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fraud-rate", type=float, default=0.002, help="Real-world card fraud is ~0.1-0.5%%")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--signal", type=float, default=2.0,
                        help="Multiplier on the true fraud drivers; higher = more separable")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", action="store_true",
                        help="Write train.csv/test.csv (last 20%% of time as test) instead of one file")
    args = parser.parse_args(argv)

    rows = generate(args.rows, args.seed, args.fraud_rate, args.days, args.signal)
    dataset_dir = args.out_dir / "synthetic_fraud"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if args.split:
        cut = int(len(rows) * 0.8)
        write_csv(dataset_dir / "train.csv", rows[:cut])
        write_csv(dataset_dir / "test.csv", rows[cut:])
    else:
        write_csv(dataset_dir / "transactions.csv", rows)
    (dataset_dir / "data_dictionary.md").write_text(DICTIONARY, encoding="utf-8")
    pos = sum(r["is_fraud"] for r in rows)
    (args.out_dir / "synthetic_fraud_ANSWER_KEY.md").write_text(ANSWER_KEY.format(
        seed=args.seed, rows=args.rows, rate=args.fraud_rate, days=args.days, one_in=round(1 / args.fraud_rate),
        n=len(rows), pos=pos, actual=pos / len(rows)), encoding="utf-8")
    print(f"Wrote {len(rows)} rows ({pos} fraud, {pos / len(rows):.2%}) to {dataset_dir}")
    print(f"Answer key: {args.out_dir / 'synthetic_fraud_ANSWER_KEY.md'} (keep it away from the agent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
