# synthetic_fraud

**What it is:** generated card transactions (`scripts/generate_synthetic_fraud.py`, stdlib,
deterministic per seed). One row = one transaction, about six months of activity, ~12 rows per customer.

**Prediction task:** `is_fraud` (1 = fraudulent). The score ranks an analyst review queue that can
absorb about 1% of daily volume, so the primary metric is PR-AUC.

**Why it's in the benchmark set:** it has a known generating process plus deliberate traps, so
a scored eval can check the agent made the right calls rather than only that it finished.

**Known traps:** `chargeback_flag` is post-outcome leakage; `currency` is constant; `merchant_id`
has 4,000 levels and no signal (in-sample target encoding makes it look leaky); exact duplicate
rows; fraud rate drifts upward over time; customers repeat (use a temporal split on `event_time`).

**Access:** `scripts/download_datasets.py --id synthetic_fraud` runs the generator into
`data/benchmarks/synthetic_fraud/`. The answer key is written outside the dataset directory;
never put it where a trial can read it.
