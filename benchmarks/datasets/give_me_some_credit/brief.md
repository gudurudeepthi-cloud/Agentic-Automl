# give_me_some_credit (draft)

**What it is:** Kaggle "Give Me Some Credit". One row = one borrower (150,000 rows in
`cs-training.csv`): revolving utilisation, age, past-due counts, debt ratio, income, open lines.

**Prediction task:** `SeriousDlqin2yrs` (1 = 90+ days past due within two years). Primary metric ROC-AUC.

**Why it's in the benchmark set:** exercises the `credit_risk` domain overlay, heavy-tailed numerics,
sentinel values (96/98 in the past-due counts) and ~20% missing income.

**Known traps:** the unnamed index column; sentinel codes in the past-due columns; `MonthlyIncome` missingness.

**Access:** manual. It needs a Kaggle account that has accepted the competition rules. Put `cs-training.csv` in
`data/benchmarks/give_me_some_credit/`, then set the entry's `status` to `active` once the
owner confirms the licence allows internal benchmarking.
