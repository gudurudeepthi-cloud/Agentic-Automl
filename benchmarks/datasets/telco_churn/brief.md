# telco_churn

**What it is:** IBM's Telco customer churn sample. One row = one customer (7,043 rows, 21 columns):
services, contract, billing and tenure.

**Prediction task:** `Churn` (`Yes` = left within the last month). Used to target a retention
campaign; primary metric ROC-AUC.

**Why it's in the benchmark set:** a small, real, public tabular dataset with mostly categorical
features and a moderate positive rate (~27%): the "ordinary" case.

**Known traps:** `customerID` is a row identifier. `TotalCharges` is numbers stored as text, with
blanks for brand-new customers; a naive profile calls it a unique-per-row identifier (found by
the component smoke eval). It must be profiled `numeric_as_text` and coerced. There is no time
column, so the split is random stratified.

**Access:** `scripts/download_datasets.py --id telco_churn` (public URL, no credentials).
