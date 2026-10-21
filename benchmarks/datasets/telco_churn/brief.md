# telco_churn

**What it is:** IBM's Telco customer churn sample. One row = one customer (7,043 rows, 21 columns):
services, contract, billing and tenure.

**Prediction task:** `Churn` (`Yes` = left within the last month). Used to target a retention
campaign; primary metric ROC-AUC.

**Why it's in the benchmark set:** a small, real, public tabular dataset with mostly categorical
features and a moderate positive rate (~27%): the "ordinary" case.

**Known traps:** `customerID` is a row identifier; `TotalCharges` is stored as text with blanks
for brand-new customers (≈ tenure × MonthlyCharges, so it adds little); no time column, so the
split is random stratified.

**Access:** `scripts/download_datasets.py --id telco_churn` (public URL, no credentials).
