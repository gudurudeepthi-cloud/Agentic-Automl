# Domain overlay: credit_risk

Read alongside the platform skill whenever `ml_domain: credit_risk`. Where it
conflicts with the generic skill, this overlay wins (a user override still
wins over both).

## prediction-contract
- The target is usually default within a performance window (e.g. 90+ DPD in
  12 months). Ask for the exact definition and record it in `notes`.
- Prefer a **temporal** (out-of-time) split. If the data has an application
  or observation date, `random` needs an explicit justification.
- Report `roc_auc` and Gini (= 2·AUC − 1) and `ks`, plus `pr_auc` when the
  default rate is under 5%.
- Watch for reject-inference bias: the data covers approved applicants only. Note it.

## workflow-planning
- Consider monotonic constraints on features with a known risk direction
  (utilisation ↑ → risk ↑).
- Exclude protected attributes and close proxies unless the user explicitly
  scopes a fairness analysis. List what was excluded.
- Favour interpretable candidates (e.g. regularised logistic, constrained GBM)
  alongside the strongest model.

## evaluation
- Add trust checks: `population_stability` (PSI of score, train vs test; flag
  > 0.25) and `rank_ordering` (default rate decreasing across score deciles).
- Include a score-decile table: count, defaults and default rate.

## model-development-documentation
- Add sections on target definition and performance window, sample exclusions,
  stability (PSI), and the limitations of reject inference.
