# Going back to an earlier step

Artifacts are saved in order, and each one was built from the artifacts upstream
of it. When the user wants to change an earlier decision (a different target, a
new split, another preprocessing engine), you re-run that upstream skill. The
artifacts downstream of it are then **stale**: still on disk and still
satisfying their gates, but built from the old inputs.

## How the validator tells you

`agentic-automl artifacts begin-skill <earlier-skill>` returns:

```json
{"allowed": true, "rerun": true, "downstream_artifacts_present": ["workflow_plan", "preprocessing_summary"]}
```

## What to do

1. **Say it plainly.** "Changing the split invalidates the plan, preprocessing
   and training built on it. I'll redo those after the contract is re-approved."
   Get confirmation before discarding work the user may still want.
2. If the user wants to keep the old line of work for comparison, suggest they
   exit and run `agentic-automl fork <trial> --name <slug>` first. The fork is a
   full copy, and this trial is then free to change.
3. Re-run the upstream skill and save its artifact (the version number bumps).
4. Walk forward with `agentic-automl artifacts next-step`, re-running **every**
   listed downstream skill in order. `next-step` won't flag stale artifacts on
   its own (gates are existence-based), so use the `downstream_artifacts_present`
   list from step 1 as your to-do list.
5. Don't hand-delete artifact files to "reset" a gate. Re-save through the
   validator.

## Which changes invalidate what

| Changed | Re-run from |
|---|---|
| Target, task, split, metric, data segments | `prediction-contract` (everything after it) |
| Dataset contents or schema files | `data-profiling` |
| Model family, features, engine | `workflow-planning` |
| Encoding / imputation only | the preprocessing skill |
| Hyperparameters only | `training` |
