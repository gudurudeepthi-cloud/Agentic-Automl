# Long-running jobs

1. **Estimate first.** Time one fold, one epoch or a 5% sample, and
   extrapolate. Tell the user the estimate. Anything over ~5 minutes needs
   their go-ahead.
2. **Run in the background.** Use the Bash tool with `run_in_background: true`
   for the script, and read progress with the background-task output tool.
   Don't use `sleep` loops, `pgrep`, `nohup ... &`, or `tail -f`.
3. **Log progress from the script.** Print one line per fold or epoch with
   elapsed time, and write partial results to `work/` so a crash doesn't lose
   everything.
4. **Checkpoint.** For loops over models or iterations (auto-research), make
   the script resumable: skip work whose output already exists.
5. **Resource sanity.** Check the row count × column count and dtypes before
   loading. Downcast numerics and use categoricals, or sample, when memory
   is tight. Report what you did.
6. When the job finishes, summarise the results before moving on.
