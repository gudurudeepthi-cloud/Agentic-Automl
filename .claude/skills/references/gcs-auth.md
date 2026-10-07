# GCS access

1. **Credentials are inherited** from the runtime (workload identity / gcloud
   application-default credentials). Run `gsutil` and `gcsfs` bare. Don't
   pass key files and don't run `gcloud auth` yourself.
2. **Verify access with Python**, not `gsutil`:
   ```python
   # work/check_gcs.py
   import gcsfs
   fs = gcsfs.GCSFileSystem()
   print(fs.ls("gs://bucket/prefix")[:5])
   ```
   Then read with `pandas.read_parquet("gs://...")` / `read_csv`.
3. **Only on an actual credentials error** (`401`, `403`, `DefaultCredentialsError`):
   ask the user for a service-account key path, and set
   `GOOGLE_APPLICATION_CREDENTIALS` for that one command or script. Never copy
   the key into the trial.
4. **Mutating commands** (`gsutil cp/mv/rm/rsync`, writes via gcsfs) need user
   approval unless the trial is in trust mode (`work/trust.md`). Write
   scratch output to the configured scratch bucket
   (`agentic-automl config show` → `scratch_bucket`), never next to source data.
5. For large reads, list and size the files first (`fs.du`) and sample before
   loading everything into memory.
