# Private trace archive operations

The live TraceStore and Git archive are separate. Successful local commit means local trace persistence; only a published archive receipt means the remote ref was verified. pending is retried without rerunning the model. A missing target returns not_configured and creates no repository or cloud resource.

Prepare an existing **private** GitHub repository and select a branch. Configure normal Git SSH/HTTPS authentication. Separately set TRACE_ARCHIVE_GITHUB_TOKEN to a read credential able to query that repository's metadata. The default visibility verifier checks GitHub's private flag before publishing; a public target or unverified visibility is rejected. The token is never written to a trace/receipt or passed as a command argument. Metadata verification cannot protect against a repository owner changing visibility later.

```
python -m membraneclaw_runtime.archives.cli publish \
  --trace-dir ./var/traces --episode EPISODE_ID \
  --remote git@github.com:YOUR_ACCOUNT/YOUR_PRIVATE_TRACE_REPO.git \
  --branch trace-data --state-dir ./var/trace-archive
```

The command seals a fixed revision under the execution/commit guards, then releases them before copying immutable bytes. The archive worker locks its control directory and builds each attempt in a fresh checkout. It stages raw Git blobs with `hash-object --no-filters` and explicit index entries, bypassing clean filters and end-of-line conversion. It adds only exact snapshot paths under traces/tenant/run/episode/snapshot_hash. No git add ., rebase or force push occurs. The live code repository is not used as the archive checkout.

An ordinary push is followed by fetch and verification that the target ref contains the commit and every expected blob byte against the sealed snapshot, without applying working-tree filters. A push timeout may have succeeded; the verifier checks that before retrying. A non-fast-forward race rebuilds the same additive snapshot on the new branch head, preserving unrelated files. Repeated snapshots do not create another commit. Retry exhaustion stores pending in the local control receipt, with the original trace intact. Receipts live outside snapshots to avoid a self-referential commit SHA loop. One task or explicit batch invokes this publisher, never one token/frame.

Restore from a separate archive clone:

```
git clone --no-checkout --branch trace-data git@github.com:YOUR_ACCOUNT/YOUR_PRIVATE_TRACE_REPO.git private-trace-checkout
python -m membraneclaw_runtime.archives.cli restore \
  --git-repository private-trace-checkout \
  --snapshot-path traces/TENANT/RUN/EPISODE/SNAPSHOT_HASH \
  --output ./recovered-traces
python -m membraneclaw_runtime.tracing.cli validate --trace-dir ./recovered-traces --episode EPISODE
```

Restore reads committed blobs directly and bypasses checkout/smudge transformations. `--git-repository` also accepts a bare clone. For a standalone snapshot directory, use `--snapshot-dir`; when it is inside a Git checkout, restore uses committed bytes at HEAD rather than edited working files. Restore requires a new isolated output directory. It checks snapshot inventory, path safety, payload hashes and commit chain before rebuilding the manifest. Never resume the same episode concurrently from two archive clones; file execution locking is single-host only.

Read-only operational gauges:

```
python -m membraneclaw_runtime.operations --trace-dir ./var/traces --archive-state ./var/trace-archive
```

The command reports episode/capture status, unknown tool outcomes, model attempts/retries, pending continuations, byte size and pending/published archive receipts. Write-failure counts, commit-latency histograms and last-push wall time require an external metrics/log collector; they are not fabricated from absent data. Watch disk capacity and archive_pending independently. A dead process should be recovered through FileTraceStore.recover after obtaining its execution guard; an active request must remain busy, not interrupted.

Keep private traces/bundles/worker directories outside public version control. Configure disk backups and retention for the actual workload. Inspect size with the gauges/du; export or archive before manual cleanup. No automatic deletion policy is enabled. Deleting current files does not remove secrets/private content from Git history. Keep raw archives private from the start; public research exports require separately reviewed derived data. Default analysis bundles omit private payloads and cannot restore execution.

Tests use real local bare Git remotes for publish/restore, failed push/retry, non-fast-forward races and a successful push with a lost response. The visibility checker is explicitly synthetic in those tests. A real private GitHub archive smoke remains unverified because no trace repository/credential was configured. Rollback stops the archive worker; local traces and existing snapshots remain readable.
