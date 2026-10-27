# Trace storage v1 (PR 1)

This library provides the storage foundation for the multi-provider tracing plan.
It does **not** trace existing Qwen requests yet. Provider adapters, gateway,
CLI/HTTP integration, streaming, evaluation replay, and Git archive follow in
separate PRs. No model service, cloud database, or trace archive is configured here.

## Install and test

Python 3.11+ on a POSIX host:

```sh
python -m pip install --no-deps -e .
python -m unittest discover -s tests/tracing -v
python -m membraneclaw_runtime.tracing.example
```

The package has no runtime dependencies and imports no `agent` modules. Other
projects, including `experiments/auto-evaluate`, can install this distribution
rather than changing `sys.path`. Its dependency wiring belongs to the integration
PR. Existing `requirements.txt` continues to install the Qwen application.

Only local persistent filesystems with working atomic rename, directory fsync,
and POSIX advisory locks are eligible for durable use. NFS/shared multi-host writes
are unsupported; `multi_host=True` is rejected. The Linux CI workflow exercises
Python 3.11–3.13. A green Linux run is required before declaring Linux validated;
local macOS test success alone does not establish production durability. SIGKILL
checks process-crash behavior, not actual device power-loss guarantees.

## Execution and storage contract

`TraceStore` in `store.py` is the public protocol. `FileTraceStore` is durable under
the filesystem assumptions above; `MemoryTraceStore` is deliberately non-durable.
Both run the same semantic contract tests. Episode references include strictly
validated tenant/run/episode IDs; applications must bind tenant to authenticated
identity. Namespaces are not an authorization system.

1. Create an episode, then acquire its `execution_guard` before reading/restoring
   execution state. Hold it across model network waits, tool execution, and state
   commits. Release only at terminal status or a persisted client-tool handoff.
2. Commit request/start events **before** executing an external action. Commit raw
   responses/results **before** consuming or parsing them. Store failures propagate;
   callers must stop new external actions and report the failure outside tracing
   if the store cannot record it.
3. Construct `Event`, `Payload`, and `commit_id` once. Reuse identical objects/data
   on storage retry. A same-ID/same-content retry returns its original receipt
   **before** checking `expected_revision`. Different content conflicts; a new ID
   with a stale revision conflicts. Repeated event IDs are rejected.
4. `TraceRecorder` retains an unresolved commit, including generated IDs, on error.
   `retry_pending()` retries storage only; new appends are refused until resolved.
   After a process restart, inspect persisted events and recover; do not guess or
   repeat a model/tool action with an unknown outcome.

The execution lock and commit lock are separate nonblocking OS locks. Busy access
raises `Busy`, so it cannot block the async event loop. When nested, acquire
execution then commit, never the reverse. Low-level commits do not acquire the
execution lock: the runtime owns the whole execution segment. Do not share a
recorder between threads. Async runtimes should offload synchronous disk I/O to
a worker thread and explicitly carry the episode context.

`recover(episode)` only marks an unlocked `running` episode `interrupted` with
partial capture and unknown external outcome. It never calls a provider/tool.
Active execution returns `Busy`; `awaiting_client_tool` stays pending. Startup
scans can page `list_episodes({'status': 'running'})` and recover each, skipping
`Busy`. Interrupted remote work can still finish and incur costs.

## Authoritative files

```text
<root>/<tenant>/<run_id>/<episode_id>/
  genesis.json              immutable creation metadata and hash
  commits/000001.json        immutable authoritative commit chain
  payloads/<sha256>.bin      exact bytes; JSON checkpoints use the same object store
  manifest.json             derived summary, safe to rebuild
  .execution.lock           OS lock, not a PID/liveness file
  .commit.lock              short storage transaction lock
```

All payload bytes use `.bin`, including canonical JSON checkpoints. This keeps
references independent of media type and deduplicates identical bytes. Files are
mode 0600 and new directories 0700; choose a private root and verify existing
parent permissions. Runtime traces default to `var/traces/` in the example and
are ignored by Git. The library accepts a configured root, never an untrusted
request path. Symlink components under that root are rejected. The root itself
is a trusted operator setting, resolved at construction. Hostile concurrent
filesystem modifications by the same OS account are outside the threat model.

A commit writes/fsyncs payload temporary files, atomically renames them, and
fsyncs their directory; it then does the same for the commit file. The commit-file
rename followed by directory fsync is the durable commit point. Only afterward
is the derived manifest replaced. Readers scan/validate the commit chain and
ignore stale/missing manifests. An uncertain retry also re-fsyncs committed data
before returning its original receipt. Six-digit file numbering supports up to
999,999 revisions per episode; start a new episode before that bound.

Crash leftovers may include hidden temporary files (ignored by readers) and
unreferenced content objects. `orphan_payloads()` identifies the latter; it does
not delete them. Do cleanup only offline under execution/commit exclusion after
retention review. A missing committed file/reference, gap, or invalid hash raises
`Corruption`; do not skip it or overwrite evidence. `rebuild_manifest()` restores
the derived summary from a validated chain. `events.jsonl`, summary exports, and
Git receipts are not authorities and will be added by later integration PRs.

## Serialization and event schema

`canonical_json_v1`: UTF-8, sorted string keys, `ensure_ascii=False`, compact
separators, `allow_nan=False`, array order unchanged, no Unicode normalization.
Only JSON types are accepted; raw bytes use `Payload(bytes)` and are hashed
without reserialization. This is a project protocol, not a general cross-language
canonical JSON standard. The shared tests contain byte-level golden data and
synthetic model/tool/checkpoint fixtures; no real provider results are included.

An event envelope has `schema_version=1`, `event_id`, `episode_id`, `sequence`,
`type`, `wall_time`, `parent_event_id`, `request_id`, `attempt_id`, `model_call_id`,
`tool_call_id`, `payload_ref`, `state_before_ref`, `state_after_ref`, and `data`.
Non-applicable optional fields are null. Sequences start at 1 across commits;
parents must precede children. Event types are extensible, while terminal event
names must match their terminal state. Payload references are opaque to business
code; create them with `Payload.ref(episode)` and resolve via `read_payload(ref)`.
Payload/state references cannot cross episodes.

Commit content hashes cover every field except their own `hash`, including
schema/serialization version, commit ID, request hash, previous hash, revision,
events, reference inventory, checkpoint, status and completeness. Genesis anchors
the first commit and hashes immutable metadata. The request hash excludes the
expected revision so uncertain retries remain idempotent; it includes stable
pre-sequence event envelopes, submitted payload references, checkpoint and
explicit state/completeness arguments. Unknown schema/serialization versions
fail closed. No downgrade or migration by silently rewriting old files.

Execution status: `running`, `awaiting_client_tool`, `completed`, `failed`,
`cancelled`, `interrupted`. Terminal states require a matching final event and
reject later commits. Capture (`complete/partial/unknown/redacted`), replay
(`ready/unsupported/missing_payload`), and external outcome (`known/unknown`) are
separate. Defaults are unknown/unsupported/unknown. The storage layer validates
these labels; the future gateway/runtime must supply evidence for completeness.
A failed task may have complete capture. No usage counters are fabricated here.

## Credentials and private data

Do not pass authentication headers/URL credentials into tracing. `CapturePolicy`
rejects conventional authentication field names in JSON and registered secret
values anywhere in metadata/events/payload bytes **before persistence**. Future
transports must register the exact credential values they use; the policy does
not read environment variables or guess all possible secrets. It deliberately
rejects rather than silently modifying byte evidence. Credentials in novel
formats or unknown strings require transport-side exclusion; this is not an
automatic universal redactor. The policy may reject legitimate similarly named
data; explicitly prepare an auditable redacted view before capture in that case.

Provider thought/continuation signatures are business data and are preserved.
Prompts, tools, checkpoints and signatures remain sensitive even when credentials
are absent. Store raw traces in a private location. Never push them to the public
code repository. No automatic Git commits, network uploads, or archive repository
creation occur in this PR.

## Snapshot and restore

`seal(store, episode)` takes execution then commit locks and captures one validated
revision as immutable in-memory bytes. Two seals at the same revision produce the
same hash. Further source commits cannot alter an existing snapshot. Only genesis,
commits, and referenced payloads are included; orphans, lock files and stale
manifests are excluded. The hash covers a canonical sorted relative-path/hash/size
inventory, excluding timestamps, host paths, receipts and its own hash.

`restore(snapshot, new_directory)` checks the inventory, rejects unsafe/duplicate
paths, copies exact bytes into a **new** isolated directory, verifies the copy and
all references, then rebuilds the manifest. Restore does not resume external
execution. The snapshot API currently holds the episode in memory, so budget
memory for its complete referenced payload volume. A streaming archive/bundle
format and remote backup transport belong to PR 8. Until configured, data remains
local and is subject to host loss. Back up only sealed data into access-controlled
storage and validate restoration before relying on it.

## Rollback and next PR

No existing agent, HTTP, CLI or evaluation behavior changes. Revert this library
and CI/docs to roll back; retain existing traces and their version information.
Do not delete or downgrade traces as rollback. PR 2 adds the provider contracts,
controlled raw transport, mandatory gateway recording, and Gemini adapter. Real
provider and Qwen smoke tests are not part of this standalone storage PR.

Current scale boundary: commits validate the existing chain and referenced payloads
before extending it. Repeated writes therefore become expensive for long episodes
and repeated large histories. This intentionally simple correctness baseline needs
measurement before streaming workloads; caching/index changes must preserve
corruption detection and restart validation. It is not a high-volume database.
