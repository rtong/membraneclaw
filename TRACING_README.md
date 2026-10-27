# Trace Recording User Guide

This feature records the **model requests, model responses, tool arguments, tool results, and application state** for each task. Each task has an `episode_id`. Use that ID to inspect the original data, validate its integrity, export it, replay it offline, and archive it to a private Git repository.

“Complete” means complete within the application's observable capture boundary. Hidden model state is outside that boundary. Data the API did not return, or a remote result the application never received, cannot be reconstructed. The legacy Qwen path remains available but does not automatically provide these recording guarantees.

## 1. Install

Until the full PR stack is merged, use the branch containing the complete implementation:

```bash
git clone --branch codex/tracing-user-guide https://github.com/regulusv/membraneclaw.git
cd membraneclaw
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[tools,http]'
umask 077
```

Python 3.11 or later is required. Durability and cross-process locking CI runs on Linux. The file backend requires local persistent storage on one host; shared writes from multiple hosts are unsupported. `umask 077` makes subsequent private exports accessible only to the current operator by default.

## 2. Try recording without an API key

```bash
python -m membraneclaw_runtime.tracing.example
```

This creates a **synthetic example** without contacting a model or executing tools. Its output includes:

```text
episode=local/<run_id>/<episode_id> status=completed
```

Copy the full episode key into `EPISODE`:

```bash
EPISODE='paste-the-full-local/run_id/episode_id-key-here'
python -m membraneclaw_runtime.tracing.cli validate --trace-dir ./var/traces --episode "$EPISODE"
python -m membraneclaw_runtime.tracing.cli inspect --trace-dir ./var/traces --episode "$EPISODE"
```

`validate` should exit with code 0 and report the revision, event count, payload count, status, and completeness. `inspect` displays events, state, and complete payloads. Payloads are represented with explicit base64 encoding rather than truncated previews.

This example verifies storage and inspection. It does not test a live API and is not an execution-replay fixture.

## 3. Record a live model task

Set the appropriate credential environment variable, then explicitly select a provider and model. The current code accepts these reviewed compatibility profiles; this table is not a recommendation of the latest models:

| Provider | Model | Credential environment variable |
|---|---|---|
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| `openai` | `gpt-4.1-2025-04-14` or `o4-mini-2025-04-16` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |

For example, set your `GEMINI_API_KEY` in the current shell, then run:

```bash
membraneclaw \
  --provider gemini --model gemini-2.5-flash \
  --trace-dir ./var/traces --no-mcp \
  -p 'Use calculator to compute 27 * 19, then report the result.'
```

**Pass the prompt with `-p` or `--prompt`**, not as a positional argument. `--no-mcp` disables external MCP discovery while retaining local tools such as calculator. A correct final answer alone does not prove a tool ran: check `tool.started` and `tool.completed` in the trace.

For OpenAI or Claude, change the provider, model, and corresponding credential. Do not switch providers inside an existing episode. API keys come from the environment and are excluded from request bodies and configuration snapshots.

The CLI prints the episode ID at startup, then the answer, `status`, and `reason` at completion. Successful execution exits with code 0; failure exits with a nonzero code. Copy this episode ID and use the inspection commands in section 2. A short episode ID is accepted if it is unique within the selected trace directory.

## 4. Find and interpret a trace

The default location is `./var/traces`, relative to the current working directory. Each task's authoritative data includes:

```text
var/traces/<tenant>/<run_id>/<episode_id>/
  genesis.json
  manifest.json
  commits/000001.json ...
  payloads/<sha256>.bin ...
```

The immutable commits form a hash chain. Payloads preserve complete bytes, including tool results larger than UI preview limits. The manifest can be rebuilt; validation and recovery use the commit chain as the source of truth. Do not edit the original files manually.

| Question | Evidence to inspect |
|---|---|
| What did the model receive? | `attempt.started` payload: the final serialized request body |
| What did the model return? | `attempt.response` payload; also all `stream.frame` events for streaming |
| Was a request retried? | Separate attempts for the model call and `attempt.retry_scheduled` |
| Were tool arguments valid? | `tool.requested`, `tool.arguments_validated`, or `tool.validation_failed` |
| What did a tool return? | The full `tool.completed` payload, paired by `tool_call_id` |
| Could a tool have run without a confirmed result? | `tool.outcome_unknown`; do not automatically repeat a side effect |
| Why did execution stop? | `runtime.finished`, episode status, and the final state checkpoint |

Execution status and recording completeness are separate dimensions:

| Field | Meaning |
|---|---|
| `status=completed` | The application task completed normally |
| `status=awaiting_client_tool` | A durable handoff is waiting for client tool results; this is not corruption |
| `status=failed/cancelled/interrupted` | Execution failed, was cancelled, or was interrupted; committed records remain inspectable |
| `capture=complete` | Observed data was fully recorded within the declared capture boundary |
| `capture=partial/unknown` | Some data is missing or completeness cannot be established |
| `external_outcome=unknown` | The application cannot confirm the final outcome of a remote model or tool action |
| `replay=unsupported` | The producer did not automatically certify execution replay; inspection/playback remain available, and execution replay requires the conditions in section 6 |

A task may complete after a retry while an earlier remote outcome remains unknown. A trace write failure stops new model/tool actions. Receiving an answer does not by itself prove that its trace was saved.

## 5. Export an analysis view or a complete private record

The default analysis export contains event IDs, order, types, and status. It excludes prompts, tool content, native state, and signatures:

```bash
python -m membraneclaw_runtime.tracing.cli export \
  --trace-dir ./var/traces --episode "$EPISODE" \
  --output ./var/traces/analysis.zip
```

To preserve all original data for inspection or replay on another machine, explicitly select private replay:

```bash
python -m membraneclaw_runtime.tracing.cli export \
  --trace-dir ./var/traces --episode "$EPISODE" \
  --mode private-replay --output ./var/traces/private-replay.zip
```

Exports do not overwrite existing files. Use a new output filename when exporting again. A private bundle is created with mode 0600 and contains private inputs, tool data, and opaque continuation/signature data. Share it only with authorized recipients. Analysis bundles are marked derived/redacted and cannot restore execution losslessly.

## 6. Choose an offline replay mode

**Recording playback** validates a bundle and reconstructs recorded events and state. It executes no models or tools. The synthetic storage example also supports this mode:

```bash
python -m membraneclaw_runtime.tracing.cli replay \
  --bundle ./var/traces/private-replay.zip --offline
```

**Execution replay** drives the current runtime using RecordedProvider and RecordedTool. It checks request bytes, tool identities/arguments, and native history, with no fallback to a live network call:

```bash
python -m membraneclaw_runtime.tracing.cli replay \
  --bundle ./var/traces/private-replay.zip --offline --execute
```

Execution replay requires a completed runtime_v1/gateway_v1 recording and a matching adapter version. The storage-only example in section 2 does not meet those conditions; use a complete runtime task recorded in section 3. Cross-request HTTP client tool handoffs and interrupted recordings currently support recording playback only. Real-time parallel completion timing is not reproduced. A request/argument mismatch stops replay rather than requesting missing output from a live model.

## 7. Use streaming and HTTP

Add `--stream` to use native SSE in the CLI:

```bash
membraneclaw --provider gemini --model gemini-2.5-flash \
  --stream --no-mcp --trace-dir ./var/traces -p 'Explain the basic principle of reverse osmosis.'
```

Each SSE frame is persisted before delivery to a consumer. The CLI prints the final answer after the stream finishes. Tools cannot run until their arguments and the model response are complete. On cancellation, all bytes already received by the application, including additional frames and an unfinished suffix in the current HTTP chunk, are captured before cancellation propagates. Cleanup does not forward those frames or execute tools. The trace is marked partial; capture still requires a writable store and payloads accepted by the credential policy. Attempts that have delivered frames are not automatically retried or spliced together.

Start the HTTP server:

```bash
export AGENT_PROVIDER=gemini
export AGENT_MODEL=gemini-2.5-flash
export TRACE_DIR=./var/traces
export AGENT_HTTP_NO_MCP=on
uvicorn server:app --host 127.0.0.1 --port 8001
```

Send a local single-request task:

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Calculate 27 * 19"}],"stream":false}'
```

If `AGENT_API_KEY` is configured, clients must also send `Authorization: Bearer <AGENT_API_KEY>`. A shared API key does not identify an individual user. Client tool continuation requires the existing signed Open WebUI JWT identity. Anonymous requests cannot declare client tools or restore private continuations.

HTTP responses include the trace schema, episode ID, request ID, and locator. Subsequent client tool results must preserve their handles and the exact history prefix. Partial results are persisted; generation resumes only when all required results arrive. Identical submissions are idempotent. Cross-user handles, expired handles, edited history, and conflicting results are rejected.

With `stream=true`, the server sends keepalives after committed frames, then complete visible content/tool declarations, finish/usage, and `[DONE]`. **HTTP currently buffers visible model text rather than rendering it incrementally.** Raw frames and signatures are not exposed to the UI. Ordinary new chat turns create new episodes; the server does not infer hidden cross-task context from a user ID. See the [HTTP guide](docs/http-runtime.md) for the full protocol.

For partial client tool submissions, the reply is an acknowledgement with
`continuation.accepted_handles` and `continuation.pending_handles`, with no
executable `tool_calls`. Keep the original handoff history prefix for later
submissions; do not append the acknowledgement to that prefix. Repeating an
accepted result does not issue another model request.

A synchronous tool that times out or is cancelled may still finish its action.
The runtime waits for that worker while holding execution ownership and records
its `late_result` or `late_exception_type`. Its outcome remains unknown and is
never automatically retried. Inspect a returned receipt before deciding what to
do next.

Registered credentials are checked in raw bytes and decoded JSON strings/keys,
including SSE `data:` JSON, before persistence. This does not detect unknown
credentials or provide a general sensitive-data classifier.

## 8. Archive to a private GitHub repository

Local persistence does not automatically push to Git. Prepare an existing **private trace repository**, select a branch, and configure normal Git authentication. Also set `TRACE_ARCHIVE_GITHUB_TOKEN` so the publisher can verify that the target repository is private.

```bash
python -m membraneclaw_runtime.archives.cli publish \
  --trace-dir ./var/traces --episode "$EPISODE" \
  --remote git@github.com:YOUR_ACCOUNT/YOUR_PRIVATE_TRACE_REPO.git \
  --branch trace-data --state-dir ./var/trace-archive
```

| Archive status | Meaning and next action |
|---|---|
| `published` | The remote ref was verified to contain the snapshot commit and all expected blobs |
| `pending` | Local data remains available, but archival is unconfirmed; fix connectivity/authentication and retry the same command without rerunning the model |
| `not_configured` | No target is configured; no repository is created and nothing is uploaded |

Retries are idempotent by snapshot hash. The publisher stages only exact snapshot paths and never force-pushes. If the remote branch advances, it creates a fresh checkout and appends the snapshot while preserving unrelated commits. Archive receipts are stored separately from runtime traces.

See [archive operations](docs/archive-operations.md) for restore, access control, and retention. Deleting current files does not remove Git history. Choose a private target before publishing raw records.

## 9. Integrate evaluation and check health

See [evaluation tracing](docs/evaluation-tracing.md) for installation and role configuration. Solver, router, teacher, judge, and recovery calls use the same gateway but can select different providers. Judges and tool-disabled roles receive no tools.

```bash
python -m pip install -e experiments/auto-evaluate
python -m auto_evaluate auto --help
python -m membraneclaw_runtime.operations \
  --trace-dir ./var/traces --archive-state ./var/trace-archive
```

Evaluation records retain canonical event IDs and private bundles. A trimmed judge view lists omitted event IDs without trimming the original trace. Reused results do not count as new API calls. Set the dataset split explicitly for research exports; test traces must not be automatically mixed into training data.

The health command reports storage size, incomplete episodes, unknown tool outcomes, requests/retries, pending continuations, and archive status. Distinguish execution failures, incomplete capture, and archive backlog before deciding which operation to retry.

## 10. Verification and troubleshooting

Offline validation covers synthetic native protocols for all three providers, real local MCP, file faults/locks, HTTP continuation, network-free replay, and local bare Git archives. CI covers Python 3.11–3.13. Live API, WaterTAP, Open WebUI, and private GitHub archive smoke tests require separate results; passing offline tests does not establish live end-to-end success.

| Symptom | Check or action |
|---|---|
| Missing model/key or ConfigurationError | Select a reviewed provider/model and set the matching key; unsupported options are rejected before sending |
| trace_storage_error | Check disk space, permissions, and the local filesystem; stop new actions and preserve existing data |
| continuation_busy | The original worker may still be active; do not declare a task dead while it is waiting on the network |
| outcome_unknown | The remote action may already have happened; inspect the trace and remote facts before repeating a side effect |
| Output bundle already exists | Choose a new filename; exports do not overwrite files |
| Replay mismatch or unsupported version | Preserve the bundle and use recording playback to investigate; do not automatically switch to a live API |
| Archive pending | Retry archival without rerunning the task; check Git authentication, private-target verification, and connectivity |

Further references: [storage format](docs/tracing.md), [runtime](docs/runtime.md), [streaming](docs/streaming.md), and [replay](docs/replay.md).

Live API smoke tests require explicit opt-in. Set the corresponding key and reviewed `LIVE_OPENAI_MODEL` / `LIVE_ANTHROPIC_MODEL`, then run:

```bash
RUN_LIVE_PROVIDER_SMOKE=1 python -m unittest discover \
  -s tests/providers/contract -p test_live_native.py -v
```

A provider with no flag, key, or model is skipped rather than counted as passed. Live calls consume API usage. This check verifies a tool-free task only; it does not establish WaterTAP or Open WebUI integration success.

## PR map

The PRs are stacked by dependency. Each diff contains its own stage of work. They are not merged automatically.

| Implementation plan scope | PR |
|---|---|
| PR 1: storage foundation, merged upstream | [rtong/membraneclaw #41](https://github.com/rtong/membraneclaw/pull/41) |
| PR 2: Gemini and gateway | [#1](https://github.com/regulusv/membraneclaw/pull/1) |
| PR 5: provider-neutral runtime and CLI | [#2](https://github.com/regulusv/membraneclaw/pull/2) |
| PR 3: OpenAI Responses | [#3](https://github.com/regulusv/membraneclaw/pull/3) |
| PR 4: Claude Messages | [#4](https://github.com/regulusv/membraneclaw/pull/4) |
| PR 6: durable streaming | [#5](https://github.com/regulusv/membraneclaw/pull/5) |
| PR 7: HTTP and client tool continuation | [#6](https://github.com/regulusv/membraneclaw/pull/6) |
| PR 8a: export and offline replay | [#7](https://github.com/regulusv/membraneclaw/pull/7) |
| PR 8b: evaluation integration | [#8](https://github.com/regulusv/membraneclaw/pull/8) |
| PR 8c: Git archive and operations | [#9](https://github.com/regulusv/membraneclaw/pull/9) |
| User guide and explicit live checks | [#10](https://github.com/regulusv/membraneclaw/pull/10), branch `codex/tracing-user-guide` |
