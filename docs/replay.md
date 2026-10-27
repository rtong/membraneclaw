# Inspect, export and replay

Local operator commands (installed package):

```
python -m membraneclaw_runtime.tracing.cli inspect --trace-dir ./var/traces --episode EPISODE_ID
python -m membraneclaw_runtime.tracing.cli validate --trace-dir ./var/traces --episode EPISODE_ID
python -m membraneclaw_runtime.tracing.cli export --episode EPISODE_ID --output analysis.zip
python -m membraneclaw_runtime.tracing.cli export --episode EPISODE_ID --mode private-replay --output private.zip
python -m membraneclaw_runtime.tracing.cli replay --bundle private.zip --offline
python -m membraneclaw_runtime.tracing.cli replay --bundle private.zip --offline --execute
```

inspect prints the full timeline/state and full payloads with explicit base64 encoding. Treat stdout as private data. Short episode IDs must resolve uniquely; tenant/run/episode keys are also accepted. validate reconstructs the immutable commit chain and verifies every payload, rather than trusting manifest.json.

Default analysis export contains event IDs, sequence, event types, status and redacted completeness only. It excludes prompt/tool content, native state, signatures and payloads. It is explicitly derived/redacted and cannot restore execution. Private replay export is an explicit local operator choice containing all sealed native data. It creates a new mode-0600 ZIP without overwriting another file. Bundles have a hashed file inventory plus the original snapshot hash and restore into an isolated directory. Duplicate entries, unexpected/unsafe paths, changed hashes and oversized bundles are rejected. The initial uncompressed read cap is 256 MiB; applications may explicitly raise load_bundle(max_bytes=...) when needed.

Recording playback validates and reconstructs existing events/state; it executes nothing. Execution replay drives the current runtime with RecordedProvider and RecordedTool, never an HTTP client or live handler. It compares native request method/path/body bytes and ordered tool identity/raw/validated arguments. Tool result payload kind/error semantics remain intact. Runtime-generated local IDs and timing are excluded from native-history comparison; provider IDs and opaque data remain exact. No network fallback exists. A mismatch terminates replay.

Execution replay v1 requires a completed runtime_v1/gateway_v1 recording and the matching adapter version. Pending/interrupted/client-handoff recordings remain available for lossless recording playback; execution replay of cross-request client handoffs is not implemented. Replay does not model real-time scheduling or latency, and observed parallel completion timing is not re-created. Native protocol/result pairing is deterministic. Existing records without the full ModelRequest use an explicit native-config reconstruction path; new runtime.started events also save the complete ModelRequest.

Tests block socket creation while replaying native Gemini, OpenAI and Claude tool loops, verify large payload/opaque retention through ZIP restore, reject mutated bytes/arguments/requests and verify that default exports omit private fixtures. Run `python -m unittest discover -s tests/runtime -v`. No live model or tool is used in these replay tests. Rollback removes the operator commands; raw episodes and bundles remain readable under their recorded schema/version.
