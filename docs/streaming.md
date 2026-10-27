# Durable native streaming

Run the traced CLI with `--stream` (plus the provider/model flags). The CLI prints the final answer after the stream completes. HTTP rendering is handled by the subsequent HTTP integration; the runtime exposes an awaited `on_frame` consumer for committed frames. Never expose raw frames on a public channel: they can contain private content and signatures.

All three REST transports support SSE. A pull-based reader separates protocol frames from TCP chunks, preserves exact frame bytes and receipt timestamps, and awaits one trace commit per frame before delivering it. There is no producer queue. Both storage and a slow consumer stop the next read. The conservative initial policy uses one-frame batches; this is intentionally durability-first, with configurable batching deferred until an equivalent bounded durability contract is measured.

The assembler reconstructs Responses terminal output items, Messages thinking/signature/tool-argument deltas, and Gemini candidate parts in order. Gemini signed/empty parts remain separate. The original frames remain authoritative evidence; attempt.response is explicitly marked as a derived aggregate. Tools run only after explicit response completion and successful aggregate persistence. Usage is final cumulative metadata, never the sum of frame snapshots. Missing terminal evidence makes capture partial. A stream that delivered any frames is never automatically retried or spliced into another attempt. Unknown/duplicate Messages events and unordered Responses sequence numbers fail closed after raw capture.

The SSE frame limit is 4 MiB. An oversized/incomplete frame is captured as partial then rejected; it is not silently truncated. Native response/history size still grows with the complete output, as required for lossless capture. This is not a constant-memory system. HTTPX 0.28.1 owns the async connection, disables redirects/proxy environment trust, and closes it on cancellation/error. Single-host execution guards remain held through cleanup. Cancellation cannot guarantee the provider stopped billing.

Run `python -m unittest discover -s tests/runtime -v`. Tests include byte-by-byte framing, signature/argument cut points, terminal mismatch/duplicates, final usage, commit-before-consumer, slow consumer backpressure, write failure, partial termination, cancellation, and the actual HTTPX stream path with a synthetic byte source. Real three-provider SSE smoke remains unverified without credentials.

## Local synthetic measurement

Run `python tests/runtime/benchmark_streaming.py` after installing the current distribution. Results below measure assembler-only processing versus full FileTraceStore capture, including request/aggregate/state commits. They exclude network/model latency and are not production throughput estimates. No performance pass threshold is claimed.

Environment: Linux x86_64, Python 3.12.14; 5 synthetic runs, 31 frames/run, 3,840 output characters. No network or provider latency. Files include commit/checkpoint metadata.

| Mode | First frame ms (median) | Frame gap ms (median) | Frames/s | Bytes on disk |
|---|---:|---:|---:|---:|
| assembler only | 0.024 | 0.011 | 72692.0 | 0 |
| FileTraceStore | 10.748 | 7.546 | 115.5 | 81486 |

Synchronous persistence has material per-frame cost in this small fixture. The initial implementation keeps the documented durable-before-delivery guarantee; a future batching change needs its own loss-bound and latency tests. Rollback: omit `--stream` to use the existing captured non-streaming path. Existing partial episodes must remain partial.

Sources: [OpenAI events](https://developers.openai.com/api/reference/resources/responses/streaming-events), [Claude streams](https://platform.claude.com/docs/en/build-with-claude/streaming), [Gemini signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).

On cancellation the framer retains each received byte until its capture commit
has settled. Cleanup captures remaining complete frames and the unfinished
suffix exactly once, without parsing or forwarding them. Storage failures or
credential policy rejection still stop capture. Raw credential checks also
inspect decoded SSE data JSON keys and values before any frame is committed.
