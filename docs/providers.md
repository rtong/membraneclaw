# Traced providers: PR 2

This PR adds provider-neutral types, a registry, the mandatory ModelGateway,
and a non-streaming Gemini Developer API adapter. It depends on the trace
storage contracts in upstream PR #41. Existing Qwen, agent CLI, HTTP server and
evaluation behavior are unchanged. Gemini tools are declared and returned
explicitly; this layer never executes them.

## Supported implementation profile

| Setting | Value |
| --- | --- |
| Provider | `gemini` |
| API family | `generate_content_v1beta` |
| Endpoint | `https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent` |
| Adapter schema | `gemini_v1` |
| Reviewed model ID | `gemini-2.5-flash` |
| Implemented capabilities | Text, multiple turns, explicit/parallel tool calls, usage and errors |
| Deferred capabilities | Streaming (PR 6), vision, structured output, built-in tools, Vertex AI |
| Reasoning namespace | `{"gemini":{"thinkingConfig":{...}}}` |

The model ID is an explicit allowlist entry, not a latest-model recommendation
or proof of current account availability. No model name is guessed from another
provider and no fallback occurs. The remote model ID may be an alias; a stable
provider weight revision is not implied. The returned `modelVersion` stays in
the full native capture. Live availability/protocol behavior needs the opt-in
smoke check below.

This implementation uses CPython's standard-library HTTPS client directly,
rather than a model SDK. This is a deliberate refinement of the plan's SDK spike:
the final serialized body and arbitrary raw response bytes are visible before
JSON parsing, including malformed bodies. There is no SDK dependency to pin,
no implicit SDK retries, and no automatic function execution. The API family and
adapter schema are fixed in code; CI exercises CPython 3.11–3.13. The package
still has no third-party runtime dependencies, so no dependency lockfile changes
are necessary.

Public generation controls are optional temperature (finite 0–2) and output
limit (integer 1–65,536). The reviewed Flash thinking budget is -1 (dynamic) or
0–24,576; `includeThoughts` is boolean. Other model profiles/fields require a
reviewed adapter change, not an arbitrary passthrough dictionary. Omitted options
remain omitted: the API's evolving server-side defaults are not known or
claimed to be locally expanded. These controls do not expose private model
internals; returned thought text is a summary, and signatures are opaque data.

## Install and verify offline

```sh
python -m pip install --no-deps .
python -m compileall -q membraneclaw_runtime
python -m unittest discover -s tests/tracing -v
python -m unittest discover -s tests/providers/contract -v
```

Fixtures are synthetic and contain no real model results or credentials.
Provider tests use a fake transport and loopback HTTP server, never Google.
The live smoke test is skipped unless all three explicit settings below exist.
The workflow runs storage and provider tests independently; it does not install
the Qwen/WaterTAP application or declare those integrations verified.

## Use the gateway

```python
import asyncio

from membraneclaw_runtime.gateway import Budget, ModelGateway, RetryPolicy
from membraneclaw_runtime.providers.gemini_transport import GeminiTransport
from membraneclaw_runtime.providers.registry import default_registry
from membraneclaw_runtime.providers.types import ContentBlock, ModelRequest, ModelRoute
from membraneclaw_runtime.tracing import FileTraceStore

async def main():
    route = ModelRoute("gemini", "gemini-2.5-flash", "generate_content_v1beta")
    adapter = default_registry().create(route)
    gateway = await ModelGateway.start(
        FileTraceStore("./var/traces"), adapter, GeminiTransport.from_env(), route,
        role="solver", budget=Budget(max_requests=2),
        retry=RetryPolicy(max_attempts=2),
    )
    async with gateway.execution():
        output = await gateway.generate(ModelRequest(
            route, (ContentBlock.text_block("Reply with OK."),),
            max_output_tokens=128,
            provider_options={"gemini": {"thinkingConfig": {"thinkingBudget": 0}}},
        ))
        await gateway.finish("completed" if output.finish_status == "completed" else "failed")
    # The response is available to the caller; normal logs should use only IDs/status.
    return gateway.recorder.episode, output

# asyncio.run(main()) contacts Google and requires a privately supplied key.
```

The same gateway accepts another registered adapter and a role such as
`judge`, `router`, or `teacher`. It contains no provider-name dispatch logic.
Unknown providers/API families fail before requests. GPT and Claude adapters
are PR 3/4. Actual evaluation routing is PR 8.

`ModelRequest.content` contains only this turn's NEW user input.
`ProviderState.native_history` owns history. Repeated full histories would be
duplicated by the caller and are not a supported input format.
Route/API/model, adapter version, system instruction, tools and generation
configuration are locked for an episode. Configuration changes require a new
episode; never switch providers while tool calls are pending.

## Native state and tools

The adapter's prepare/normalize/advance functions are pure and offline-testable.
`prepare` produces immutable final HTTP-body bytes. `normalize` creates an
application view while retaining the complete native response and its payload
reference. Candidate policy is the first array entry (position 0); all candidates
and unknown response fields remain in the raw payload. Usage covers the whole
response, not only the selected candidate.

`advance` copies the selected native content—including all parts, their order,
thought signatures and unknown fields—into history. It never regenerates content
from text/tool-call summaries. A state document converts tuple containers to JSON
arrays at an explicit serialization boundary without weakening canonical_json_v1.

Tool definitions use a conservative JSON Schema subset and the REST
`parametersJsonSchema` field. Supported keywords are type, description,
properties, required, items, enum, boolean/schema additionalProperties, minimum,
maximum, minItems, maxItems, title and default. Unsupported keywords, duplicate/invalid portable
names or foreign provider options fail explicitly; no constraints are dropped.
The portable tool-name limit is intentionally narrower than Gemini's native
limit. Name mapping and argument validation/execution belong to PR 5.

Each returned tool call has a unique internal ID, its optional provider ID,
name, native position and raw/parsed JSON-object arguments. Raw argument spelling
is retained in the complete response bytes rather than claimed recoverable from
the parsed dictionary. `gateway.add_tool_results(results)` requires exactly one
result per pending internal ID, persists results and the new state, then allows
the next generation. Results are ordered by original calls even if callers supply
them in reverse order. Present provider IDs are echoed; absent IDs are not invented.
Tool errors use a native error response. No actual external tool is invoked here.

A future runtime must hold `async with gateway.execution()` around the entire
model/tool/model segment. Individual calls acquire this context if not already
owned by the current task. Concurrent use of the same gateway is rejected.
The guard remains held during network I/O and disk work; recovery cannot interrupt
a live request. Disk commits run off the async event loop.

## Capture and failure semantics

The transport awaits request capture with its FINAL serialized bytes before the
single HTTP request. It awaits raw response capture before returning to the
adapter/parser. The gateway also verifies that returned data matches the captured
response and refuses missing/repeated/changed capture callbacks.

The transport requests identity encoding. Gzip responses preserve both compressed
bytes and decompressed client-visible bytes. Malformed gzip/unsupported encodings
are retained and marked partial, never parsed as valid JSON. Capture covers the
HTTP client's body bytes, not TLS packets or private model state. Non-streaming
responses are buffered in memory; this is not the PR 6 streaming/backpressure path.

Only safe response headers are kept: content-type, content-encoding, retry-after,
date, x-request-id. Authentication/cookies and credential query parameters are
excluded. The trusted host and endpoint path are fixed; redirects do not generate
another request. Actual keys are passed only in the live authentication header
and registered with the gateway capture policy. Credential-bearing responses
are rejected before either compressed or decoded data is persisted. No prompt,
tool data, output, signature or exception URL is emitted to normal logs.

CapturePolicy is defense in depth, not a universal redactor. Unknown credentials,
transformed/encoded secrets and arbitrary sensitive user data require upstream
handling. Private trace storage and archive policy still apply. A rejected
credential-bearing payload cannot be called a complete raw capture.

Storage errors stop new actions and prevent normalization/output consumption.
The recorder keeps stable IDs for uncertain storage retries; retry storage only,
never repeat a model/tool action to repair tracing. A failed gateway refuses new
generation. Storage may be too broken to write its own failure event: surface
`TraceStorageError` outside tracing, then inspect/recover persisted state.
A disconnected/interrupted remote request may still complete and incur cost.

## Attempts and budgets

429 and selected 5xx (500/502/503/504) get bounded retries. Other HTTP statuses,
including redirects and configuration/schema errors, do not. Retry-After accepts
seconds or an HTTP date; the gateway never retries earlier than the provider's
delay. Otherwise it uses bounded exponential delay plus jitter. A delay beyond
the total time budget stops rather than clipping Retry-After.

Each HTTP request has its own attempt ID, request/response payloads and lifecycle
events under one logical model-call ID. Parsing failures do not erase raw bytes.
Network/timeouts default to unknown external outcome and no retry. Service/proxy
5xx errors also retain unknown external generation outcome even if a later
attempt succeeds. Explicit
`retry_timeouts=True` permits a new attempt with separate possible billing;
unknown outcome and partial capture remain sticky after a later success.
Partial bodies are never concatenated with retries or executed as tool calls.

Budgets cover actual request count, total elapsed time and per-attempt timeout.
The HTTP worker is interrupted/drained before the execution guard is released
after timeout/cancellation. Native DNS/TLS or OS operations that do not promptly
respond to socket shutdown can delay cancellation; there is no claim of remote
cancellation or exactly-once service execution.

Optional token budgeting requires an explicit output cap and a trusted upper
bound for the ENTIRE input request, including native history/system/tools.
The gateway reserves input upper bound + output cap before EVERY attempt,
retains reservations for uncertain/failed requests, and does not release them
when a later attempt succeeds. Each attempt is accounted as the greater of its reservation and known actual
usage; these per-attempt debits are summed before subsequent admission. Known
overruns and separate unknown-usage reservations therefore both remain charged. This is conservative admission control, not a
guarantee against inaccurate caller bounds or unexpected service accounting.
No pricing/fee estimates are fabricated.

Usage fields are nullable. Gemini promptTokenCount includes cached input;
candidatesTokenCount and thoughtsTokenCount remain distinct fields. The
provider's totalTokenCount is used directly; cached/reasoning counts are never
added to it. Failed/missing-usage attempts make the episode's usage accounting
unknown even if a later response reports usage. A full HTTP response with invalid
JSON may still have complete byte capture while semantic replay is unsupported.

## Explicit real smoke

Privately supply `GEMINI_API_KEY`, choose the reviewed model, then invoke:

```sh
python -m membraneclaw_runtime.providers.smoke --model gemini-2.5-flash --trace-dir ./var/traces
```

This command sends one real text request, no tools and no automatic retry.
It is separate from the existing agent CLI and prints only episode ID, finish
status and nullable usage. The model option is required; an empty env example
does not silently select a model. Review the resulting private trace with PR 1's
validation API. A key/model/access error is a failed smoke, not an offline pass.

To opt into the live unittest, set `MEMBRANECLAW_LIVE_SMOKE=1`,
`GEMINI_API_KEY`, and `GEMINI_MODEL_ID=gemini-2.5-flash`. Otherwise it is
reported as skipped. No live key or real Gemini smoke was available during
implementation. Do not call synthetic fixtures real experimental results.

## Review and rollback

PR 2 is stacked against the PR 1 branch in the fork so its review diff excludes
the storage implementation. After PR #41 is merged upstream, replay/retarget the
PR 2 commits onto upstream main and open the upstream review. Do not merge PR 2
into upstream before its prerequisite; nothing here authorizes merging either PR.

Rollback removes/does not select the Gemini route. Existing Qwen behavior is
unchanged. Retain trace data and schema versions. Next in the plan's execution
order is PR 5: provider-neutral tool execution loop and agent CLI integration.

Primary protocol references checked during implementation:

- [GenerateContent REST API](https://ai.google.dev/api/generate-content)
- [Function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Thinking configuration and signatures](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [Thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)

These describe the wire protocol. They do not substitute for an account-specific
live smoke. Adapter/API/model support is intentionally narrower than the current
remote product surface.
