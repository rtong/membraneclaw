# Provider-neutral runtime and CLI (PR 5)

PR 5 builds on the native provider/gateway in PR 2 and the file-store contracts
in PR 1. It adds the model → tools → model loop and reuses existing builtin,
Google REST and MCP implementations. It does not implement GPT/Claude transports,
streaming, HTTP client-tool continuations, evaluation replay or Git archiving.
PR numbers identify scope: the agreed execution order puts PR 5 before PRs 3/4.

## Install and run

From a source checkout, Python 3.11+:

```sh
python -m pip install '.[tools]'
python cli.py --provider gemini --model gemini-2.5-flash --trace-dir ./var/traces -p 'Convert 1 bar to kPa and explain the result.'
```

Supply GEMINI_API_KEY privately through the environment. A real Gemini call has
not been verified without an operator-provided key. The model must have a reviewed
capability profile; currently that is gemini-2.5-flash. No silent fallback occurs.
The installed entrypoint is `membraneclaw`, or `python -m membraneclaw_runtime.cli`.
Use `--no-mcp` for only the local builtins. The base install needs jsonschema;
MCP and Google add optional dependencies from the tools extra.

Without a provider flag or AGENT_PROVIDER setting, the CLI keeps the original
Qwen route. Existing `-p/--prompt` and `-f/--file` options remain. Qwen uses its
original agent, tool behavior, per-operator Google binding and interactive history;
its existing application dependencies still come from requirements.txt. Importing
or running a new-provider path does not import Qwen or apply agent.compat patches.
RAG file input is rejected for new providers until explicitly implemented.

New-provider configuration can use AGENT_PROVIDER=gemini and AGENT_MODEL or
GEMINI_MODEL_ID, or the explicit flags. A Gemini model env var does not override
the Qwen model when Qwen is selected. No-prompt mode is interactive: each entered
prompt starts a new episode. This mode deliberately has no opaque continuation
between completed episodes. Within an episode, all native model/tool turns retain
the provider's original state, including signatures. HTTP/conversation handles
are separate work in PR 7.

The CLI prints the episode ID before execution, the final answer when available,
and final status/reason. Failed/incomplete runs exit nonzero. Storage errors stop
new external work and expose only the exception type in ordinary CLI error logs.
Full tool results go into private traces rather than debug previews. The Qwen
renderer's existing preview limits remain unchanged.

## Reuse boundaries

- `agent/tools/builtin_impl.py` contains the shared calculator/HTTP/date logic;
  the original Qwen classes are wrappers. New runtime HTTP reads return the full
  result, while the Qwen wrapper retains its original 200,000-byte limit.
- `agent/tools/google_impl.py` contains the existing identity-bound Google REST
  logic. Qwen wrappers preserve the original API and 20,000-character output cap.
  The new runtime selects uncapped results and observes actual access tokens so
  the gateway can reject those values before trace persistence.
- MCP adapters call the existing server implementations, not copied chemistry,
  RO simulation, conversion or constraint code. They capture the complete native
  CallToolResult, including text/image blocks, structured content and error flags.
  Available server name/version is included in the tool metadata. No MCP reconnect
  retry or HTTP redirect following is added by the runtime.

The runtime package now also includes agent and mcp_rocalc Python modules so its
installed entrypoint can reuse the same code. The heavy mcp_watertap simulation
server remains an external deployment/configured endpoint. Merely packaging agent
modules does not import Qwen or its optional dependencies.

The tested MCP extra pins mcp 1.20.0, Pydantic 2.12.3 and pydantic-settings 2.11.0.
Allowing the newest Pydantic with this SDK broke an SDK-internal import during
setup. These are compatibility pins tested by CI, not latest-version claims.
The optional legacy-test extra installs Qwen's base/test dependencies so the
wrapper/identity tests can run without a live vLLM backend or the heavy RAG stack.
Existing requirements.lock.txt is not rewritten or claimed to lock this new extra.

## Tool connection configuration and identities

RO_CALC_MCP, RO_CALC_MCP_URL, RO_CALC_MCP_TOKEN, RO_MCP, RO_MCP_URL,
RO_MCP_PYTHON and MCP_BEARER_TOKEN retain their existing meaning. If enabled and
available, ro-calc uses the local Python server or its configured HTTP endpoint;
ro-chem uses the configured endpoint or existing simulator environment.
Use RO_CALC_MCP=off/RO_MCP=off or `--no-mcp` to disable discovery. A configured
server that fails discovery is an explicit startup error, rather than silently
removing tools. MCP discovery/initialization is setup, not a model-request attempt.

Google tools are opt-in for the new CLI through `--google-tools`. The CLI passes
its local operator token path explicitly. Library callers pass the verified user's
path to runtime_session(google_token_path=...). None means no Google tools: there
is never an implicit fallback to the owner's token. The default new-runtime Google
subset matches the existing default REST subset; callers can supply an explicit
allowlist. Existing Qwen/server Google identity behavior stays intact. Official
Workspace MCP OAuth forwarding remains in the legacy path; new runtime support
here uses the shared Google REST tools.

The runtime records tool schemas, their hashes, implementation versions, server
metadata and both directions of the tool-name mapping. Native MCP names are mapped
to a portable 64-character provider name. Mapping collisions are rejected rather
than silently overwriting another handler. Long names include a deterministic hash
suffix. Unsupported provider schema features fail before a model request. Gemini's
reviewed schema subset now includes title/default and schema-valued additional
properties so the existing ro-calc declarations can pass without losing constraints.

## Execution and trace behavior

AgentRuntime owns one episode and is single-use. It holds the store execution
lock across all model waits, tool work and state commits. A gateway execution
context serializes recorder updates; parallel tool tasks never write the recorder.

1. Persist runtime configuration, complete schemas/mappings and input.
2. Check step/context/time admission. The full native request is checked against
   an optional byte cap without trimming or claiming it is a token count.
3. Call ModelGateway, which persists exact model requests/responses before use.
4. For tool calls, persist raw arguments and IDs, apply explicit defaults, and
   validate arguments with JSON Schema. Persist validation errors or the validated
   effective arguments. Invalid input produces a known error result without
   invoking a handler, allowing a later model turn to correct it.
5. Commit tool.started before invoking each handler. Capture complete results as
   exact bytes, UTF-8 text or canonical JSON with an explicit payload kind; binary
   provider input uses an explicit base64 wrapper. Results match call IDs even for
   same-name calls or out-of-order completion. No UI truncation is applied.
6. Persist native tool-result history and state before asking the model again.
   Snapshots include provider state, pending calls/results, tools/mappings,
   runtime limits/usage, next action, logical model IDs and completeness state.
7. Mark completed only after an explicit completed model output with no pending
   tool calls. Blocked, incomplete, unknown and inconsistent output stop as failed.

Tool events carry model_call_id/tool_call_id in their canonical event envelopes;
raw payloads retain provider IDs. Application snapshots accompany events and
model commits, rather than replacing the provider's native history with text.

ToolFailure is a declared known domain failure that may be sent back for model
correction. A timeout or unexpected handler exception is outcome_unknown: preserve
that uncertainty, stop the loop, leave remaining side effects unexecuted, and do
not retry automatically. MCP timeout/failure follows this rule. A known MCP
isError flag is retained as a known tool error. Failures returned inside a server's
own text remain present as server data, not reclassified as invented success.

Storage failure prevents new model/tool actions. Already-running work cannot be
made atomic with recording. Startup recovery identifies interrupted work; the
runtime never automatically resumes an old episode or reissues a pending side
effect. Cancellation marks the run cancelled/partial with unknown external outcome.
Synchronous workers are drained before the execution lock is released. A handler
that ignores cancellation or blocks in OS/native code can delay shutdown; this is
not a sandbox or a hard process-termination guarantee.

## Limits and concurrency

Defaults are operational caps: 12 logical steps, 32 requested tool calls, 60 seconds
per tool, 36 actual model requests and 300 seconds of execution time in the CLI.
The values are configurable; they are not performance claims or scientific cutoffs.
The gateway's whole-execution elapsed budget also covers tool time: a new tool/model
operation is refused once it is exhausted. The tool timeout is capped by remaining
time. OS/native operations can still delay cancellation as described above.

Relevant flags: --max-steps, --max-tool-calls, --tool-timeout, --max-requests,
--max-seconds, --max-context-bytes and --parallel-read-only. Requested tool calls,
including invalid arguments, count against the tool budget. There is no automatic
context trimming, token estimation or provider fallback. The lower gateway's
optional token admission contract remains available to library callers with trusted
full-input bounds; the CLI does not pretend to estimate tokens after tool results.

Tools are serial by default. Opt-in parallel mode executes a whole call round in
parallel only when every selected tool is explicitly both read_only and
parallel_safe. Otherwise the round stays serial. Schema declarations/annotations
do not automatically establish safety. Local unit conversion/limit checks may be
classified read-only; external simulations and Google writes are not granted
parallel execution by default. Persisted completion order may differ from the
provider-required result order; both associations remain visible.

## Verification and rollback

```sh
python -m pip install '.[tools,legacy-test]'
python -m unittest discover -s tests/tracing -v
python -m unittest discover -s tests/providers/contract -v
python -m unittest discover -s tests/runtime -v
python agent/test_google_tools.py
python agent/test_google_oauth.py
python agent/test_identity.py
python mcp_rocalc/test_rocalc.py
```

Runtime tests include three independent fake provider routes, two synthetic
simulations plus a constraint call, a real Gemini adapter driven by synthetic
responses, and real local MCP unit/constraint calls. These are offline fixtures,
not real Gemini/GPT/Claude quality results. CI also tests the installed runtime
outside the checkout. The old Google/OAuth/identity programs run with mocked
HTTP and real wrapper types. Their own check counts are separate from unittest
method counts; no live Google account is required.

Not verified without external services/credentials: real Gemini solver/tool smoke,
GPT/Claude adapters (PRs 3/4), live Qwen/vLLM, WaterTAP/Reaktoro, Open WebUI, HTTP
continuations, streaming and archive/replay. Roll back by selecting Qwen, or revert
PR 5 while retaining versioned traces. The tool-implementation extraction keeps
legacy wrapper behavior covered. This PR is stacked on PR 2 in the fork; no PR is
merged automatically. Next in the agreed plan: PR 3, the OpenAI Responses adapter.

For synchronous tools, timeout or cancellation does not stop the worker. The
runtime drains it under execution ownership and records `late_result` (binary
values use explicit base64 encoding) or `late_exception_type`, including during
parallel cancellation. The outcome stays unknown with its timeout/cancellation
marker. A recorded late receipt is evidence for reconciliation, never permission
to retry the action automatically.
