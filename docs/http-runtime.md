# Traced HTTP and client tools

Install `.[tools,http]` and configure `AGENT_PROVIDER`, explicit `AGENT_MODEL`, provider API key and `TRACE_DIR` on a local persistent Linux disk. Start `uvicorn server:app --host 127.0.0.1 --port 8001`. Configure `AGENT_API_KEY` for the existing shared server authentication. Qwen remains the default path; the new branch imports neither Qwen nor its compatibility patches. Health returns status/provider without paths or upstream credentials.

Open WebUI uses `/v1/chat/completions`, with ordinary OpenAI messages, client function declarations and tool result messages. Its existing signed user JWT forwarding must be configured (see agent/identity.py). A shared API key alone is not a user identity. Anonymous requests may perform a single task with server tools; they cannot declare client tools or resume private handoffs. Google tools bind only to the verified user's token path. Request IDs and trace IDs are observation metadata, never authorization.

A new ordinary chat request creates a new episode. Submitted text history is explicitly serialized as a portable role/text document in the new user input; system messages remain system instructions. No hidden conversation state is guessed from text/user ID. This first route accepts text, tools and stream. Generation overrides and route/model mismatches fail explicitly. RAG/attachments remain on Qwen.

A client tool round first commits native state, validated arguments, pending calls, owner hash, expiry, exact outgoing history-prefix fingerprint and random handles, then changes status to awaiting_client_tool before replying. The client must append the assistant message exactly and return text tool results with those handles. A request may return a subset; the server persists results and waits. Identical submissions are idempotent; a different result for the same handle is rejected. Only the complete set can transition to running and issue the next request. Local and client tools may share a round; local results are retained while waiting.

Authoritative state is inside the episode commit/checkpoint. Handle discovery currently scans paginated episode checkpoints (O(number of episodes)); an index can be added as a rebuildable cache when measured scale requires it. File locks cover result acceptance, restore, model/tool execution and new handoff/terminal commit. Concurrent submission returns 409 busy. A running episode left after a crash returns outcome_unknown and must go through storage recovery; it never reissues the model automatically. A prior completed submission can return its cached visible reply without another model request. Adapter/tool version or schema changes reject resume. Provider disablement never falls back to Qwen.

Default TTL is 900 seconds; waiting time is excluded from the active execution-time budget, while request/token/step/tool counters persist. Expired handles return 410. Cross-user handles return 403; edited prefixes/results return 409. Authentication and expired/unknown outcomes are covered by offline service and ASGI tests.

`stream=true` uses the durable native stream path. HTTP emits bounded keepalive comments after committed frames and buffers visible text/tool declarations until the response is complete. This avoids presenting partial tool arguments or opaque signatures and keeps the next history prefix identical to the persisted reply. One final content/tool chunk, one finish/usage chunk and `[DONE]` terminate the UI response. A stream error emits an error envelope and `[DONE]`, without claiming success. A queue of capacity one applies consumer backpressure; disconnect cancels the upstream task and waits for cleanup. This version does not provide incremental UI text rendering.

Usage reports observed totals, null when unknown; split prompt/completion counters are null where no aggregate exists. Cached idempotent replies have null usage rather than claiming a second paid generation. Trace schema, episode ID, request ID and locator accompany each response. HTTP trace download is not exposed: export remains an authenticated local operator action.

Run `python -m unittest discover -s tests/runtime -v`, plus existing identity/OAuth/Google tests. Offline fixtures cover restart, partial/multiple results, idempotence/conflict, owner/history/TTL rejection, simultaneous requests, interrupted sends, stream correlation and ASGI SSE. Real Open WebUI and live provider smoke are unverified without the deployment. Before rollout, check that the deployed Open WebUI version preserves assistant content/tool_call IDs and handles partial tool rounds.

Rollback: set AGENT_PROVIDER=qwen for new tasks, or reject new tasks on the disabled route. Drain old handoffs on a compatible isolated worker, or return provider_disabled; never translate opaque state into a different provider.

Partial result submissions return a non-executable acknowledgement with
`continuation.status=partial_results`, `accepted_handles`, `pending_handles`, and
`preserve_history_prefix=true`. Its assistant message has no tool declarations
and its finish reason is `stop`; the trace remains `awaiting_client_tool`. Both
JSON and SSE replies carry this metadata. Preserve the original handoff history
prefix when submitting the remaining results. Do not add acknowledgements to it.
