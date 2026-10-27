# OpenAI Responses provider

Use `membraneclaw --provider openai --model o4-mini-2025-04-16 --no-mcp -p "Calculate 2+2"` with `OPENAI_API_KEY` set. The other reviewed profile is `gpt-4.1-2025-04-14`; these are explicit compatibility profiles, not a latest-model recommendation. Unsupported models/options fail before a request. Qwen remains the default.

The adapter uses the Responses REST API through the same byte-capturing HTTP transport as Gemini. There is no SDK, automatic retry, redirect, or automatic tool execution. Gateway retries remain separately recorded attempts. `store=false` and explicit local input/output history avoid relying on an unrecorded remote conversation. All output items, unknown fields, call IDs, and reasoning encrypted content remain in private native state. Function schemas are sent with `strict=false`; local JSON Schema validation enforces constraints without rewriting optional properties. Tools unsupported by this implementation (including hosted tools) are rejected.

`openai.reasoning` accepts effort and summary for the reviewed o4 profile; temperature on that profile is rejected. Refusal maps to blocked, incomplete remains incomplete, and unknown item kinds cannot complete a successful tool round. Usage breakdowns retain their original meanings; cached/reasoning counts are not added again.

Offline tests cover two tool rounds, reverse result ordering, opaque reasoning, malformed arguments/body, refusal/incomplete, missing usage, invalid configuration, real loopback 429→success, and timeout unknown. Run `python -m unittest discover -s tests/providers/contract -v`. Live OpenAI smoke is not run without credentials; fixture success does not demonstrate account/model availability.

Rollback: disable the openai route for new episodes; preserve existing traces and do not change providers inside an episode.

Protocol references: [Responses reasoning](https://developers.openai.com/api/docs/guides/reasoning), [stream events](https://developers.openai.com/api/reference/resources/responses/streaming-events). Raw REST is intentional: exact request/response capture precedes parsing, and there are no implicit SDK attempts to hide.
