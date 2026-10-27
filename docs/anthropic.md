# Anthropic Messages provider

Set `ANTHROPIC_API_KEY`, then run `membraneclaw --provider anthropic --model claude-sonnet-4-20250514 --no-mcp -p "Calculate 2+2"`. This explicit compatibility profile is not a claim about the latest available model. Other models fail before network I/O. The REST transport fixes `anthropic-version: 2023-06-01`; no SDK, implicit retries, redirects or tool runner is involved.

Every assistant content block is preserved in native history, including thinking signatures, redacted thinking and unknown block fields. Tool results follow tool_use order even if submitted in reverse order, and preserve is_error. Hosted tool configuration is rejected; unknown server tool blocks cannot trigger local execution or successful completion. Missing usage stays unknown. Inclusive normalized input tokens explicitly add input_tokens + cache_read_input_tokens + cache_creation_input_tokens; the raw breakdown remains available.

Optional `anthropic.thinking` supports enabled/budget_tokens with 1024 <= budget < max_output_tokens and no explicit temperature. Tool choice supports auto/any/tool/none with validated names; forced tools with thinking are rejected. Defaults: max_tokens=4096, no thinking. The gateway locks route/options for an episode.

Tests: `python -m unittest discover -s tests/providers/contract -v`. Contracts cover two tool rounds, signature/order preservation, redacted data, errors, stop mapping, cache usage, duplicate IDs, unsupported options and loopback 429/malformed response capture. Live Claude smoke remains unverified without credentials/services. Disable the anthropic route to roll back; preserve old traces and never silently switch pending episodes to another provider.

Official protocol sources: [Messages](https://platform.claude.com/docs/en/api/messages/create), [streaming and thinking signatures](https://platform.claude.com/docs/en/build-with-claude/streaming).
