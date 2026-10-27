# Native evaluation tracing

Install the runtime first (`pip install '.[tools,http]'`), then `pip install -e experiments/auto-evaluate`. Native solver/router/recovery calls opt in with generation.provider or generation.routes. An example generation object is:

```json
{"provider":"gemini","trace_dir":"./var/traces/evaluation","dataset_split":"test","no_mcp":true,"max_tokens":2048,"temperature":0.0}
```

Existing system definitions select model IDs; routes can override provider/model/provider_options per role. Native top_p/enable_thinking overrides are rejected; use provider_options. Router translation explicitly removes the legacy enable_thinking=false hint. Native RAG attachments are not implemented: do not label tool-free runs as RAG experiments.

Teacher/judge automation can opt in with `python -m auto_evaluate auto --native-provider openai --codex-model gpt-4.1-2025-04-14` plus existing run/benchmark flags. This branch uses the traced API client before Codex selection; default account-based Codex behavior is unchanged. Judge/router/recovery/tool-free teacher calls have no tools. Tool teachers use configured MCP tools and still pass existing output/tool policy validation.

Each native call records role, context, dataset split and episode ID. Validation repair and context recovery create child episodes; the original failed record remains. Recovery records excerpt hash, truncation rule and parent ID. Native responses include a mode-0600 private portable bundle and its locator; copy that bundle when moving results to another host. Storage failure cannot promise an export.

Canonical trajectory extraction retains full tool results and canonical event IDs. Legacy text remains readable with legacy/inferred provenance. Judge trimming lists omitted IDs and its first-N rule without changing raw data. Native evaluator output records provider/model/prompt hash/version/episode and supplied evidence IDs. This is provenance, not proof that a score is correct.

Policy reuse records its source episode and zero new model requests. Completeness reporting separates execution failure, incomplete recording and untraced legacy data. Dataset split defaults to unspecified and must be set for research use; no automatic training-data publication occurs.

Run `python -m unittest discover -s experiments/auto-evaluate/tests -v` and the runtime suite. Live model, WaterTAP and Open WebUI tests remain environment-dependent. Roll back by removing provider/routes/--native-provider; retain existing traces.
