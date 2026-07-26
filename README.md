# MembraneClaw

Agent harness over a local Qwen3.5-9B-AWQ deployment on vLLM, built on
[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent). Supports custom tool calling,
RAG over local documents, and multi-turn chat.

## Layout

| Path | Purpose |
| --- | --- |
| `serve_vllm.sh` | Starts the vLLM backend with tool calling enabled |
| `agent/config.py` | Endpoint, model, and generation settings |
| `agent/tools/builtin.py` | Tool definitions — add your own here |
| `agent/core.py` | Builds the `Assistant` |
| `agent/compat.py` | Shims for qwen-agent 0.0.34 vs. vLLM 0.25.1 |
| `cli.py` | Interactive / one-shot CLI |
| `server.py` | OpenAI-compatible HTTP front end for chat UIs |
| `smoke_test.py` | End-to-end check of the backend and agent |

## Running

```bash
./serve_vllm.sh                 # backend; takes ~2 min to load
.venv/bin/python cli.py         # interactive chat
.venv/bin/python cli.py -p "What is 17^4?"
.venv/bin/python cli.py -f docs/sample.md -p "What port does the service use?"
.venv/bin/python smoke_test.py  # verify everything
```

Credentials live in `.env` (gitignored); see `.env.example`.

## Open WebUI

```bash
./run_openwebui.sh                                       # http://localhost:3000
tailscale funnel --bg --https=443 http://127.0.0.1:3000  # publish the UI
```

Open WebUI calls the agent from its **backend**, not the browser, which is why the
agent can stay on loopback: `--network=host` lets the container reach
`127.0.0.1:8001` without the agent listening on any routable interface. Only the UI
is published; the agent API is no longer internet-facing. None of the browser-side
constraints (CORS, mixed content) apply on this path.

Signup is disabled (`ENABLE_SIGNUP=false`) — Open WebUI grants **admin to the first
account that registers**, and Funnel hostnames show up in certificate-transparency
logs, so leaving registration open on a public URL is a real hijack risk. Create
users from the admin panel. To re-open registration temporarily,
`ENABLE_SIGNUP=true ./run_openwebui.sh`.

Generation settings in the Open WebUI sidebar (temperature, max tokens) are **not**
applied: `server.py` runs the request through Qwen-Agent, which uses the values in
`agent/config.py`. Change them there.

### Open WebUI tools

Tools defined in Open WebUI work, but only with **Function Calling: Native** on the
model (Default mode has Open WebUI do its own tool selection, which the agent loop
then wraps and breaks). `agent/bridge.py` handles the round trip: client tools are
declared to the agent as stubs, and when the agent picks one the run is abandoned
and the call is returned as OpenAI `tool_calls`. Open WebUI executes it and replays
the conversation with a `tool` message, which is converted back so the agent resumes.
The agent's own tools (`calculator`, …) still execute in-process in the same turn.

**Cap how many tools reach the model.** Open WebUI declares ~26 built-in tools
(knowledge, notes, chats, tasks, automations, calendar) *in addition* to yours. This
model reliably calls a tool with ≤18 declared, and reliably calls **nothing** at 27 —
it answers from memory instead, which looks exactly like "my tool is not visible".
Restrict what gets bridged:

```bash
AGENT_CLIENT_TOOLS=get_current_weather   # comma-separated; empty or "*" = all
```

Matching is case-insensitive **substring**, not exact: Open WebUI renames a tool to
`{tool_id}_{name}` when it collides with a built-in, so an exact allowlist would
drop the tool you meant to keep.

Each request logs `client_tools=<declared> bridged=<names> declared=<names>`, warns
past `AGENT_TOOL_COUNT_WARN` (default 18), and logs a reason for every tool it
skips. If `bridged` is empty while `declared` is not, the tools were rejected — not
ignored by the model.

**Schema normalization.** `qwen_agent.tools.base.is_tool_schema` asserts the
parameters block is *exactly* `{type, properties, required}`. Open WebUI omits
`required` when every argument has a default, and Pydantic-derived schemas add
`title`/`additionalProperties`/`$defs`; both are valid JSON Schema and both fail
that equality check. `bridge.normalize_schema` rebuilds the block with just those
three keys (and drops `required` entries that aren't in `properties`), so client
tools are not silently rejected at construction.

Pick the container name with `CONTAINER_NAME`, not `NAME` — `NAME` is frequently
already set in the shell and silently wins.

## Using it from BetterGPT (or any OpenAI client)

BetterGPT talks straight to an OpenAI-compatible endpoint. Pointed at vLLM
(`:8000`) it bypasses this project entirely — you get the raw model, no tools and
no RAG. `server.py` exposes the *agent* over the same protocol, so point BetterGPT
at it instead:

```bash
AGENT_FILES=docs/sample.md .venv/bin/python server.py   # listens on :8001
```

In BetterGPT → Settings → API: set the endpoint to `http://127.0.0.1:8001/v1/chat/completions`.
The API key is ignored unless you set `AGENT_API_KEY`.

| Env var | Default | Meaning |
| --- | --- | --- |
| `AGENT_PORT` / `AGENT_HOST` | `8001` / `127.0.0.1` | Bind address |
| `AGENT_API_KEY` | *(unset)* | If set, required as `Bearer` token |
| `AGENT_FILES` | *(unset)* | Comma-separated docs attached to every request (RAG) |
| `AGENT_MODEL_ID` | `qwen3.5-9b-agent` | Id advertised by `/v1/models` |

Tool calls are streamed into the reply as `` `🔧 name(args)` `` / `` `↳ result` ``
lines, so you can see the agent working in the chat window — that is how you tell
a real agent turn from a plain model turn.

Two things this front end needs that a plain proxy does not: CORS (the browser
calls it directly from `https://bettergpt.chat`) and the
`Access-Control-Allow-Private-Network` header, without which Chrome's Private
Network Access check blocks an HTTPS page from reaching a loopback server.

Because a chat UI cannot upload files to us, RAG is wired through `AGENT_FILES`
rather than per-message attachments; the CLI's `-f` flag is still the per-run path.

## Exposing it (Tailscale)

```bash
.venv/bin/python server.py                               # binds 127.0.0.1:8001
tailscale funnel --bg --https=443 http://127.0.0.1:8001  # public internet
tailscale funnel --https=443 off                         # tear down
```

Endpoint: `https://anton.tail35bed8.ts.net/v1/chat/completions`, with
`AGENT_API_KEY` as the bearer token.

**Funnel vs. serve.** `tailscale serve` is tailnet-only; `tailscale funnel` is
public. Only Funnel works for a browser that is not itself on the tailnet: public
DNS points `anton.tail35bed8.ts.net` at Tailscale's ingress
(`208.111.34.x`), and with Funnel off that ingress drops the connection —
which surfaces in the browser as `ERR_CONNECTION_CLOSED`, not a DNS error.
Both commands claim port 443, so each silently replaces the other.

Do not use a raw tailnet IP. BetterGPT is an HTTPS page and browsers only exempt
`localhost`/`127.0.0.1` from mixed-content blocking, so `http://100.x.y.z:8001`
is blocked outright. Funnel and serve both terminate real TLS on the `ts.net` name.

Keep the server bound to `127.0.0.1`. Tailscale proxies from loopback, so the port
is unreachable on the tailnet IP and the proxy is the only way in.

### Security posture while Funnel is on

`AGENT_API_KEY` is the *only* thing between the open internet and this endpoint,
and the `http_get` tool will fetch any URL the model picks — including your LAN and
tailnet addresses — and return the body. Anyone holding the key can read internal
HTTP endpoints through this machine and consume GPU time; there is no rate limiting.
Treat the key as a production secret, rotate it in `.env` if it leaks, and turn
Funnel off when you are not using it. `/v1/models` is intentionally unauthenticated
so clients can enumerate; it exposes only the model id.

## Adding a tool

Add a `BaseTool` subclass to `agent/tools/builtin.py` and list its name in
`DEFAULT_TOOLS` in `agent/core.py`. `parameters` must be a **JSON Schema dict**,
not qwen-agent's list form — the list form is not valid JSON Schema and vLLM
rejects it with a 400 when tools are passed natively.

```python
@register_tool("lookup_order")
class LookupOrder(BaseTool):
    description = "Look up an order by ID."
    parameters = {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "Order ID"}},
        "required": ["order_id"],
    }

    def call(self, params: str, **kwargs) -> str:
        return json.dumps(fetch(json5.loads(params)["order_id"]))
```

## Deployment notes

Things that were needed to make this stack work together:

- **`--enable-auto-tool-choice --tool-call-parser qwen3_xml`.** Without these the
  server rejects any request carrying `tools`. Qwen3.5 emits XML tool calls
  (`<tool_call><function=...><parameter=...>`), so `qwen3_xml` is correct here —
  `hermes` expects JSON and will not match.
- **Thinking is disabled by default** (`ENABLE_THINKING=false`). With thinking on,
  the model frequently completes its entire answer inside the `<think>` block and
  emits nothing afterwards; `--reasoning-parser qwen3` then routes all of it to
  `reasoning`, leaving `content` empty and stalling the agent loop until it burns
  through its call budget.
- **`agent/compat.py`** patches two qwen-agent 0.0.34 bugs against vLLM: streaming
  tool-call deltas arrive with `name`/`arguments` as `None`, and tool results are
  labelled `id` instead of the `tool_call_id` the API requires.
- `ninja` ships only inside `vllm-env/bin`, so `serve_vllm.sh` puts it on `PATH`;
  torch inductor shells out to it during startup.

vLLM 0.25.1 returns reasoning text in a field named `reasoning`, while qwen-agent
reads `reasoning_content` — so reasoning text is dropped rather than displayed.
That is harmless while thinking is off.
