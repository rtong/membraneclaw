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
| `tests/fixtures/` | Synthetic fixtures for the tests — **not** documentation |
| [`OPERATIONS.md`](OPERATIONS.md) | Symptom-indexed failure log, measured limits, and dead ends |

## Running

The vLLM backend runs 32k context on a 16 GB card via `--kv-cache-dtype fp8`
(~20 KiB/token instead of ~36) plus `--max-num-batched-tokens 2048`, which
stops the activation peak scaling with context length — see
`deploy/vllm-qwen.service` for why each flag is there.

```bash
./serve_vllm.sh                 # backend; takes ~2 min to load
.venv/bin/python cli.py         # interactive chat
.venv/bin/python cli.py -p "What is 17^4?"
.venv/bin/python cli.py -f /path/to/your/doc.md -p "What does it say about X?"
.venv/bin/python smoke_test.py  # verify everything
```

Credentials live in `.env` (gitignored); see `.env.example`.

**Something already broken?** [`OPERATIONS.md`](OPERATIONS.md) is indexed by
symptom, and records the approaches that turned out not to work here.

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

## RO simulation and scaling chemistry (MCP)

`mcp_watertap/` is an MCP server exposing five tools from two engines:

| Tool | Engine | Answers |
| --- | --- | --- |
| `describe_ro_parameters` | — | What `simulate_ro` accepts |
| `simulate_ro` | WaterTAP [ReverseOsmosis0D](https://watertap.readthedocs.io/en/stable/technical_reference/unit_models/reverse_osmosis_0D.html) | Flux, recovery, rejection, polarization |
| `describe_reaktoro_options` | — | Valid ions, minerals, reagents |
| `equilibrate_feed` | [Reaktoro-PSE](https://github.com/watertap-org/reaktoro-pse) | What precipitates from a given water |
| `analyze_ro_scaling` | both | Solves the module, then what scales in *its* concentrate |

The agent picks the server up automatically (`RO_MCP=off` to disable), so it works
from the CLI, the HTTP front end, and Open WebUI without extra wiring.

Both engines live in **one process** because they must: `analyze_ro_scaling`
attaches a `ReaktoroBlock` to a Pyomo model built from the RO result, and a Pyomo
graybox cannot cross a process boundary. The agent still reaches all of it over
MCP rather than importing it, so the heavy Pyomo/IDAES/Reaktoro stack never
touches the agent's dependencies. Tools appear to the model as
`watertap-ro-simulate_ro` and so on.

The environment is **conda, not a venv** — Reaktoro is published on conda-forge
only and has no PyPI wheel:

```bash
micromamba create -p ~/reaktoro-mcp/env -c conda-forge python=3.13 reaktoro cyipopt
~/reaktoro-mcp/env/bin/pip install -r mcp_watertap/requirements.txt
```

Set `RO_MCP_URL` to use a remote server (streamable-http, with `MCP_BEARER_TOKEN`
sent as a bearer header); leave it unset to spawn the local one over stdio. The
deployed setup points anton's agent at temur — see below.

Sanity check — defaults are 1 kg/s of ~35 g/kg seawater at 50 bar over 50 m²:

```
feed osmotic 28.5 bar · flux 16.4 LMH · recovery 23.4% · permeate 349 ppm · rejection 99.03%
seawater at 80% recovery · Calcite ST 3.83 · Gypsum ST 1.44 · pH 6.72 · both at risk
```

**The two engines will not agree on osmotic pressure for the same water** — 28.5
bar from `simulate_ro`, 24.5 from `equilibrate_feed`. That is expected:
`simulate_ro`'s property package treats all dissolved solids as NaCl, which
carries more osmoles per gram than the Mg/Ca/SO₄ salts in a real analysis. Quote
each number against its own tool. `analyze_ro_scaling` likewise reports a
slightly higher flux than `simulate_ro`'s default, because it derives salinity
from the ion list (34.7 g/kg) rather than assuming 35 g/kg NaCl.

### Validation

`test_reaktoro_model.py` checks the chemistry against `phreeqc_data.json`, which
ships with reaktoro-pse and was generated by **PHREEQC** — a different code, so
this is a cross-check rather than a regression pin. Across a 0–90% water-removal
sweep: calcite scaling tendency within 0.38% while spanning 6.9 → 198, pH within
0.030 units, osmotic pressure within 0.06%.

```bash
~/reaktoro-mcp/env/bin/python mcp_watertap/test_reaktoro_model.py
~/reaktoro-mcp/env/bin/python mcp_watertap/test_ro_model.py
```

### Using it from the ChatGPT desktop app

The desktop app, Codex CLI and IDE extension share one MCP config at
`~/.codex/config.toml`. (This is a different surface from Settings → Connectors →
Developer mode, which is web/Windows only and cannot reach localhost.)

```toml
# ~/.codex/config.toml
[mcp_servers.watertap-ro]
url = "https://temur.tail35bed8.ts.net/mcp"
bearer_token_env_var = "WATERTAP_MCP_TOKEN"
```

with `export WATERTAP_MCP_TOKEN=…` matching `MCP_BEARER_TOKEN` in `.env`.

The MCP server runs on **temur** (`192.168.86.42`, tailnet `100.64.77.85`) — the RO
solve is CPU-only and does not belong on the GPU host. It binds loopback there and
Tailscale terminates TLS.

**Port 443, not 8443.** Hosted connectors (Claude, ChatGPT web) only reach remote
MCP servers on 443; a non-standard port fails with a generic "couldn't connect",
and the giveaway is that *nothing* from the vendor's backend appears in the access
log while a browser reaches the URL fine. temur's 443 already Funnels a second
vLLM at `/`, so the MCP endpoint shares the port through path-routed Funnel
entries — vLLM serves only `/v1/*`, so nothing collides:

```bash
# on temur
sudo systemctl enable --now reaktoro-mcp    # 127.0.0.1:8003
for p in /mcp /authorize /token /register /revoke /consent \
         /.well-known/oauth-authorization-server /.well-known/oauth-protected-resource; do
  tailscale funnel --bg --https=443 --set-path "$p" "http://127.0.0.1:8003$p"
done
tailscale serve status   # confirm it still says "Funnel on"
```

`MCP_PUBLIC_URL` must then be the port-less host, or discovery advertises endpoints
on a port the connector will not use.

Use `tailscale funnel` for **every** edit, never `tailscale serve`. `serve` writes
the same mapping but silently drops Funnel to tailnet-only, which takes the hosted
connectors offline with no error anywhere — hence the `status` check above. The
`on`/`off` subcommands that used to re-enable it were removed in Tailscale 1.98;
re-issuing the mapping under `funnel` is now the only way back.

**While Funnel is on, `MCP_BEARER_TOKEN` is the only thing between the internet and
the solver.** The tools cannot read files or run shell commands — the blast radius
is CPU, not data — but there is **no solve timeout and no rate limiting**, so a
leaked token means someone can pin temur's cores with pathological inputs. Turn
Funnel off when you are not using it.

### Two ways to authenticate

| Client | Credential |
| --- | --- |
| Agent, CLI, ChatGPT desktop (`config.toml`) | `MCP_BEARER_TOKEN` as a bearer header |
| Claude / ChatGPT hosted connectors | OAuth 2.1 + `MCP_OAUTH_APPROVAL_KEY` |

Hosted connector UIs authenticate MCP servers over **OAuth with dynamic client
registration** — there is no field to paste a static token into, so a
token-only endpoint answers them `401` no matter what is configured. That is the
failure to look for: repeated `401`s from a public IP in `mcp.log` mean a
connector is registered and trying, not that the URL is wrong.

`mcp_watertap/oauth.py` implements the authorization server (`/register`,
`/authorize`, `/token`, `/revoke`, plus both discovery documents). There is no
user directory behind it, so consent is a single shared **approval key**: any
client may register itself, but only someone holding `MCP_OAUTH_APPROVAL_KEY`
can approve one at the `/consent` page. Enable it by setting that key together
with `MCP_PUBLIC_URL` — the issuer must be the externally reachable URL, or the
discovery documents advertise endpoints the client cannot reach. The static
bearer keeps working alongside it.

In Claude, the Client ID / Client Secret fields sit under **Advanced settings** and
are optional — left blank, Claude registers itself dynamically. When a UI insists
on values, mint a pair on the server:

```bash
.venv/bin/python register_client.py claude          # create
.venv/bin/python register_client.py claude --show   # print later
```

OAuth 2.1 requires **exact** redirect-URI matching, so the client is registered
with Claude's callbacks (`https://claude.ai/api/mcp/auth_callback`,
`https://claude.com/api/mcp/auth_callback`) and the Claude Code CLI's fixed local
port; anything else is rejected at `/authorize`. Pass `--redirect-uri` to add more.

The approval key is **not** entered in Claude — it is entered once on the
`/consent` page that opens in the browser during the connect flow.

### "A server with this URL already exists"

Raised by the connector UI against its own account records, **before any
request reaches this server** — the access log shows no external hit while the
error recurs, so nothing configured here affects it. The stale record lives in
the Claude *account*, not the desktop app: check claude.ai → Settings →
Connectors in a browser (desktop and web share account-level connectors), and
check org-scoped connectors, which personal settings cannot remove.

## Using it from BetterGPT (or any OpenAI client)

BetterGPT talks straight to an OpenAI-compatible endpoint. Pointed at vLLM
(`:8000`) it bypasses this project entirely — you get the raw model, no tools and
no RAG. `server.py` exposes the *agent* over the same protocol, so point BetterGPT
at it instead:

```bash
AGENT_FILES=/path/to/your/docs.md .venv/bin/python server.py   # listens on :8001
```

In BetterGPT → Settings → API: set the endpoint to `http://127.0.0.1:8001/v1/chat/completions`.
The API key is ignored unless you set `AGENT_API_KEY`.

| Env var | Default | Meaning |
| --- | --- | --- |
| `AGENT_PORT` / `AGENT_HOST` | `8001` / `127.0.0.1` | Bind address |
| `AGENT_API_KEY` | *(unset)* | If set, required as `Bearer` token |
| `AGENT_FILES` | *(unset)* | Real documents attached to every request (RAG). Leave empty unless you have documents to ground on — a placeholder here gets stated as fact. |
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

**Point it only at documents you want stated as fact.** Whatever is listed is
attached to *every* request and the model grounds its answers in it — a
placeholder or sample file makes the agent confidently assert invented details.
Leave it empty when you have nothing real to ground on; RAG over fabricated
content is worse than no RAG.

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
