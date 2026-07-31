# Operations

Failures hit while building and deploying this, indexed by **what you see** rather
than by component. Setup lives in [README.md](README.md); this is the file to grep
when something is already broken.

Commit messages carry the fuller reasoning behind each fix — `git log --grep=<term>`
is the deep archive. This file exists because that only works when you already know
the term, and a symptom is what you actually arrive with.

---

## Symptom → cause → fix

| Symptom | Cause | Fix |
| --- | --- | --- |
| vLLM returns 400 on any request carrying `tools` | Server started without tool-call parsing | `--enable-auto-tool-choice --tool-call-parser qwen3_xml`. Qwen3.5 emits XML tool calls; `hermes` expects JSON and never matches |
| Agent returns empty text and burns ~20 LLM calls per turn | Thinking is on; the model finishes its whole answer inside `<think>`, so `--reasoning-parser qwen3` routes everything to `reasoning` and leaves `content` empty | `ENABLE_THINKING=false` (default). Only bites with a system prompt + several tools |
| Tool runs, result is correct, model's follow-up is empty | qwen-agent labels the tool result `id`; the API keys off `tool_call_id`, so the model never sees it | `agent/compat.py` |
| Crash on first streaming tool-call delta (`arguments` is `None`) | vLLM sends name-only in the first delta; qwen-agent 0.0.34 passes it straight into a `str` field | `agent/compat.py` |
| "My tool isn't visible" — model answers from memory | More than ~18 tools declared. Open WebUI adds ~26 built-ins to yours; at 27 this model calls **nothing**, 3/3 runs | `AGENT_CLIENT_TOOLS=<substring>` |
| Log shows `bridged=[] declared=[...]` | Tool schema rejected at construction, not ignored by the model. `is_tool_schema` asserts the parameters block is *exactly* `{type, properties, required}` | `bridge.normalize_schema`. Open WebUI omits `required` when every arg has a default |
| Connector: "A server with this URL already exists" | The connector UI checks its own account records — **the request never reaches this server** (access log shows zero external hits) | Remove the stale connector in the *account*: claude.ai in a browser, and org-scoped connectors, which personal settings can't delete |
| Connector: "Couldn't connect… valid MCP server" | Non-443 port. Hosted connectors only reach remote MCP servers on 443 | Serve on 443; path-routed Funnel entries let it share the port |
| OAuth loops `/authorize → /consent` forever, never reaches `/token` | Consent redirect returned 302, which lets the browser repeat it as a POST; the callback only accepts GET | Return **303** |
| MCP 401s right after a successful token exchange | Token minted with no scope. Connectors request no `scope` at `/authorize`, so a non-empty `required_scopes` rejects every token | `required_scopes=[]`, and default the grant scope |
| Pre-registered OAuth client vanishes later | `register_client.py` and the running server share one state file; the server's next `save()` overwrote it | `Store.save()` merges on-disk state; provider re-reads on unknown client |
| vLLM won't start: "Free memory on device… less than desired GPU memory utilization" | `--gpu-memory-utilization` is a fraction of **total** VRAM but must be **free** at startup | Keep ≤ 0.92 here — see Hard limits |
| vLLM won't start: "estimated maximum model length is …" | Activation buffers scale with `--max-model-len`; raising it can cost more than fp8 saves | `--max-num-batched-tokens 2048` decouples the activation peak from context length |
| vLLM engine dies at startup: `No such file or directory: 'ninja'` | torch inductor shells out to `ninja`, which ships only inside the vLLM venv | Put `vllm-env/bin` on `PATH` in the unit |
| ipopt: "Solver did not exit normally" / `libgfortran.so.5` not found | IDAES has no Debian 13 build; the `ubuntu2204` one links libs Debian 13 didn't ship at the time | `apt install libgfortran5 liblapack3` — Debian ships both now. The old fix was a `.vendor/` tree unpacked from `.deb` and prepended to `LD_LIBRARY_PATH`; that is removed, and should not come back. Pointing the loader at a vendored libgfortran risks shadowing conda's with a mismatched build, which surfaces as a solver crash rather than a clear error |
| `127.0.0.1:8000` times out but `172.17.0.1:8000` works | Host firewall drops loopback to that port — connections **time out** rather than being refused | Point `VLLM_BASE_URL` at the reachable address, or fix the rule |
| RO simulation returns a plausible but wrong number | WaterTAP builds property vars on demand; one first touched *after* the solve is created unconstrained at its default | `_touch_reported_properties` builds everything reported before solving |
| `EngineDeadError` in the vLLM log | Usually **not** a crash — check for `EngineCore: trigger received signal=SIGTERM` just above it, which is a normal `systemctl restart` | Ignore if SIGTERM is present |
| Connectors stop reaching the MCP server right after a `tailscale serve` edit | `tailscale serve` on a Funnel-enabled port silently drops Funnel — status flips to "tailnet only" and nothing warns you | Re-issue every mapping with `tailscale funnel --bg --https=443 --set-path …`. Always `tailscale serve status` after editing |
| `tailscale funnel --https=443 on` → `non-localhost target "http://on"` | The `on`/`off` subcommands are gone as of 1.98; `on` is parsed as a proxy target | Enable Funnel by re-issuing the mapping under `tailscale funnel` instead of `tailscale serve` |
| `pip install reaktoro` → no matching distribution | Reaktoro is published on **conda-forge only**; there is no PyPI wheel at any version | Build the env with micromamba (`-c conda-forge python=3.13 reaktoro cyipopt`), then layer watertap/reaktoro-pse on with pip |
| MCP server fails to start after a fresh `pip install mcp` | Unpinned installs now resolve **mcp 2.0**, a major release. That package owns the OAuth provider, `AuthSettings`, transport security and `FastMCP` | Pin `mcp==1.28.1`. Treat 1→2 as its own migration with its own validation |
| Reaktoro pH output exactly equals the pH you passed in | Concentration was modelled by subtracting H2O from the composition, which holds the concentrate at the feed's fixed pH — an input echoed back as a result | Dose `H2O_evaporation` as a `chemistry_modifier` against the speciated feed instead, so pH floats. Validated against PHREEQC in `test_reaktoro_model.py` |
| Reaktoro numbers look plausible but are wrong in brine | reaktoro-pse defaults **every** phase to an ideal activity model and never warns | Pin `activity_model="ActivityModelPitzer"` explicitly. Not caller-selectable in this wrapper, deliberately |

---

## Hard limits (measured on this hardware)

| Thing | Value | Why it matters |
| --- | --- | --- |
| `--gpu-memory-utilization` ceiling | **0.921** | ~1.6 GiB is held by the display with *no compute process attached*; free is 14.66/15.92 GiB. Anything above aborts before the model loads |
| KV cache cost | 36.3 KiB/token fp16 → **~20 KiB fp8** | Only 8 of 32 layers hold KV (hybrid, `full_attention_interval: 4`), which is why 32k fits on 16 GB at all |
| KV pool | 0.73 GiB = **38,804 tokens** @ 32k ctx | Concurrency 1.18x. At 16k it was 30,895 tokens / 1.89x |
| Model weights | 11.53 GiB of 15.92 | KV is the *remainder*, so small utilization changes move it a lot |
| Model context ceiling | `max_position_embeddings` = **262144** | The limit here is VRAM, not the model |
| Declared tools | reliable ≤ **18**, zero at **27** | Measured 3/3 at each point |
| Tailscale Funnel ports | 443, 8443, 10000 | Hosted connectors only accept **443** |

---

## Dead ends — tried, does not work here

Recorded so they aren't retried. Each of these looked reasonable.

| Approach | Why it fails |
| --- | --- |
| `--gpu-memory-utilization 0.95` | Needs 15.12 GiB free; only 14.66 GiB is. The display reservation is not reclaimable |
| Raising `--max-model-len` on its own | Activation buffers grow with it. Doubling to 32k cost ~0.69 GiB — more than fp8 saved — and available KV *fell* to 0.38 GiB, under the 0.61 GiB one full request needs |
| **FP8 weights** to save VRAM | The model is already AWQ **4-bit**; FP8 is 8-bit and would roughly double the weight footprint. Only the *KV cache* dtype is worth changing |
| Port 8443 for hosted connectors | Reachable from a browser, ignored by the vendor backend. The tell: zero vendor-side requests in the access log while the browser succeeds |
| Exact-match `AGENT_CLIENT_TOOLS` | Open WebUI renames a tool to `{tool_id}_{name}` on collision with a built-in, so an exact allowlist drops the tool you meant to keep. Matching is substring |
| stdio MCP for a server on another host | stdio requires the server to be a local child process. Remote means streamable-http |
| `--tool-call-parser hermes` | Expects JSON; Qwen3.5's chat template emits XML `<tool_call><function=…>` |
| Static bearer token for hosted connectors | Their UIs authenticate over OAuth with dynamic registration; there is no field to paste a token into |
| A sample/placeholder file in `AGENT_FILES` | It is attached to *every* request and the model states it as fact. Point it only at documents you want asserted |
| **ROSSpy** for RO scaling chemistry | Last release Jun 2022, last commit May 2023, and it declares *no* dependencies at all, so `pip install` yields an unusable package. Needs IPHREEQC compiled from USGS source. Runs beside WaterTAP rather than inside it, so results cannot be optimized jointly. Reaktoro-PSE is maintained by `watertap-org`, installs as a binary, and is a Pyomo graybox |
| Subtracting H2O from the composition to concentrate a feed | Holds the concentrate at the feed's fixed pH, so reported pH is the input echoed back and carbonate scaling is understated. PHREEQC shows pH falling 7.800 → 7.585 over a 0–90% removal sweep |
| Two MCP processes sharing one `.oauth_state.json` | Both merge-on-save, so whichever writes last wins and registered clients silently disappear. During a cutover, stop the old service *before* copying the state file |

---

## Keeping this current

Add a row when something costs more than one debugging round. The bar is "would
future-you lose an hour rediscovering this" — not every fix belongs here.

Dead ends matter as much as fixes: an approach that looks obviously correct and
isn't will be retried otherwise. Record the *measurement* that killed it, not just
the conclusion.
