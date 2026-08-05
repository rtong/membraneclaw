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
| `tailscale funnel --https=443 on` → `non-localhost target "http://on"` | `on` has no meaning without an existing mapping to revive — there's no "last state" to turn back on — so it's parsed as a proxy target instead. `off` is fine: `tailscale funnel --https=<port> off` or `--https=<port> --set-path <path> off` removes exactly that mapping, no `reset` needed | Enable Funnel by re-issuing the mapping under `tailscale funnel` instead of `tailscale serve`. To remove one mapping, use `off` with the same `--set-path`/`--https` that created it |
| OAuth refresh tokens accumulate forever, deleting one doesn't stick | `Store.save()` in `oauth.py` merged refresh tokens as a union of memory and disk. `revoke_token`/`exchange_refresh_token` pop a token from memory, but the next `save()` re-added it from disk. Grew 34 → 61 tokens in two days with nothing ever actually leaving | Write `refresh` from memory only, not merged (`clients` still merges — `register_client.py` writes that concurrently). Added `REFRESH_TTL` so new tokens expire and old ones get backfilled once. Regression test: `test_oauth_store.py` |
| `pip install reaktoro` → no matching distribution | Reaktoro is published on **conda-forge only**; there is no PyPI wheel at any version | Build the env with micromamba (`-c conda-forge python=3.13 reaktoro cyipopt`), then layer watertap/reaktoro-pse on with pip |
| MCP server fails to start after a fresh `pip install mcp` | Unpinned installs now resolve **mcp 2.0**, a major release. That package owns the OAuth provider, `AuthSettings`, transport security and `FastMCP` | Pin `mcp==1.28.1`. Treat 1→2 as its own migration with its own validation |
| Reaktoro pH output exactly equals the pH you passed in | Concentration was modelled by subtracting H2O from the composition, which holds the concentrate at the feed's fixed pH — an input echoed back as a result | Dose `H2O_evaporation` as a `chemistry_modifier` against the speciated feed instead, so pH floats. Validated against PHREEQC in `test_reaktoro_model.py` |
| A solve time limit has no effect | The limit was applied only to the final `solve()`. `initialize()` runs its own solves and does most of the work, so by the time the capped solve runs the model is already converged and finishes in ~0 iterations | Pass the same options to `initialize(optarg=…)`. Verify with `max_iter=0`: it must fail *during initialize* |
| `AttributeError: '_ScalarFlowsheetBlock' object has no attribute 'costing'` from the SWRO flowsheet | WaterTAP **1.7.0 has two costing packages** — `m.fs.zo_costing` and `m.fs.ro_costing` — aggregated into Expressions on the *model*: `m.LCOW`, `m.SEC`, `m.total_capital_cost`, `m.total_operating_cost` (USD_2018). Later versions collapse this to a single `m.fs.costing` with `.LCOW`/`.specific_energy_consumption`. GitHub `main` shows the *newer* shape, so coding from it against 1.7.0 fails | `swro_model._costing_metrics` handles both. Costing is initialized via `initialize_costing(m)` in 1.7.0, not `m.fs.costing.initialize()` |
| Writing a WaterTAP flowsheet wrapper from the docs or GitHub and it doesn't match the installed package | The `seawater_RO_desalination` reference page is stale in two ways: it cites the old module path (`watertap.examples.flowsheets.case_studies…`; the real one is `watertap.flowsheets.seawater_RO_desalination.seawater_RO_desalination`) and documents helpers `build_flowsheet()`/`solve_flowsheet()` that **do not exist** in any installed version checked | Read the *installed* source, not the docs or `main`: `~/reaktoro-mcp/env/lib/python3.13/site-packages/watertap/flowsheets/...`. Validate any wrapper by comparing to the flowsheet's own `main()` — it should match to zero delta |
| SWRO results look plausible but describe the wrong-sized plant | The flowsheet sizes membrane area from feed flow (`RO.area = flow_vol * 4.5e4`). Overriding `feed_flow_m3_s` without also setting `ro_area_m2` leaves the area fixed at the *default* flow's value, quietly simulating a mis-sized plant | `swro_model` derives area from flow whenever `ro_area_m2` is unset. Pass area explicitly only when you mean to decouple the two |
| Reaktoro numbers look plausible but are wrong in brine | reaktoro-pse defaults **every** phase to an ideal activity model and never warns | Pin `activity_model="ActivityModelPitzer"` explicitly. Not caller-selectable in this wrapper, deliberately |
| Google Drive/Sheets tools stop being called; model answers from memory instead | Same ≤18-tool cliff as Open WebUI, just from `GOOGLE_MCP_TOOLS=*` (or too many `GOOGLE_MCP_SERVERS`) declaring all 22+ Drive+Sheets tools on top of the builtins and ro-chem | Leave `GOOGLE_MCP_TOOLS` at its default allowlist (15 declared total); check the `declaring N tools to the model` startup log |
| Google MCP tool calls 401 partway through a long-running process | Access token expired (1hr) and the refreshing `httpx.Auth` never got attached — usually because `agent/compat.py`'s `streamablehttp_client` patch wasn't applied, or the URL host isn't in `agent/compat.py:_GOOGLE_MCP_HOSTS` | Confirm `compat.apply()` runs before any MCP connection (it's called at `agent/core.py` import time); `.venv/bin/python -m agent.google_oauth status` shows whether the *stored* token is fresh, independent of whether the shim ran |
| `redirect_uri_mismatch` during `python -m agent.google_oauth login` | Web-application OAuth clients match the redirect URI **exactly**, port included — unlike desktop/loopback clients, Google does not exempt them from port-matching | Redirect URI registered in Cloud Console must equal `http://localhost:$GOOGLE_OAUTH_PORT/oauth2callback` exactly; keep both in sync if you change the port |
| `google_oauth.py login` succeeds but Google returns no `refresh_token`, and every later run demands `login` again | Google only issues a refresh token on a **fresh** consent grant; a second `/authorize` for an already-authorized app returns just an access token unless the request forces re-consent | `login()` always sends `access_type=offline&prompt=consent` — if this still happens, the app's grant was revoked mid-flow; revoke it explicitly at `myaccount.google.com/permissions` and retry |

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
| `max_cpu_time` alone as a solve bound | ipopt only tests it *between* iterations, so one pathological iteration overruns it. Pair it with `max_iter` — the first bounds duration, the second bounds count |
| Proving a solve timeout works by setting it tiny | Does not prove anything here: `initialize()` pre-solves the model, so the capped solve converges immediately and returns `optimal` no matter how small the limit. Test option *propagation* with `max_iter=0` instead |
| Rate limiting the MCP server as HTTP middleware | streamable-http holds long-lived SSE connections open. A request-level concurrency cap counts those against the limit and deadlocks the transport. Guard the tool functions instead |
| Two MCP processes sharing one `.oauth_state.json` | Both merge-on-save, so whichever writes last wins and registered clients silently disappear. During a cutover, stop the old service *before* copying the state file |
| Model calls `http_get` on a Google Docs/Sheets URL and gets `HTTP Error 401` | No Google tools were declared for that request, so the model reached for the only URL-shaped tool it had. Means the request had no verified user (Google tools are gated on identity), or `GOOGLE_TOOLS` filtered them all out | Check the request logs `user=<email>` rather than `user=-`. The Google tools accept URLs directly, so once declared the model uses them instead |
| `redirect_uri_mismatch` when a user clicks the in-chat login link | `GOOGLE_OAUTH_PUBLIC_REDIRECT` and the URI registered on the OAuth client differ. They must match verbatim — scheme, host, path, no trailing slash | Register the exact value of `GOOGLE_OAUTH_PUBLIC_REDIRECT`; it is echoed in the login URL's `redirect_uri=` parameter, so compare that against the console |
| Login link returns "Link expired" immediately | The `state` is single-use and 10 minutes long, and the in-memory pending record dies with the process. A `server.py` restart between issuing and clicking invalidates it, as does clicking twice | Ask the assistant again for a fresh link. Both properties are deliberate — replaying a callback must not re-authorize |
| Everything under the public host reaches the agent, not just `/google` | The Funnel mapping for `/` was replaced instead of added to. `--set-path` adds one handler; re-issuing without the `/` mapping drops it | Re-issue both: `/` → `127.0.0.1:3000` and `--set-path=/google` → `127.0.0.1:8001/google`, then verify `curl https://<host>/health` returns Open WebUI's `{"status":true}`, not the agent's JSON |
| Deleted a user's `.google_tokens/<id>.json` by hand, but the app still shows as authorized in the user's Google account | Deleting the file only stops local use; it never told Google anything, so the grant is still live at `myaccount.google.com/permissions` | Use `google_disconnect` (chat) or `POST /google/disconnect` instead — both call Google's revoke endpoint before deleting the file |
| Every request logs `user=-` and per-user Google tools never appear | Open WebUI isn't signing identities: `ENABLE_FORWARD_USER_INFO_HEADERS` unset/false, or `FORWARD_USER_INFO_HEADER_JWT_SECRET` differs between the container and `.env`. A mismatched secret logs `rejected forwarded identity: Signature verification failed`; a missing header logs nothing but the warning | Set both on the container (`run_openwebui.sh` passes the secret through) and keep `.env` in sync. Chat still works throughout — identity failure removes credentials, it does not break the turn |
| **Google's official Workspace MCP servers** (`drivemcp.googleapis.com`, `sheetsmcp.googleapis.com`, …) with a personal `@gmail.com` account | Every tool call — any operation, any service, any arguments — returns `"The caller does not have permission"` (a 200 at the JSON-RPC layer, `isError: true`). Confirmed not a scope/file/token problem: the *same* access token against the plain REST APIs (`sheets.googleapis.com/v4/...`) works fully for the same file. The MCP wrapper endpoints are gated separately, behind the Workspace Developer Preview Program, which enrolls at the **org** level via a Workspace Admin Console — a personal account has no such console and cannot enroll. Only a Google Workspace (paid/custom-domain) account can currently use these endpoints at all |

---

## Keeping this current

Add a row when something costs more than one debugging round. The bar is "would
future-you lose an hour rediscovering this" — not every fix belongs here.

Dead ends matter as much as fixes: an approach that looks obviously correct and
isn't will be retried otherwise. Record the *measurement* that killed it, not just
the conclusion.
