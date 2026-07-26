# Migration plan: Qwen3.5-9B on RTX 5070 Ti (WSL2 + Tailscale)

Full rebuild plan for reproducing the `anton` deployment from scratch on a
brand new Windows machine with the same GPU class (RTX 50-series/Blackwell).
This is the infra/OS-level bootstrap; the actual model-serving scripts live
in the `llm` repo (cloned in Phase 4) and are intentionally not duplicated
here — this plan gets you to the point where that repo's `qwen-rtx/setup.sh`
just works.

Repo with the deployment scripts: `git@gitlab.com:alcuka/ai/llm.git`
(private GitLab repo — separate from this backup, which lives on GitHub).

**Fast path**: `scripts/deploy.sh` runs Phases 1-5 for you (skipping whatever
it detects is already done) and pauses at the two points that genuinely
can't be automated from inside WSL — the Windows-side steps in Phase 0, and
the elevated PowerShell firewall rules in Phase 3. Run it interactively,
inside WSL, on the new machine: `./scripts/deploy.sh` (tailnet-only) or
`./scripts/deploy.sh funnel` (public internet). The phase-by-phase writeup
below is the reference for *why* each step exists and what to do if a phase
doesn't just work.

## Phase 0 — Windows-side driver + WSL

1. Install the normal NVIDIA driver **on Windows** (Game Ready or Studio —
   whatever matches the card). Do **not** install a separate Linux/WSL NVIDIA
   driver inside WSL — that breaks GPU passthrough. Windows' driver is what
   WSL2 uses automatically.
2. WSL2, recent version: from an elevated PowerShell, `wsl --update`, then
   `wsl --shutdown`.
3. Copy `.wslconfig.example` (in this dir) to `%UserProfile%\.wslconfig` on
   Windows, enabling mirrored networking mode. This is what makes a WSL
   service bound to `0.0.0.0` reachable via Windows' own `localhost`, and
   what puts WSL on the same LAN IP as Windows. `wsl --shutdown` again to
   apply, then start your distro.

## Phase 1 — Verify GPU passthrough

```
nvidia-smi
```
Should show the GPU immediately, no extra driver install inside WSL. If this
doesn't work, stop here — nothing downstream will either.

## Phase 2 — CUDA toolkit (nvcc)

The driver alone is **not** enough for models that JIT-compile CUDA kernels
at load time (this one does — see Phase 4 notes). You need the separate CUDA
*toolkit* (the compiler, `nvcc`), matched to your driver's CUDA ceiling
(check `nvidia-smi`'s reported CUDA version) and your distro:

```
scripts/wsl-cuda-toolkit.sh 13-3   # or whatever version matches
```

## Phase 3 — Tailscale

```
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=anton   # prints a browser auth link
sudo tailscale set --operator=$USER  # avoid needing sudo for tailscale commands after this
```

Confirm HTTPS certs are enabled for the tailnet (needed for Tailscale
Serve/Funnel later):
```
tailscale cert anton.tail35bed8.ts.net
```
If this fails outright (not just "needs sudo"), HTTPS Certificates may need
enabling in the tailnet admin console (login.tailscale.com → DNS settings)
first.

**Windows Firewall** — run `scripts/windows-firewall-rules.ps1` in an
elevated PowerShell on Windows. Without this, WSL2 mirrored mode's loopback
relay silently drops traffic to fresh app ports (confirmed: port 22 worked
without this, a freshly-bound app port didn't) — a very confusing failure
mode since `nvidia-smi`/local processes all look fine.

**dbus gotcha** — this WSL instance doesn't reliably start dbus on boot
(seen after suspend/resume more than a true cold boot), which breaks
`systemctl` entirely and silently takes out `tailscaled` with it. If
`systemctl status` fails with "Failed to connect to system scope bus", run
`scripts/wsl-dbus-fix.sh`. (The qwen-rtx deployment's own `start.sh` also
runs this check automatically, so this is mostly a fallback / a fix for
before you've cloned that repo yet.)

## Phase 4 — Clone the deployment and bring the model up

```
git clone git@gitlab.com:alcuka/ai/llm.git
cd llm/qwen-rtx
./setup.sh   # builds its own uv-managed venv, generates a fresh .api_key
./start.sh   # downloads the model from HF (~12GB), launches vLLM
```

Full reasoning for every value in `qwen-rtx/profile.env` (GPU arch, dtype,
quantization kernel choice, why `MAX_NUM_SEQS`/`REASONING_PARSER` are set)
is documented in `qwen-rtx/README.md` in that repo — authoritative source,
not duplicated here. The short version of what's model-specific and
easy to forget when rebuilding from scratch:

- This model (Qwen3.5, hybrid linear-attention arch) needs `nvcc` (Phase 2)
  and `ninja` (installed automatically by `setup.sh`) to JIT-compile a
  kernel at load time. Driver-only (`nvidia-smi` working) is not sufficient.
- `MAX_NUM_SEQS` is capped low (16) because the hybrid architecture's
  Mamba-style cache is a fixed slot per concurrent sequence, not something
  that scales down gracefully like a normal KV cache — vllm's much higher
  default fails CUDA graph capture outright at startup rather than just
  queuing requests.
- It's a reasoning/"thinking" model — `REASONING_PARSER=qwen3` splits
  chain-of-thought into a separate API field; without it, clients either
  show raw "Thinking Process: ..." text or truncate before the real answer
  if `max_tokens` is too low.
- **Don't trust self-directed localhost health checks on this host** — the
  same mirrored-mode loopback relay issue from Phase 3 means
  `curl localhost:8000` (including `start.sh`'s own readiness probe) can
  hang even when the server is completely healthy. Verify via
  `nvidia-smi`/`server.log`/the tailscale IP directly instead of trusting
  a "not ready" verdict at face value.

## Phase 5 — Expose over Tailscale

Tailnet-only:
```
tailscale serve --bg --https=443 http://<tailscale-ip>:8000
```
Public internet (anyone with the URL + API key, no tailnet membership
needed — a real exposure decision, not a default):
```
tailscale funnel --bg --https=443 http://<tailscale-ip>:8000
```

Must target the machine's actual tailscale IP (`tailscale ip -4`), **not**
`127.0.0.1` — same mirrored-mode loopback relay issue. Bind vllm to
`0.0.0.0` (already the default in `qwen-rtx/profile.env` via `start.sh`) so
both the direct port and the serve/funnel proxy can reach it.

This config persists in `tailscaled`'s own state across reboots — vLLM
itself does not, always needs `./start.sh` again after a restart.

## Phase 6 — Point clients at it

Full path including `/chat/completions`, not just the base URL:
```
https://anton.tail35bed8.ts.net/v1/chat/completions
```
Bearer auth via the contents of `qwen-rtx/.api_key`. Model aliases already
set up so tools hardcoding OpenAI names work: `gpt-4o`, `gpt-4`,
`gpt-3.5-turbo`, plus `qwen3.5-9b`.

## Not covered here (by design)

- Anything inside `qwen-rtx/` itself (profile.env values, setup/start/stop
  scripts) — that's the `llm` repo's job, kept as the single source of
  truth so this backup and that repo don't drift out of sync.
- Actual secrets (`.api_key`, any live keys) — never committed here or
  there; regenerated fresh by `setup.sh` on any new machine.
