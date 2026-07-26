#!/bin/bash
# Orchestrates ../README.md's migration plan end-to-end on a fresh WSL
# instance. Run this yourself, interactively, inside WSL on the new machine
# (not something to run unattended — it pauses for the Windows-side manual
# steps and for Tailscale's browser auth link).
#
# Safe to re-run: every phase checks whether it's already done and skips.
#
# Usage:
#   ./deploy.sh            # expose over Tailscale Serve (tailnet-only, default)
#   ./deploy.sh funnel      # expose over Tailscale Funnel (public internet)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_REPO="git@gitlab.com:alcuka/ai/llm.git"
LLM_DIR="$HOME/kod/llm"
TS_HOSTNAME="${TS_HOSTNAME:-anton}"
EXPOSE_MODE="${1:-serve}"

if [ "$EXPOSE_MODE" != "serve" ] && [ "$EXPOSE_MODE" != "funnel" ]; then
  echo "Usage: $0 [serve|funnel]"
  exit 1
fi

pause() {
  echo ""
  read -rp ">> Press Enter once done, or Ctrl-C to stop here... " _
}

echo "=== Phase 0: Windows driver + WSL mirrored networking ==="
echo "Can't be automated from inside WSL. On the Windows host:"
echo "  1. Install/update the NVIDIA driver (Game Ready or Studio)."
echo "  2. Elevated PowerShell: wsl --update && wsl --shutdown"
echo "  3. Copy $SCRIPT_DIR/../.wslconfig.example to %UserProfile%\\.wslconfig"
echo "  4. wsl --shutdown again, then restart your distro and re-run this script"
pause

echo ""
echo "=== Phase 1: verify GPU passthrough ==="
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not working. Stop and fix Phase 0 before continuing."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo "GPU passthrough OK."

echo ""
echo "=== Phase 2: CUDA toolkit (nvcc) ==="
if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
  echo "nvcc already present, skipping."
else
  CUDA_UMD=$(nvidia-smi | grep -oP 'CUDA (UMD )?Version:\s*\K[0-9]+\.[0-9]+' | head -1)
  CUDA_VER="${CUDA_UMD//./-}"
  echo "Detected CUDA ceiling: $CUDA_UMD -> installing cuda-toolkit-$CUDA_VER"
  "$SCRIPT_DIR/wsl-cuda-toolkit.sh" "$CUDA_VER"
fi
export PATH="/usr/local/cuda/bin:$PATH"

echo ""
echo "=== Phase 3: Tailscale ==="
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
"$SCRIPT_DIR/wsl-dbus-fix.sh" || true
if ! tailscale status 2>/dev/null | grep -q "$TS_HOSTNAME"; then
  echo "Bringing up tailscale — click the auth link that appears below."
  sudo tailscale up --hostname="$TS_HOSTNAME"
fi
sudo tailscale set --operator="$USER"
tailscale cert "${TS_HOSTNAME}.$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin).get("MagicDNSSuffix",""))')" >/dev/null 2>&1 || true

echo ""
echo "IMPORTANT: on Windows, run scripts/windows-firewall-rules.ps1 in an"
echo "elevated PowerShell now if you haven't already — mirrored-mode's"
echo "loopback relay silently drops traffic to fresh app ports otherwise."
pause

echo ""
echo "=== Phase 4: clone + deploy qwen-rtx ==="
if [ ! -d "$LLM_DIR" ]; then
  git clone "$LLM_REPO" "$LLM_DIR"
fi
cd "$LLM_DIR/qwen-rtx"
./setup.sh
./start.sh

echo ""
echo "=== Phase 5: expose over Tailscale ($EXPOSE_MODE) ==="
TS_IP=$(tailscale ip -4)
if [ "$EXPOSE_MODE" = "funnel" ]; then
  echo "Exposing PUBLICLY via Funnel — anyone with the URL + API key can reach this, no tailnet membership needed."
  tailscale funnel --bg --https=443 "http://${TS_IP}:8000"
else
  echo "Exposing over Serve — tailnet members only."
  tailscale serve --bg --https=443 "http://${TS_IP}:8000"
fi
tailscale serve status

echo ""
echo "=== Done ==="
echo "API key:      $LLM_DIR/qwen-rtx/.api_key"
echo "Served names: see $LLM_DIR/qwen-rtx/profile.env (SERVED_NAMES)"
echo "URL:          see 'tailscale serve status' output above"
echo ""
echo "Remember (see qwen-rtx/README.md): don't trust self-directed localhost"
echo "health checks on this host — verify via the tailscale IP/URL instead."
