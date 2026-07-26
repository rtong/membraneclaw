#!/usr/bin/env bash
# Open WebUI, pointed at the agent server rather than at vLLM directly.
#
# --network=host is what lets the container reach the agent on 127.0.0.1:8001,
# so the agent never has to listen on a routable interface.
set -euo pipefail

cd "$(dirname "$0")"
set -a; source .env; set +a

# Deliberately not NAME/WEBUI_NAME: NAME is often already set in the shell, and
# WEBUI_NAME is Open WebUI's own branding variable.
CONTAINER_NAME=${CONTAINER_NAME:-open-webui}
PORT=${WEBUI_PORT:-3000}

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exec docker run -d --name "$CONTAINER_NAME" \
  --network=host \
  --restart unless-stopped \
  -e PORT="$PORT" \
  -e OPENAI_API_BASE_URL="http://127.0.0.1:8001/v1" \
  -e OPENAI_API_KEY="$AGENT_API_KEY" \
  -e ENABLE_OLLAMA_API=false \
  -e ENABLE_SIGNUP="${ENABLE_SIGNUP:-false}" \
  -e WEBUI_SECRET_KEY="$AGENT_API_KEY" \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
