#!/usr/bin/env bash
# Start the vLLM backend with tool calling enabled.
#
# --enable-auto-tool-choice/--tool-call-parser are required: without them the
# server rejects any request carrying `tools`. Qwen3.5 emits XML tool calls, so the
# parser must be qwen3_xml — `hermes` expects JSON and will not match.
set -euo pipefail

cd "$(dirname "$0")"
set -a; source .env; set +a

VLLM_HOME=${VLLM_HOME:-/home/bayan/vllm-env}
MODEL_PATH=${MODEL_PATH:-/home/bayan/models/Qwen3.5-9B-AWQ}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8000}

# torch inductor shells out to `ninja`, which only ships inside the venv's bin.
export PATH="$VLLM_HOME/bin:$PATH"

exec "$VLLM_HOME/bin/vllm" serve "$MODEL_PATH" \
  --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --served-model-name qwen3.5-9b gpt-4o gpt-4 gpt-3.5-turbo \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --api-key "$VLLM_API_KEY"
