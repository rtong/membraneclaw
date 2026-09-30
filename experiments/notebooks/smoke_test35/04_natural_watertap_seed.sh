#!/usr/bin/env bash
# The scaffold seed: from raw Qwen3.5-0.8B, on the train split's blind reference episodes (every
# candidate simulated in one turn, then one comparison line per candidate and the JSON answer).
# Labelled: the calls turn, and the answer's scaffolding (line skeletons, field names, operators,
# braces, value keys). Left to PPO: every simulated value, limit and verdict in the lines, and the
# decision, violations and numbers in the JSON. 3 epochs, lr 1e-4, one step per 8 episodes, LoRA r16.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
../../membrane_grpo/.venv/bin/python 04_natural_watertap.py seed --out seed-scaffold --label scaffold --reasoning 1 \
    --epochs 3 --lr 1e-4 --accum 8 --lora-r 16
