#!/usr/bin/env bash
# The verdict seed: `seed-scaffold`'s recipe with the comparison lines' verdicts (meets/breaks,
# pass/fail) labelled too. Labelled: the calls turn, the line skeletons and their verdicts, the JSON's
# scaffolding. Left to PPO: every simulated value and limit in the lines, and the JSON's decision,
# violations and numbers. 3 epochs, lr 1e-4, one step per 8 episodes, LoRA r16; then each epoch on
# dev, greedy, answers stopped at their closing brace.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
$PY 04_natural_watertap.py seed --out seed-verdict --label scaffold --reasoning 1 --epochs 3 --lr 1e-4 --accum 8 --lora-r 16
for e in 1 2 3; do
    $PY 04_natural_watertap.py probe --split dev --stop-at-json 1 \
        --adapter 04_natural_watertap_runs/seed-verdict/epoch$e --out seed-verdict/epoch$e
done
