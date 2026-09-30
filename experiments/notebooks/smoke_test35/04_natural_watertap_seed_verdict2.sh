#!/usr/bin/env bash
# The verdict seed, JSON syntax labelled: `seed-verdict`'s recipe, but the answer's quotes, brackets and
# commas are always labelled and a scan's decision is the setting as the user named it ("2,100 ft²").
# Left to PPO: every simulated value and limit in the lines, and the JSON's decision,
# violations and numbers. 3 epochs, lr 1e-4, one step per 8 episodes, LoRA r16; then each epoch on
# dev, greedy, answers stopped at their closing brace.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
$PY 04_natural_watertap.py seed --out seed-verdict2 --label scaffold --reasoning 1 --epochs 3 --lr 1e-4 --accum 8 --lora-r 16
for e in 1 2 3; do
    $PY 04_natural_watertap.py probe --split dev --stop-at-json 1 \
        --adapter 04_natural_watertap_runs/seed-verdict2/epoch$e --out seed-verdict2/epoch$e
done
