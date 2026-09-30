#!/usr/bin/env bash
# The scaffold seed's three epochs on dev, greedy, answers stopped at their closing brace (the
# protocol PPO evaluates with). Run after the seed has finished training.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
for e in 1 2 3; do
    ../../membrane_grpo/.venv/bin/python 04_natural_watertap.py probe --split dev --stop-at-json 1 \
        --adapter 04_natural_watertap_runs/seed-scaffold/epoch$e --out seed-scaffold/epoch$e
done
