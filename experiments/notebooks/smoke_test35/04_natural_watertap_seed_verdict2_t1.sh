#!/usr/bin/env bash
# seed-verdict2 epoch 3 sampled at T=1, two samples per dev question, answers stopped at their closing
# brace: does every answer get said (PPO can only reweight what it samples), and do sampled calls hold?
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
../../membrane_grpo/.venv/bin/python 04_natural_watertap.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 \
    --adapter 04_natural_watertap_runs/seed-verdict2/epoch3 --out seed-verdict2/epoch3
