#!/usr/bin/env bash
# Held out: ppo-s0's best checkpoint (step 300, chosen on 120 dev questions) on test, holdout_shift and
# all of dev; the label seed alone (the no-PPO control) on test and holdout_shift. Greedy, answers
# stopped at their closing brace -- the protocol every evaluation in this notebook uses.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
for split in test holdout_shift dev; do
    $PY 04_natural_watertap.py probe --split $split --stop-at-json 1 \
        --adapter 04_natural_watertap_runs/ppo-s0/adapter-best --out ppo-s0/best
done
for split in test holdout_shift; do
    $PY 04_natural_watertap.py probe --split $split --stop-at-json 1 \
        --adapter 04_natural_watertap_runs/label-seed-pc2/epoch3 --out label-seed-pc2/epoch3
done
