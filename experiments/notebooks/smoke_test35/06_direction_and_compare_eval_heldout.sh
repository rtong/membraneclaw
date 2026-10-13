#!/usr/bin/env bash
# Held out: the seed PPO started from on test and holdout_shift and, when the run's best evaluation is a
# checkpoint of its own (best_step > 0), that checkpoint on test, holdout_shift and all of dev. Greedy,
# answers stopped at their closing brace -- the protocol every evaluation in this notebook uses. The
# other runs and doses are held out the same way (SEED_RUN=label-seed-n1200 PPO_RUN=ppo-n1200-open-s0 ./...).
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
R=06_direction_and_compare_runs
SEED_RUN=${SEED_RUN:-label-seed-n600}    # the seed PPO started from: the dose
SEED_CKPT=${SEED_CKPT:-epoch1}           # its checkpoint
PPO_RUN=${PPO_RUN:-ppo-n600-open-turn-s0}   # the PPO run from it: ppo-n<dose>-open-turn-s0 (the turn as the unit of credit), ppo-n<dose>-open-lam1-s0 (lambda 1 per token), ppo-n<dose>-open-s0 (lambda 0.95 per token), ppo-n<dose>-s0 (tool gate on)
for split in test holdout_shift; do
    [ -f $R/$SEED_RUN/$SEED_CKPT/$split-T0-stop.json ] || $PY 06_direction_and_compare.py probe --split $split --stop-at-json 1 \
        --adapter $R/$SEED_RUN/$SEED_CKPT --out $SEED_RUN/$SEED_CKPT
done
[ -f $R/$PPO_RUN/ending.json ] || exit 0
BEST=$(python3 -c "import json; print(json.load(open('$R/$PPO_RUN/ending.json'))['best_step'])")
[ "$BEST" != 0 ] || { echo "$PPO_RUN: the best evaluation is step 0, the seed itself -- nothing more to hold out"; exit 0; }
for split in test holdout_shift dev; do
    $PY 06_direction_and_compare.py probe --split $split --stop-at-json 1 --adapter $R/$PPO_RUN/adapter-best --out $PPO_RUN/best
done
