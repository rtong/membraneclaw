#!/usr/bin/env bash
# The full run: gate40-s0's configuration (PPO from label-seed-pc2 epoch 3; critic warm-up 25; turn ends
# and answer scaffolding out of the actor's loss; Adam eps 1e-6; answers stopped at their closing brace;
# KL 0.01 to the seed; value head on layer 23 warm-started at 0.807; tool gate 0.90), run until a gate
# or convergence (8 evaluations without a new best) stops it. Greedy evals on 120 dev questions every
# 25 steps; every evaluation checkpoints `last/`, and a run the watchdog ends is resumed from it (up to
# 8 times). Each (re)start waits until the card has room (display and games keep 2 GB).
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
RUN=ppo-s0
ARGS="--out $RUN --start-from 04_natural_watertap_runs/label-seed-pc2/epoch3 --value-init-bias 0.807 --steps 2000 --eval-every 25 --tool-stop 0.9"
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2500 ]; do sleep 60; done; }
wait_card
$PY 04_natural_watertap.py ppo $ARGS
for attempt in 1 2 3 4 5 6 7 8; do
    [ -f 04_natural_watertap_runs/$RUN/ending.json ] && exit 0
    wait_card
    if [ -f 04_natural_watertap_runs/$RUN/last/state.pt ]; then
        $PY 04_natural_watertap.py ppo $ARGS --resume 1
    else
        $PY 04_natural_watertap.py ppo $ARGS
    fi
done
