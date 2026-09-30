#!/usr/bin/env bash
# The full run: 04's configuration (critic warm-up 25; turn ends and answer scaffolding out of the
# actor's loss; Adam eps 1e-6; answers stopped at their closing brace; KL 0.01 to the seed; value head
# on layer 23 warm-started at the seed's own sampled dev reward; tool gate 0.90), from this notebook's
# label seed, with the policy's token budget at 2,048. Runs until a gate or convergence (8 evaluations
# without a new best) stops it. Greedy evals on 120 dev questions every 25 steps; every evaluation
# checkpoints `last/`, and a run the watchdog ends is resumed from it (up to 8 times). Each (re)start
# waits until the card has room (display and games keep 2 GB).
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
RUN=ppo-s0
BIAS=$(python3 -c "import json; print(round(json.load(open('05_units_and_scans_runs/label-seed-pc2/epoch3/dev-T1-x2-stop.json'))['overall']['reward'], 4))")
ARGS="--out $RUN --start-from 05_units_and_scans_runs/label-seed-pc2/epoch3 --value-init-bias $BIAS --steps 2000 --eval-every 25 --tool-stop 0.9"
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2500 ]; do sleep 60; done; }
wait_card
$PY 05_units_and_scans.py ppo $ARGS
for attempt in 1 2 3 4 5 6 7 8; do
    [ -f 05_units_and_scans_runs/$RUN/ending.json ] && exit 0
    wait_card
    if [ -f 05_units_and_scans_runs/$RUN/last/state.pt ]; then
        $PY 05_units_and_scans.py ppo $ARGS --resume 1
    else
        $PY 05_units_and_scans.py ppo $ARGS
    fi
done
