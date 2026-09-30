#!/usr/bin/env bash
# 40-step gate run: PPO from the label seed (label-seed-pc2, epoch 3), 03's configuration (critic
# warm-up 25, turn ends and answer scaffolding out of the actor's loss, Adam eps 1e-6, answers stopped
# at their closing brace, KL 0.01 to the seed), value head on layer 23 warm-started at the seed's own
# sampled dev reward (0.807), tool gate 0.90 on the sampled call fidelity. Greedy evals on 120 dev
# questions at steps 0, 20 and 40.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
../../membrane_grpo/.venv/bin/python 04_natural_watertap.py ppo --out gate40-s0 \
    --start-from 04_natural_watertap_runs/label-seed-pc2/epoch3 --value-init-bias 0.807 \
    --steps 40 --eval-every 20 --tool-stop 0.9
