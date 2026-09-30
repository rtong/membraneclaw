#!/bin/sh
# 02_tool_seed_ppo: the held-out measurements for kl001-s0. The converged run's best checkpoint
# (step 500, chosen on dev) and the tool seed it started from, on test and holdout_shift --
# neither split chose anything -- plus the checkpoint on train against dev, for overfitting.
# Greedy, the same turn protocol as training (a turn ends where its JSON object closes).
cd "$(dirname "$0")"
PY=../../membrane_grpo/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
BEST=02_tool_seed_ppo_runs/kl001-s0/adapter-best
SEED=02_tool_seed_ppo_runs/tool-seed-anchor/epoch3
for split in test holdout_shift train; do
  timeout 1800 $PY 02_tool_seed_ppo.py eval --adapter $BEST --split $split --stop-at-json 1 \
    --out 02_tool_seed_ppo_runs/kl001-s0/best-$split.json --max-vram-mib 13000
done
for split in dev test holdout_shift; do
  timeout 1800 $PY 02_tool_seed_ppo.py eval --adapter $SEED --split $split --stop-at-json 1 \
    --out 02_tool_seed_ppo_runs/tool-seed-anchor/epoch3-stopjson-$split.json --max-vram-mib 13000
done
echo HELDOUT-DONE
