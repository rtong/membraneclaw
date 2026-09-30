#!/bin/sh
# 02_tool_seed_ppo: the raw model with the calculator on dev, then the tool seed
# (every train case, loss on the three calls only, 3 epochs at lr 1e-4) and each
# epoch's adapter on dev, greedy.
cd "$(dirname "$0")"
mkdir -p 02_tool_seed_ppo_runs
PY=../../membrane_grpo/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
{
  $PY 02_tool_seed_ppo.py eval --out 02_tool_seed_ppo_runs/raw-calculator-dev.json --max-vram-mib 13000 &&
  $PY 02_tool_seed_ppo.py seed --out tool-seed --epochs 3 --lr 1e-4 --max-vram-mib 13000 &&
  for e in 1 2 3; do
    $PY 02_tool_seed_ppo.py eval --adapter 02_tool_seed_ppo_runs/tool-seed/epoch$e \
      --out 02_tool_seed_ppo_runs/tool-seed/epoch$e-dev.json --max-vram-mib 13000 || exit 1
  done
  echo SEED-SCRIPT-DONE
} > 02_tool_seed_ppo_runs/tool-seed.log 2>&1
