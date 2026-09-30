#!/bin/sh
# 02_tool_seed_ppo: the tool seed with the answer turn's first token anchored
# (every train case, loss on the three calls, <|im_end|> and the answer's opening
# `{`; 3 epochs at lr 1e-4), then each epoch's adapter on dev, greedy.
cd "$(dirname "$0")"
PY=../../membrane_grpo/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
{
  $PY 02_tool_seed_ppo.py seed --out tool-seed-anchor --epochs 3 --lr 1e-4 --anchor-open --max-vram-mib 13000 &&
  for e in 1 2 3; do
    $PY 02_tool_seed_ppo.py eval --adapter 02_tool_seed_ppo_runs/tool-seed-anchor/epoch$e \
      --out 02_tool_seed_ppo_runs/tool-seed-anchor/epoch$e-dev.json --max-vram-mib 13000 || exit 1
  done
  echo SEED-SCRIPT-DONE
} > 02_tool_seed_ppo_runs/tool-seed-anchor.log 2>&1
