#!/bin/sh
# 03_label_levers, run eps1e6-s0: frozen-s0's configuration (stage seed, answer structure frozen)
# with the actor's Adam epsilon at 1e-6, the median root second moment measured at frozen-s0's
# step 50 -- the one change. Every evaluation's adapter kept; resumes after a stall.
cd "$(dirname "$0")"
RUN=03_label_levers_runs/eps1e6-s0
PY=../../membrane_grpo/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p 03_label_levers_runs
tries=0
while [ ! -e "$RUN/ending.json" ] && [ "$tries" -lt 8 ]; do
  tries=$((tries + 1))
  # the card is shared with the display and games: (re)start only when nothing else holds it
  while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 2500 ]; do
    sleep 60
  done
  if [ -e "$RUN/last/state.pt" ]; then
    echo "=== resume attempt $tries $(date)" >> $RUN.log
    $PY 03_label_levers.py train --run eps1e6-s0 --resume --max-vram-mib 13000 >> $RUN.log 2>&1
  else
    rm -rf "$RUN"
    echo "=== start attempt $tries $(date)" >> $RUN.log
    $PY 03_label_levers.py train --run eps1e6-s0 --start-from 03_label_levers_runs/tool-seed-stage/epoch3 \
      --value-layer 23 --value-lr 0.0009189867391542886 --value-init-bias 0.4344 --critic-warmup 25 \
      --freeze-turn-ends 1 --freeze-answer-structure 1 --adam-eps 1e-6 --entropy-coef 0 --stop-at-json 1 --kl-coef 0.01 --flat-credit-scale 3.0 \
      --keep-eval-checkpoints 1 --max-vram-mib 13000 >> $RUN.log 2>&1
  fi
  [ -e "$RUN/ending.json" ] || sleep 60
done
