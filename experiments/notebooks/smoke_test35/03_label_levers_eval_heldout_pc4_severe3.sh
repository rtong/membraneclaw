#!/bin/sh
# 03_label_levers, held out: pc4-severe3-s0's best adapter (chosen on dev) on dev (with episodes), test,
# holdout_shift and train. Its seed is pc4-eps-s0's, already evaluated by 03_label_levers_eval_heldout_pc4.sh.
# Greedy, the answer ending where its JSON closes, as in training.
cd "$(dirname "$0")"
PY=../../membrane_grpo/.venv/bin/python
R=03_label_levers_runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
for SPLIT in dev test holdout_shift train; do
  [ -e $R/pc4-severe3-s0/best-$SPLIT.json ] || $PY 03_label_levers.py eval --adapter $R/pc4-severe3-s0/adapter-best \
    --split $SPLIT --stop-at-json 1 --batch 32 --out $R/pc4-severe3-s0/best-$SPLIT.json --max-vram-mib 8000 \
    >> $R/pc4-severe3-s0/heldout.log 2>&1
done
