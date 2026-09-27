#!/bin/sh
# 03_label_levers, held out: eps1e6-s0's best adapter (step 700, chosen on dev) on dev (with episodes),
# test, holdout_shift and train; the stage seed alone on test and holdout_shift as the control.
# Greedy, the answer ending where its JSON closes, as in training.
cd "$(dirname "$0")"
PY=../../membrane_grpo/.venv/bin/python
R=03_label_levers_runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
for SPLIT in dev test holdout_shift train; do
  [ -e $R/eps1e6-s0/best-$SPLIT.json ] || $PY 03_label_levers.py eval --adapter $R/eps1e6-s0/adapter-best --split $SPLIT \
    --stop-at-json 1 --batch 32 --out $R/eps1e6-s0/best-$SPLIT.json --max-vram-mib 8000 >> $R/eps1e6-s0/heldout.log 2>&1
done
for SPLIT in test holdout_shift; do
  [ -e $R/tool-seed-stage/epoch3-$SPLIT.json ] || $PY 03_label_levers.py eval --adapter $R/tool-seed-stage/epoch3 \
    --split $SPLIT --stop-at-json 1 --batch 32 --out $R/tool-seed-stage/epoch3-$SPLIT.json --max-vram-mib 8000 >> $R/eps1e6-s0/heldout.log 2>&1
done
