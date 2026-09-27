#!/bin/sh
# 03_label_levers, the stage seed: 02's tool seed (the three gold calls, <|im_end|>, the answer's
# `{`, on the 400 train cases) with the stage value labelled too -- a copy of a value the prompt
# states. 02's seed wrote a root cause into `stage` on 164 of 200 dev answers. Then greedy on dev.
cd "$(dirname "$0")"
PY=../../membrane_grpo/.venv/bin/python
R=03_label_levers_runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
[ -e $R/tool-seed-stage/epoch3/adapter_model.safetensors ] || $PY 03_label_levers.py seed --out tool-seed-stage \
  --slots stage --max-vram-mib 13000 >> $R/tool-seed-stage.log 2>&1
[ -e $R/tool-seed-stage/epoch3-dev.json ] || $PY 03_label_levers.py eval --adapter $R/tool-seed-stage/epoch3 \
  --split dev --stop-at-json 1 --batch 32 --out $R/tool-seed-stage/epoch3-dev.json --max-vram-mib 13000 >> $R/tool-seed-stage.log 2>&1
