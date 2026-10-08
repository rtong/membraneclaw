#!/usr/bin/env bash
# The label seed, 04's recipe at 04's chosen dose: the scaffold seed with the comparison lines and the
# aggregation line labelled, the JSON's syntax labelled, and the decision itself labelled on 2 train
# records per (family, answer class). New in 05: the convert turn, labelled like every calls turn.
# 3 epochs, lr 1e-4, one step per 8 episodes, LoRA r16. Then epoch 3 on dev: greedy, and sampled at
# T=1 twice per question (does every answer class get said; do the converts and the calls hold).
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
$PY 05_units_and_scans.py seed --out label-seed-pc2 --label scaffold --reasoning 1 --epochs 3 --lr 1e-4 \
    --accum 8 --lora-r 16 --label-per-class 2
$PY 05_units_and_scans.py probe --split dev --stop-at-json 1 \
    --adapter 05_units_and_scans_runs/label-seed-pc2/epoch3 --out label-seed-pc2/epoch3
$PY 05_units_and_scans.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 \
    --adapter 05_units_and_scans_runs/label-seed-pc2/epoch3 --out label-seed-pc2/epoch3
