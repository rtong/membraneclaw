#!/usr/bin/env bash
# Label-seed dose sweep: seed-verdict2's recipe, plus the decision itself labelled on k train records per
# (family, answer class) -- membrane A-D/none, plant options A-C/none, scans a-setting/none -- for
# k = 2, 4, 8. Each seed's epoch 3 is then sampled at T=1, twice per dev question (answers stopped at
# their closing brace): the weakest dose under which every answer class is said right is the one PPO
# starts from (03's lever: a label never sampled cannot be learned).
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
for k in 2 4 8; do
    $PY 04_natural_watertap.py seed --out label-seed-pc$k --label scaffold --reasoning 1 --epochs 3 --lr 1e-4 \
        --accum 8 --lora-r 16 --label-per-class $k
    $PY 04_natural_watertap.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 \
        --adapter 04_natural_watertap_runs/label-seed-pc$k/epoch3 --out label-seed-pc$k/epoch3
done
