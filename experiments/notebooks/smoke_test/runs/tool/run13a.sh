#!/bin/bash
# 13a: the corrected tool gate run (runs/ppo-on-tool-seed-warm-gate40) plus one change, --call-credit 1.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --resume-from runs/tool-sft-raw --value-head-reset 1 \
    --call-credit 1 --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 40 --eval-every 10 --out runs/ppo-tool-callcredit-gate40 > runs/tool/gate40_callcredit.log 2>&1
echo "13a exit $?" >> runs/tool/rerun.status
