#!/bin/bash
# 12's corrected configuration (runs/ppo-on-tool-seed-warm-gate40) run for 200 steps from the tool seed,
# evaluated every 25 steps as 09 was: can the tool line reach 09's numbers?
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --resume-from runs/tool-sft-raw --value-head-reset 1 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 200 --eval-every 25 --out runs/ppo-on-tool-seed-warm-200 > runs/tool/run12_200.log 2>&1
echo "12-200 exit $?" >> runs/tool/rerun.status
