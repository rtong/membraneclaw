#!/bin/bash
# (A) notebook 09's command on the current code, 40 steps: does the single-turn path still reproduce it?
# (B) the tool gate run again, with the value head warm-started where the policy actually is.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
COMMON="--weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 --critic-window 25 --steps 40"
$PY ppo_ac.py --prompt-v3 1 $COMMON --eval-every 40 --out runs/repro-09-current-code-40 > runs/tool/repro09.log 2>&1
echo "A exit $?" >> runs/tool/rerun.status
$PY ppo_ac.py --prompt-tool 1 --resume-from runs/tool-sft-raw --value-head-reset 1 $COMMON --eval-every 10 \
    --out runs/ppo-on-tool-seed-warm-gate40 > runs/tool/gate40_warm.log 2>&1
echo "B exit $?" >> runs/tool/rerun.status
