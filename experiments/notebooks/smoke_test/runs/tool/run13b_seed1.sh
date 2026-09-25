#!/bin/bash
# 13b, second seed: run13b_train.sh with --seed 1 and nothing else changed (gamma stays 1.0), so the
# 200-step result is not one run. Compared against runs/ppo-tool-qwen-agent-scoped-200 (seed 0).
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/tool-sft-raw --value-head-reset 1 --entropy-scope answer --numeric-stop 0.90 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 200 --eval-every 25 --seed 1 \
    --out runs/ppo-tool-qwen-agent-scoped-200-s1 > runs/tool/qa_scoped200_s1.log 2>&1
echo "13b-seed1 exit $?" >> runs/tool/rerun.status
