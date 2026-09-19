#!/bin/bash
# 13b with a per-turn discount: run13b_train.sh (seed 0) plus --gamma 0.9 --gamma-scope turn, nothing else that
# enters Config. gamma applies only on the step across the tool's response; inside a turn it stays 1.0 (see
# turn_discounts). One variable against runs/ppo-tool-qwen-agent-scoped-200; the seed-1 run is the noise scale.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/tool-sft-raw --value-head-reset 1 --entropy-scope answer --numeric-stop 0.90 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 200 --eval-every 25 --gamma 0.9 --gamma-scope turn --max-vram-mib 13700 \
    --out runs/ppo-tool-qwen-agent-scoped-200-gturn09 > runs/tool/qa_scoped200_gturn09.log 2>&1
echo "13b-gamma-turn exit $?" >> runs/tool/rerun.status
