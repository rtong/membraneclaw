#!/bin/bash
# After 12's 200 steps: (1) diagnose the arithmetic regression from step 128 on -- the final policy and
# the step-125 best, greedy on dev, with every episode kept; (2) 13a attempt 1 (call credit, lam 1.0);
# (3) 13b's equivalence gate (qwen-agent harness on the tool seed must reproduce sft_raw_dev.json).
cd "$(dirname "$0")/../.."
until grep -q "12-200 exit" runs/tool/rerun.status 2>/dev/null; do sleep 20; done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
R=runs/ppo-on-tool-seed-warm-200
mkdir -p $R/best-as-run && ln -sfn ../adapter-best $R/best-as-run/adapter && ln -sfn ../value_head-best.pt $R/best-as-run/value_head.pt
$PY ppo_ac.py --eval-only 1 --prompt-tool 1 --weights ABLATE --resume-from $R --out runs/tool/warm200_final_dev.json > runs/tool/diag12.log 2>&1
$PY ppo_ac.py --eval-only 1 --prompt-tool 1 --weights ABLATE --resume-from $R/best-as-run --out runs/tool/warm200_best125_dev.json >> runs/tool/diag12.log 2>&1
echo "diag12 exit $?" >> runs/tool/rerun.status
$PY ppo_ac.py --prompt-tool 1 --resume-from runs/tool-sft-raw --value-head-reset 1 \
    --call-credit 1 --weights ABLATE --dense-rewards 1 --lam 1.0 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 50 --eval-every 25 --out runs/ppo-tool-callcredit-lam1-50 > runs/tool/callcredit_lam1.log 2>&1
echo "13a-lam1 exit $?" >> runs/tool/rerun.status
$PY ppo_ac.py --eval-only 1 --prompt-tool 1 --harness qwen_agent --weights ABLATE \
    --resume-from runs/tool-sft-raw --out runs/tool/sft_raw_dev_qwen_agent.json > runs/tool/eq13b.log 2>&1
echo "13b-eq exit $?" >> runs/tool/rerun.status
