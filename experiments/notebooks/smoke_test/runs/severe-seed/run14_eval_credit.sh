#!/bin/bash
# 14: per-case dev pass for the severe-credit run's best adapter (step 125, EM 0.855; the run stopped on
# sampled arithmetic at 141), same harness and batch as the in-run eval.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/severe-seed
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent --weights ABLATE \
    --eval-only 1 --resume-from /tmp/claude-1000/-home-bayan-MembraneClaw/718964fa-fe7e-4622-b46c-5c00cabf37a3/scratchpad/credit_best125 \
    --max-vram-mib 13000 --out $OUT/dev_credit_best125.json > $OUT/dev_credit_best125.log 2>&1
echo "dev credit_best125 exit $?" >> $OUT/status
