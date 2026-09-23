#!/bin/bash
# 14: the severe-credit best adapter (step 125) on the two splits nothing in this notebook selected on.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/severe-seed
BEST=/tmp/claude-1000/-home-bayan-MembraneClaw/718964fa-fe7e-4622-b46c-5c00cabf37a3/scratchpad/credit_best125
for split in test holdout_shift; do
    ../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent --weights ABLATE \
        --eval-only 1 --resume-from $BEST --eval-split $split --max-vram-mib 13000 \
        --out $OUT/${split}_credit_best125.json > $OUT/${split}_credit_best125.log 2>&1
    echo "$split credit_best125 exit $?" >> $OUT/status
done
