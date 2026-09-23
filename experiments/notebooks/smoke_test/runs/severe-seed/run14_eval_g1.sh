#!/bin/bash
# 14, after step 2: per-case greedy dev passes, for the two labels the in-run eval (scalars only) cannot show.
# The PPO run's best adapter (step 150; the run stopped on sampled arithmetic at 178), and 13b seed 0's final
# adapter as the reference; same harness and batch as the in-run eval, so the joint line reproduces eval.jsonl @150.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
OUT=runs/severe-seed

dev() {  # name, run directory
    $PY ppo_ac.py --prompt-tool 1 --harness qwen_agent --weights ABLATE --eval-only 1 --resume-from "$2" \
        --max-vram-mib 13000 --out $OUT/dev_$1.json > $OUT/dev_$1.log 2>&1
    echo "dev $1 exit $?" >> $OUT/status
}
dev ppo_joint_g1_best150 /tmp/claude-1000/-home-bayan-MembraneClaw/718964fa-fe7e-4622-b46c-5c00cabf37a3/scratchpad/joint_best150
dev 13b_s0_final runs/ppo-tool-qwen-agent-scoped-200
echo "eval g1 done" >> $OUT/status
