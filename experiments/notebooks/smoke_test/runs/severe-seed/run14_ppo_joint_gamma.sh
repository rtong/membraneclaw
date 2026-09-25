#!/bin/bash
# 14, step 3: run14_ppo_joint_g1.sh with a per-token discount, gamma 0.99 then 0.999, nothing else changed.
#
# gamma_scope token: gamma on every policy token (the tool's reply is not a step). An episode is ~335 policy
# tokens, ~160 of them calls, so the answer's rewards reach the first call discounted by about 0.99^200 = 0.13
# or 0.999^200 = 0.82. Predicted before running: with V starting flat at 0.6743, every unrewarded token has a
# TD error of about -(1 - gamma) x 0.67 until V learns the discounted targets, which lam 0.95 accumulates to an
# advantage offset of about -0.11 per token at 0.99 and -0.013 at 0.999.
# Gates as g1: step 40 = calls entropy <= 0.001 and no numeric stop; critic 50-74 ahead of clock on >= 50%.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/severe-seed

for g in 0.99 0.999; do
    tag=${g/./}
    echo "ppo joint gamma $g start $(date -Is)" >> $OUT/status
    ../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent \
        --resume-from runs/tool-seed-joint-150/epoch1 --value-head-reset 1 --value-init-bias 0.6743 \
        --entropy-scope answer --numeric-stop 0.90 \
        --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
        --critic-window 25 --steps 200 --eval-every 25 --gamma $g --gamma-scope token --max-vram-mib 13700 \
        --out runs/ppo-tool-joint-seed-200-g$tag > $OUT/ppo_joint_g$tag.log 2>&1
    echo "ppo joint gamma $g exit $?" >> $OUT/status
done
