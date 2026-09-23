#!/bin/bash
# 14: (1) the per-case dev pass for gamma 0.999's final adapter -- the two labels, which eval.jsonl cannot show.
#     (2) gamma 0.99 again with --numeric-stop 0.80 instead of 0.90, from the seed, nothing else changed.
#
# gamma 0.99 stopped at step 85 on the sampled-arithmetic rule, not on the critic: its last-10 mean was 0.896
# against 0.90. The dips it stopped on were single steps of 0.75-0.88 and held-out numeric_acc was 1.000 at
# step 75; 12's real collapse took numeric to 0, which 0.80 still catches.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
OUT=runs/severe-seed

$PY ppo_ac.py --prompt-tool 1 --harness qwen_agent --weights ABLATE --eval-only 1 \
    --resume-from runs/ppo-tool-joint-seed-200-g0999 --max-vram-mib 13000 \
    --out $OUT/dev_ppo_joint_g0999_final.json > $OUT/dev_ppo_joint_g0999_final.log 2>&1
echo "dev ppo_joint_g0999_final exit $?" >> $OUT/status

echo "ppo joint gamma 0.99 stop080 start $(date -Is)" >> $OUT/status
$PY ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/tool-seed-joint-150/epoch1 --value-head-reset 1 --value-init-bias 0.6743 \
    --entropy-scope answer --numeric-stop 0.80 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 200 --eval-every 25 --gamma 0.99 --gamma-scope token --max-vram-mib 13700 \
    --out runs/ppo-tool-joint-seed-200-g099-stop080 > $OUT/ppo_joint_g099_stop080.log 2>&1
echo "ppo joint gamma 0.99 stop080 exit $?" >> $OUT/status
