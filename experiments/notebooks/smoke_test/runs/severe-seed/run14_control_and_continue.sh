#!/bin/bash
# 14: (1) the supervision-only control -- the seed PPO started from, on the two splits nothing selected on.
#     10's control, asked of this line: how much of the result is the seed and how much is the policy gradient.
#     (2) 100 more steps from the credit run's best checkpoint (step 125), same configuration, the trained
#     value head carried across rather than reset.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
OUT=runs/severe-seed

for split in test holdout_shift; do
    $PY ppo_ac.py --prompt-tool 1 --harness qwen_agent --weights ABLATE --eval-only 1 \
        --resume-from runs/tool-seed-joint-slots-150/epoch3 --eval-split $split --max-vram-mib 13000 \
        --out $OUT/${split}_slots_epoch3.json > $OUT/${split}_slots_epoch3.log 2>&1
    echo "$split slots_epoch3 (seed only) exit $?" >> $OUT/status
done

echo "ppo severe-credit cont100 start $(date -Is)" >> $OUT/status
$PY ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/ppo-tool-severe-credit-best125 \
    --entropy-scope answer --numeric-stop 0.90 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --severe-credit-scale 3.0 --gamma 0.999 --gamma-scope token \
    --critic-window 25 --steps 100 --eval-every 25 --max-vram-mib 13700 \
    --out runs/ppo-tool-severe-credit-cont100 > $OUT/ppo_credit_cont100.log 2>&1
echo "ppo severe-credit cont100 exit $?" >> $OUT/status
