#!/bin/bash
# 14, step 5: 3x credit for a correct severe action, from a seed where both labels are alive.
#
# Why. From runs/tool-seed-joint-150/epoch1, PPO (gamma 0.999) took organic_fouling to 29 of 29 right and the
# severe action to zero: the seed said severe on 27% of its emissions correctly, so on average the label cost
# reward and the policy dropped it. `09`'s `flat` was the same shape and a 3x credit took its recall 0.000 ->
# 0.984. `--severe-credit-scale 3.0` pays 0.45 instead of 0.15 for a correct severe action; every other case is
# untouched, and `reward.score` -- what every eval reports -- does not change, so EM stays comparable.
#
# Start: runs/tool-seed-joint-slots-150/epoch3, the both-slots seed. At temperature 1.0 over 600 train samples it
# says severe 78 times, 21 right (of 108), and organic_fouling 96 times, 43 right (of 80); teacher-forced call
# entropy 0.000227, below runs/tool-sft-raw's own 0.000252. It is the only measured seed with both labels alive,
# and it fails one of the seed gates: severe on 11.6% of the samples whose answer is not severe, against a 10%
# bar. --value-init-bias 0.6736 is its own held-out dev reward under ABLATE.
#
# gamma 0.999, the value that ran 200 steps without stopping (EM 0.635). Gates as before: step 40 = calls
# entropy <= 0.001 and no numeric stop; critic 50-74 ahead of the position clock on >= 50% of steps.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/severe-seed

echo "ppo severe-credit start $(date -Is)" >> $OUT/status
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/tool-seed-joint-slots-150/epoch3 --value-head-reset 1 --value-init-bias 0.6736 \
    --entropy-scope answer --numeric-stop 0.90 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --severe-credit-scale 3.0 --gamma 0.999 --gamma-scope token \
    --critic-window 25 --steps 200 --eval-every 25 --max-vram-mib 13700 \
    --out runs/ppo-tool-severe-credit-200 > $OUT/ppo_severe_credit.log 2>&1
echo "ppo severe-credit exit $?" >> $OUT/status
