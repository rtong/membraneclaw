#!/bin/bash
# 14, step 2: 13b's PPO from the joint seed, gamma 1.0, 200 steps.
#
# run13b_train.sh with two changes, both forced by the new start: --resume-from the joint seed's first epoch
# (runs/tool-seed-joint-150/epoch1, see run14_seed_joint.sh) and --value-init-bias 0.6743, that seed's
# held-out dev reward under ABLATE (dev_joint_epoch1.episodes.jsonl rescored; the same rescoring gives
# tool-sft-raw 0.6522). Everything else is 13b: the comparison is runs/ppo-tool-qwen-agent-scoped-200 (seed 0,
# EM 0.555@200) and its -s1 (0.560@200).
#
# The start, at temperature 1.0 over 600 train samples: severe action said 77 times, 21 on a case whose answer
# it is (of 108); organic_fouling said once, right (of 80). 13b's start: 1 / 0 and 48 / 27. Teacher-forced
# call entropy on dev 0.000465 (13b's start 0.000252).
# Gate at step 40: the run has not stopped on sampled arithmetic, and the call region's entropy averages
# <= 0.001 over steps 0-39 (13b: 0.00002-0.0005).
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/severe-seed

echo "ppo joint g1 start $(date -Is)" >> $OUT/status
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/tool-seed-joint-150/epoch1 --value-head-reset 1 --value-init-bias 0.6743 \
    --entropy-scope answer --numeric-stop 0.90 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 200 --eval-every 25 --max-vram-mib 13700 \
    --out runs/ppo-tool-joint-seed-200 > $OUT/ppo_joint_g1.log 2>&1
echo "ppo joint g1 exit $?" >> $OUT/status
