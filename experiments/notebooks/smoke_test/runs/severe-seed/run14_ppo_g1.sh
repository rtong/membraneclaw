#!/bin/bash
# 14, step 2: 13b's PPO from the severe seed, gamma 1.0, 200 steps.
#
# run13b_train.sh with two changes, both forced by the new start: --resume-from the chosen seed
# (runs/tool-seed-severe-150-calls/epoch1, see run14_seed_calls.sh), and --value-init-bias 0.6826, that
# seed's held-out dev reward under ABLATE (runs/severe-seed/dev_calls_epoch1.episodes.jsonl rescored;
# the same rescoring gives tool-sft-raw's 0.6521 exactly). Everything else is 13b, so the comparison is
# runs/ppo-tool-qwen-agent-scoped-200 (seed 0, EM 0.555@200) and its -s1 (0.560@200).
#
# What the seed starts from, at temperature 1.0 over 600 train samples: severe action said 16 times, 5 on a
# case whose answer it is (of 108); organic_fouling said 2 times, 0 right (of 80). 13b's start: 1 / 0 and
# 48 / 27. Question: does PPO lift two labels that are barely alive?
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/severe-seed

echo "ppo g1 start $(date -Is)" >> $OUT/status

../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/tool-seed-severe-150-calls/epoch1 --value-head-reset 1 --value-init-bias 0.6826 \
    --entropy-scope answer --numeric-stop 0.90 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 200 --eval-every 25 --max-vram-mib 13700 \
    --out runs/ppo-tool-severe-seed-200 > $OUT/ppo_g1.log 2>&1
echo "ppo g1 exit $?" >> $OUT/status
