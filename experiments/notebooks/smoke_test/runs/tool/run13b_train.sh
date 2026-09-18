#!/bin/bash
# 13b: 12's configuration for 200 steps under qwen-agent (FnCallAgent + PolicyChat), with the entropy
# bonus scoped to the answer turn and the run stopped if the sampled arithmetic fails (last-10 < 0.90).
# The harness change is decoding-neutral (runs/tool/sft_raw_dev_qwen_agent.json == sft_raw_dev.json, case
# for case), so against runs/ppo-on-tool-seed-warm-200 the one change that matters is the entropy scope.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../../membrane_grpo/.venv/bin/python ppo_ac.py --prompt-tool 1 --harness qwen_agent \
    --resume-from runs/tool-sft-raw --value-head-reset 1 --entropy-scope answer --numeric-stop 0.90 \
    --weights ABLATE --dense-rewards 1 --lam 0.95 --entropy-coef 0.005 --flat-credit-scale 3.0 \
    --critic-window 25 --steps 200 --eval-every 25 --out runs/ppo-tool-qwen-agent-scoped-200 > runs/tool/qa_scoped200.log 2>&1
echo "13b-train exit $?" >> runs/tool/rerun.status
