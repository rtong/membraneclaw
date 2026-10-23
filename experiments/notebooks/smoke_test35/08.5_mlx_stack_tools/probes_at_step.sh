#!/usr/bin/env bash
# 08.5: the held-out probes of a run's adapter-step<N> as soon as it is saved (an evaluation step), beside the training
# (probe: the 0.8B policy, MLX cache 2 GB), then both scorings offline (rescore_probes.py), paired with 07c+D37's
# probes of the same step when they exist.
#   08.5_mlx_stack_tools/probes_at_step.sh 200
set -u
TOOLS=$(cd "$(dirname "$0")" && pwd)
ST=$(cd "$TOOLS/.." && pwd)
RUN=${RUN:-ppo-mlx-qwen3.5-9b-l24-n32-lr1e6-err0-delta-s0}
R=$ST/08.5_mlx_stack_runs/$RUN
STEP=$1
while [ ! -f "$R/adapter-step$STEP/adapter_config.json" ]; do sleep 60; done
sleep 30
date '+%m-%d %H:%M probes start'
cd "$ST" && CKPTS="adapter-step$STEP:$STEP" PPO_RUN=$RUN ./08.5_mlx_stack_eval_heldout.sh > "/tmp/probes_${RUN}_${STEP}.log" 2>&1
echo "probe script exit $?"; date '+%m-%d %H:%M probes done'
../../membrane_grpo/.venv/bin/python "$TOOLS/rescore_probes.py" "08.5_mlx_stack_runs/$RUN/probes-step$STEP" \
    "08.5_mlx_stack_runs/ppo-mlx-policy-d37-s0/probes-step$STEP" 2>&1 | grep -v "Fetching\|PyTorch"
