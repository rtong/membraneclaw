#!/usr/bin/env bash
# 08.5: the Qwen3.5-9B-critic PPO run on the MLX stack, with every infra switch on (08.5_mlx_stack_RECIPE.md).
#   08.5_mlx_stack_tools/launch_9b.sh            a new run
#   08.5_mlx_stack_tools/launch_9b.sh resume     continue from the run's last/ (RESUME=1)
# Any setting can be overridden from the environment, e.g.  UPDATE_MICRO_TOKENS=10000 .../launch_9b.sh resume
# Run it in the foreground: it backgrounds the run itself (a zsh `&` would nice the run through BG_NICE).
set -u
ST=$(cd "$(dirname "$0")/.." && pwd)    # experiments/notebooks/smoke_test35
cd "$ST" || exit 1
export SEED=${SEED:-0}
export CRITIC=${CRITIC:-qwen3.5-9b} CRITIC_LAYER=${CRITIC_LAYER:-24} CRITIC_RIDGE=${CRITIC_RIDGE:-1.0} CRITIC_L2=${CRITIC_L2:-0.1}
export ALPHA=${ALPHA:-0} PROVER_K=${PROVER_K:-4}
export RUN=${RUN:-ppo-mlx-qwen3.5-9b-l24-n32-lr1e6-err0-delta-s$SEED}
export WARM_FROM=${WARM_FROM-}                # empty: the run's own 25 warm-up steps
export PROMPTS=${PROMPTS:-32} LR=${LR:-1e-6} ERR_CALL_CREDIT=${ERR_CALL_CREDIT:-0} CALL_CREDIT=${CALL_CREDIT:-delta}
export STEPS=${STEPS:-400} STOP_AT=${STOP_AT:-300} GENERATOR=${GENERATOR:-mlx-batch}
export EVAL_EVERY=${EVAL_EVERY:-20} EVAL_CASES=${EVAL_CASES:-120} PATIENCE=${PATIENCE:-100} STALL_AT=${STALL_AT:-0}
export SAVE_EVERY=${SAVE_EVERY:-5} STEP_WATCHDOG=${STEP_WATCHDOG:-14400}
# the infra switches (08.5_mlx_stack.py PPO_DEFAULTS: prefix_share, update_micro, forward_batch, fast_rule,
# fast_rule_inv, update_micro_tokens)
export PREFIX_SHARE=${PREFIX_SHARE:-1} UPDATE_MICRO=${UPDATE_MICRO:-2} FORWARD_BATCH=${FORWARD_BATCH:-2}
export FAST_RULE=${FAST_RULE:-1} FAST_RULE_INV=${FAST_RULE_INV:-1} UPDATE_MICRO_TOKENS=${UPDATE_MICRO_TOKENS:-12000}
[ "${1:-}" = resume ] && export RESUME=1
# a frozen copy: bash reads a running script as it goes, so later edits to the repo's script must not reach this run
cp 08.5_mlx_stack_ppo_run.sh 08.5_mlx_stack_ppo_run.live.sh
nohup ./08.5_mlx_stack_ppo_run.live.sh > "/tmp/$RUN.out" 2>&1 < /dev/null &
echo "run script pid $!   log: $ST/08.5_mlx_stack_runs/$RUN/run.log"
