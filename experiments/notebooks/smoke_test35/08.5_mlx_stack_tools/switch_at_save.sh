#!/usr/bin/env bash
# 08.5: wait for the run's save at step >= TARGET (last/state_mlx.json, every SAVE_EVERY steps), stop the run, and
# resume it with the changed settings (launch_9b.sh resume; the trainer records them in config-resume-step<N>.json).
#   FAST_RULE_INV=1 UPDATE_MICRO_TOKENS=12000 08.5_mlx_stack_tools/switch_at_save.sh 85
set -u
TOOLS=$(cd "$(dirname "$0")" && pwd)
ST=$(cd "$TOOLS/.." && pwd)
RUN=${RUN:-ppo-mlx-qwen3.5-9b-l24-n32-lr1e6-err0-delta-s0}
R=$ST/08.5_mlx_stack_runs/$RUN
TARGET=$1
while true; do
    s=$(python3 -c "import json; print(json.load(open('$R/last/state_mlx.json'))['step'])" 2>/dev/null || echo 0)
    [ "$s" -ge "$TARGET" ] && break
    sleep 15
done
date "+%H:%M:%S saved state step $s"
pkill -f "08.5_mlx_stack_ppo_run"; pkill -f "08.5_mlx_stack.py ppo --out $RUN"; sleep 5
pgrep -f "08.5_mlx_stack.py ppo" && { echo "still running?"; exit 1; }
RUN=$RUN "$TOOLS/launch_9b.sh" resume
sleep 60
tr '\r' '\n' < "$R/run.log" | grep -E "resumed|Traceback|Error" | tail -3
