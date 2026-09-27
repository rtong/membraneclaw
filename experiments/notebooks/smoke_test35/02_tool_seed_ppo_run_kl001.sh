#!/bin/sh
# 02_tool_seed_ppo, run kl001-s0: base-s0's configuration plus a KL penalty towards the
# seed (kl_coef 0.01, in the per-token reward). From the tool seed (calls + answer open,
# epoch 3); critic on hidden_states[23], lr 9.19e-4, bias 0.4344, 25 critic-only steps;
# the <|im_end|> that ends a turn is out of the actor's loss; no entropy bonus; a turn ends
# where its first JSON object closes. After any exit before the run has ended (the
# watchdog exits a stalled step) it resumes from the last checkpoint.
cd "$(dirname "$0")"
RUN=02_tool_seed_ppo_runs/kl001-s0
PY=../../membrane_grpo/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
tries=0
while [ ! -e "$RUN/ending.json" ] && [ "$tries" -lt 8 ]; do
  tries=$((tries + 1))
  if [ -e "$RUN/last/state.pt" ]; then
    echo "=== resume attempt $tries $(date)" >> $RUN.log
    $PY 02_tool_seed_ppo.py train --run kl001-s0 --resume --max-vram-mib 13000 >> $RUN.log 2>&1
  else
    rm -rf "$RUN"
    echo "=== start attempt $tries $(date)" >> $RUN.log
    $PY 02_tool_seed_ppo.py train --run kl001-s0 --start-from 02_tool_seed_ppo_runs/tool-seed-anchor/epoch3 \
      --value-layer 23 --value-lr 0.0009189867391542886 --value-init-bias 0.4344 --critic-warmup 25 \
      --freeze-turn-ends 1 --entropy-coef 0 --stop-at-json 1 --kl-coef 0.01 \
      --max-vram-mib 13000 >> $RUN.log 2>&1
  fi
  [ -e "$RUN/ending.json" ] || sleep 60
done
