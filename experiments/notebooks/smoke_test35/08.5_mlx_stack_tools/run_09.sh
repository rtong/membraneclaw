#!/usr/bin/env bash
# 09 (08.5_mlx_stack_RECIPE.md section 11): seed-09-k2 (07's recipe -- 200 of 06's references + 100 recovery episodes,
# one epoch -- plus the k=2 targeted dose, 16 train2 episodes) -> its held-out probes (step 0) -> PPO on train_all
# (train + train2), constant lr 1e-6, 400 steps, held-out probes at steps 120/200/300/400.
#   08.5_mlx_stack_tools/run_09.sh       run it in the foreground; it backgrounds PPO itself
# ~35 min of SFT (MPS) and ~1 h of probes before PPO starts; then ~10 min per PPO step on an M3 Max 64 GB.
set -uo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd)
ST=$(cd "$TOOLS/.." && pwd)
cd "$ST" || exit 1
PY=../../membrane_grpo/.venv/bin/python
R=08.5_mlx_stack_runs
D6=06_direction_and_compare_data
SEED=${SEED:-seed-09-k2}
DOSE=${DOSE:-$TOOLS/09_dose/dose_k2.jsonl}
export RUN=${RUN:-ppo-mlx-qwen3.5-9b-l24-n32-lr1e6c-err0-delta-seed09k2-trainall-s0}
[ -f $D6/train_all.jsonl ] || cat $D6/train.jsonl $D6/train2.jsonl > $D6/train_all.jsonl
# SFT on MPS: the default allocator cap (~57% of memory) runs out around 330 episodes; nothing else runs meanwhile
[ -f $R/$SEED/epoch1/adapter_model.safetensors ] || PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 $PY 08.5_mlx_stack.py seed \
    --out $SEED --label full --reasoning 1 --epochs 1 --lr 1e-4 --accum 8 --lora-r 16 --limit 200 --recover 100 \
    --extra "$DOSE" --extra-split train2 || { echo "seed failed"; exit 1; }
[ -f $R/$SEED/probes-step0/dev-T1-n120-stop.json ] || PPO_RUN=$SEED CKPTS="epoch1:0" ./08.5_mlx_stack_eval_heldout.sh \
    || { echo "seed probes failed"; exit 1; }
cp $R/$SEED/probes-step0/dev-T1-n120-stop.json $R/$SEED/epoch1/dev-T1-n120-stop.json   # the critic's value_init_bias
SEED_PATH=$R/$SEED/epoch1 LR_ANNEAL=0 STEPS=400 STOP_AT=0 SPLIT=train_all "$TOOLS/launch_9b.sh"
sleep 15
RS=$(pgrep -f "08.5_mlx_stack_ppo_run.live.sh" | head -1)
[ -n "$RS" ] || { echo "no run script process"; exit 1; }
nohup caffeinate -dims -w "$RS" > /dev/null 2>&1 < /dev/null &
for s in 120 200 300 400; do
  nohup "$TOOLS/probes_at_step.sh" $s > "/tmp/probes_wait_${RUN}_$s.log" 2>&1 < /dev/null &
done
echo "PPO $RUN: run script $RS; log $ST/$R/$RUN/run.log"
