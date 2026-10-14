#!/usr/bin/env bash
# The seed: 06's dose ladder, with recovery episodes. Plain SFT (every token of the calls and the answer
# labelled, reasoning lines, one epoch, lr 1e-4, one step per 8 episodes, LoRA r16) on the first N reference
# episodes of 06's shuffle, every fourth replaced by its question's recovery episode (--recovery-every 4: one
# slip, the tool's own error, the fix; the slipped turn is context, not a label), N doubling from 150 (06's doses
# below 150 scored 0.04-0.11 sampled on dev). Each dose
# is sampled once per dev question at T=1 and judged by `alive`: 06's behaviours, every answer class right at
# least once, and one behaviour more -- after an erroring simulate call, that candidate simulated cleanly. The
# ladder stops at the first dose that is alive. That dose is then sampled twice per dev question (the probe
# PPO's value head is warm-started from), and so is 06's seed of the same dose, through this script's harness
# (the comparison: the same questions, no recovery episodes). The raw model is 06's: same prompt, same tools.
# No quality bar: PPO's job is the activation, and a dose PPO cannot activate gets the next dose.
#   ./07_error_recovery_seed.sh                  the ladder (doses already trained or probed are kept)
#   DOSES="600 1200" ./07_error_recovery_seed.sh  the ladder from a later rung
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
R=07_error_recovery_runs
R06=06_direction_and_compare_runs
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
ALIVE=
for N in ${DOSES:-150 300 600 1200}; do
    S=recovery-seed-n$N
    if [ ! -f $R/$S/epoch1/adapter_model.safetensors ]; then
        wait_card
        echo "$(date +%H:%M) seed $S"
        $PY 07_error_recovery.py seed --out $S --label full --reasoning 1 --epochs 1 --lr 1e-4 --accum 8 --lora-r 16 \
            --limit $N --recovery-every 4
    fi
    if [ ! -f $R/$S/epoch1/dev-T1-stop.json ]; then
        wait_card
        echo "$(date +%H:%M) probe $S"
        $PY 07_error_recovery.py probe --split dev --stop-at-json 1 --temperature 1 --adapter $R/$S/epoch1 --out $S/epoch1
    fi
    if $PY 07_error_recovery.py alive $R/$S/epoch1/dev-T1-stop.episodes.jsonl; then ALIVE=$N; break; fi
done
[ -n "$ALIVE" ] || { echo "$(date +%H:%M) no dose alive"; exit 1; }
echo "$(date +%H:%M) alive at $ALIVE"
for pair in "recovery-seed-n$ALIVE/epoch1 $R/recovery-seed-n$ALIVE/epoch1" "06-label-seed-n$ALIVE/epoch1 $R06/label-seed-n$ALIVE/epoch1"; do
    set -- $pair
    if [ ! -f $R/$1/dev-T1-x2-stop.json ]; then
        wait_card
        echo "$(date +%H:%M) probe x2 $1"
        $PY 07_error_recovery.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 --adapter $2 --out $1
    fi
done
echo "$(date +%H:%M) seed done: $ALIVE"
