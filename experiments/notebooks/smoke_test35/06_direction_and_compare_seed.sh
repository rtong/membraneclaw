#!/usr/bin/env bash
# The seed: the smallest SFT dose that is alive, not 05's label seed. Plain SFT with every token of the
# calls and the answer labelled (--label full) on the first N reference episodes of one shuffle, one epoch,
# lr 1e-4, one step per 8 episodes, LoRA r16, N doubling from 10 to 1200 (every reference episode). Each
# dose is sampled once per dev question at T=1 and judged by `alive`: every behaviour the task needs seen
# at least once, every one of the 16 answer classes right at least once. The raw model is dose 0. There is
# no quality bar: PPO's job is the activation, and a dose PPO cannot activate gets the next dose.
#   ./06_direction_and_compare_seed.sh              the sweep (doses already trained or probed are kept)
#   DOSE=1200 ./06_direction_and_compare_seed.sh    one dose, sampled twice per dev question at T=1: the
#                                                   probe PPO's value head is warm-started from
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
R=06_direction_and_compare_runs
if [ -n "${DOSE:-}" ]; then
    $PY 06_direction_and_compare.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 \
        --adapter $R/label-seed-n$DOSE/epoch1 --out label-seed-n$DOSE/epoch1
    exit 0
fi
[ -f $R/raw/dev-T1-stop.json ] || $PY 06_direction_and_compare.py probe --split dev --stop-at-json 1 --temperature 1 --out raw
$PY 06_direction_and_compare.py alive $R/raw/dev-T1-stop.episodes.jsonl || true
for N in ${DOSES:-10 20 40 80 150 300 600 1200}; do
    [ -f $R/label-seed-n$N/epoch1/adapter_model.safetensors ] || $PY 06_direction_and_compare.py seed --out label-seed-n$N \
        --label full --reasoning 1 --epochs 1 --lr 1e-4 --accum 8 --lora-r 16 --limit $N
    [ -f $R/label-seed-n$N/epoch1/dev-T1-stop.json ] || $PY 06_direction_and_compare.py probe --split dev --stop-at-json 1 \
        --temperature 1 --adapter $R/label-seed-n$N/epoch1 --out label-seed-n$N/epoch1
    $PY 06_direction_and_compare.py alive $R/label-seed-n$N/epoch1/dev-T1-stop.episodes.jsonl || true
done
