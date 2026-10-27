#!/usr/bin/env bash
# Three more 300-episode seeds, each the ladder's alive dose (07_error_recovery_seed.sh) with one change to the recovery
# episodes, then dev T=1 x2 (re-scored -err0) and PR #33's forced probe (144 dev unit slips, greedy):
#   recovery-seed-n300-unit           unconverted slips only (the 39 factor-slip swaps moved to the nearest questions
#                                     with an unconverted slip: 75 unconverted)
#   recovery-seed-n300-maskend        an unconverted slip's conversion turn (the one that leaves the slipped input out)
#                                     keeps its calls as labels, its closing <|im_end|> is not one (26 such turns)
#   recovery-seed-n300-unit-maskend   both (57 such turns)
# The PPO runs from recovery-seed-n300-maskend use it with --err-call-credit 0 (07_error_recovery_ppo_run.sh ERR_CREDIT=0).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
R=07_error_recovery_runs
D=07_error_recovery_diag
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
for spec in "recovery-seed-n300-unit:unconverted:0" "recovery-seed-n300-maskend::2" "recovery-seed-n300-unit-maskend:unconverted:2"; do
    IFS=: read -r S K M <<< "$spec"
    if [ ! -f $R/$S/epoch1/adapter_model.safetensors ]; then
        wait_card; echo "$(date +%H:%M) seed $S"
        $PY 07_error_recovery.py seed --out $S --label full --reasoning 1 --epochs 1 --lr 1e-4 --accum 8 --lora-r 16 \
            --limit 300 --recovery-every 4 --recovery-kinds "$K" --recovery-mask-omission $M
    fi
    [ -f $R/$S/epoch1/adapter_model.safetensors ] || { echo "$(date +%H:%M) no adapter for $S"; exit 1; }
    if [ ! -f $R/$S/epoch1/dev-T1-x2-stop.json ]; then
        wait_card; echo "$(date +%H:%M) probe x2 $S"
        $PY 07_error_recovery.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 --adapter $R/$S/epoch1 --out $S/epoch1
    fi
    [ -f $R/$S/epoch1/dev-T1-x2-stop-err0.json ] || $PY $D/rescore_err0.py $R/$S/epoch1 dev-T1-x2-stop
    if [ ! -f $R/$S/epoch1/dev-T0-stop-forced.json ]; then
        wait_card; echo "$(date +%H:%M) forced probe $S"
        $PY 07_error_recovery.py probe --split dev --temperature 0 --stop-at-json 1 --forced 07_error_recovery_data/dev_recovery_pr33.jsonl \
            --adapter $R/$S/epoch1 --out $S/epoch1
    fi
done
echo "$(date +%H:%M) seed variants done"
