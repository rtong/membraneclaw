#!/usr/bin/env bash
# User 2026-10-22 ("do it"): the two seeds on the held-out splits, as the wrap-up probed their step-200 checkpoints
# (test 240 and holdout_shift 120, T=1 x2 then greedy, each re-scored -err0), so PPO's gain over its seed can be stated
# on questions no gate or checkpoint choice has seen. The maskend seed (the new run's) first, T=1 before greedy.
#   nohup setsid bash 07_error_recovery_diag/seed_heldout.sh > 07_error_recovery_runs/seed_heldout.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
R=07_error_recovery_runs
D=07_error_recovery_diag
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
# seed dir : its PPO run (both step-200 checkpoints were probed by the wrap-up)
PAIRS="recovery-seed-n300-maskend/epoch1:ppo-n300-maskend-turn-lr2e6-err0-s0 recovery-seed-n300/epoch1:ppo-n300-turn-lr2e6-s0"
for spec in "1 2" "0 1"; do
    T=${spec% *}; N=${spec#* }
    for pair in $PAIRS; do
        SEED=${pair%%:*}
        for split in test holdout_shift; do
            tag=$split-T$T$([ "$N" -gt 1 ] && echo -x$N)-stop
            if [ ! -f $R/$SEED/$tag.json ]; then
                wait_card; echo "$(date '+%m-%d %H:%M') probe $SEED $tag"
                $PY 07_error_recovery.py probe --split $split --stop-at-json 1 --temperature $T --samples $N \
                    --adapter $R/$SEED --out $SEED
            fi
            [ -f $R/$SEED/$tag.json ] || { echo "$(date '+%m-%d %H:%M') no $tag for $SEED"; exit 1; }
            [ -f $R/$SEED/$tag-err0.json ] || $PY $D/rescore_err0.py $R/$SEED $tag
        done
    done
done
for pair in $PAIRS; do
    SEED=${pair%%:*}; RUN=${pair#*:}
    for tag in test-T1-x2-stop holdout_shift-T1-x2-stop test-T0-stop holdout_shift-T0-stop; do
        echo "== $tag-err0: $RUN step 200 vs its seed"
        $PY $D/compare_runs.py $R/$SEED $R/$RUN/step200 $tag-err0 seed step200
    done
done
echo "$(date '+%m-%d %H:%M') seed held-out probes done"
