#!/usr/bin/env bash
# The probes the notebook reads that the runs' own scripts do not make, each skipped when its file exists, each
# re-scored -err0 (rescore_err0.py):
#   dev T=1 x2 on the step-200 checkpoints of the three first PPO runs (07_error_recovery_ppo_run.sh: from the seed,
#   the same centred, from recovery-seed-n300-maskend with ERR_CREDIT=0);
#   PR #33's forced probe (144 dev unit slips, greedy) on 06's 300-episode seed, the recipe's seed and those step-200s;
#   test and holdout_shift, T=1 x2 and greedy, on the two uncentred step-200s.
# Elsewhere: 07_error_recovery_seed.sh (the ladder; dev T=1 x2 of 07's and 06's 300-episode seeds), seed_variants.sh,
# seed_heldout.sh (the two seeds held out), to400.sh, lnbatch.sh, exp1_td.sh, exp3_actorlam.sh, prov_ppo.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
R=07_error_recovery_runs
D=07_error_recovery_diag
F=07_error_recovery_data/dev_recovery_pr33.jsonl
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
probe() {  # adapter, out dir (under the runs dir), split, T, samples
    local tag=$3-T$4$([ "$5" -gt 1 ] && echo -x$5)-stop
    if [ ! -f $R/$2/$tag.json ]; then
        wait_card; echo "$(date +%H:%M) probe $2 $tag"
        $PY 07_error_recovery.py probe --split $3 --stop-at-json 1 --temperature $4 --samples $5 --adapter $1 --out $2
    fi
    [ -f $R/$2/$tag-err0.json ] || $PY $D/rescore_err0.py $R/$2 $tag
}
forced() {  # adapter, out dir
    if [ ! -f $R/$2/dev-T0-stop-forced.json ]; then
        wait_card; echo "$(date +%H:%M) forced probe $2"
        $PY 07_error_recovery.py probe --split dev --temperature 0 --stop-at-json 1 --forced $F --adapter $1 --out $2
    fi
}
for RUN in ppo-n300-turn-lr2e6-s0 ppo-n300-turn-lr2e6-center-s0 ppo-n300-maskend-turn-lr2e6-err0-s0; do
    probe $R/$RUN/adapter-step200 $RUN/step200 dev 1 2
done
forced 06_direction_and_compare_runs/label-seed-n300/epoch1 06-label-seed-n300/epoch1
forced $R/recovery-seed-n300/epoch1 recovery-seed-n300/epoch1
for RUN in ppo-n300-turn-lr2e6-s0 ppo-n300-turn-lr2e6-center-s0 ppo-n300-maskend-turn-lr2e6-err0-s0; do
    forced $R/$RUN/adapter-step200 $RUN/step200
done
for spec in "1 2" "0 1"; do
    for split in test holdout_shift; do
        for RUN in ppo-n300-maskend-turn-lr2e6-err0-s0 ppo-n300-turn-lr2e6-s0; do
            probe $R/$RUN/adapter-step200 $RUN/step200 $split $spec
        done
    done
done
echo "$(date +%H:%M) probes done"
