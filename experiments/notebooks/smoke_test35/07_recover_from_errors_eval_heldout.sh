#!/usr/bin/env bash
# Held out: the seed, the control and a PPO run's best checkpoint (when it is not step 0) and last adapter, greedy on
# 120 test questions (the same 120 for every policy; sized for Apple silicon) and all 120 of holdout_shift, answers
# stopped at their closing brace; and for the PPO checkpoints the forced probe (greedy, the 144 dev questions with a
# slip) and the 120 dev questions sampled once at T=1, as the seed was -- did PPO activate the recovery the seed
# planted? Probes fold the adapter into the weights for generation (--merge 1).
#   ./07_recover_from_errors_eval_heldout.sh
#   SEED_RUN=seed-n200-r100 CTRL_RUN=seed-n300-r0 PPO_RUN=ppo-n200-r100-turn-s0 ./...
#   TEST_N=0 ./...   all 240 test questions
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
R=07_recover_from_errors_runs
SEED_RUN=${SEED_RUN:-seed-n200-r100}
CTRL_RUN=${CTRL_RUN:-seed-n300-r0}
PPO_RUN=${PPO_RUN:-ppo-n200-r100-turn-s0}
TEST_N=${TEST_N:-120}
DEV_N=${DEV_N:-120}
probe() {  # adapter, out, split, temperature, forced, limit
    local tag="$3-T$4$([ "$6" != 0 ] && echo -n$6)-stop$([ "$5" = 1 ] && echo -forced)"
    [ -f $R/$2/$tag.json ] || $PY 07_recover_from_errors.py probe --split $3 --stop-at-json 1 --temperature $4 --forced $5 \
        --limit $6 --merge 1 --adapter $1 --out $2
}
for run in $SEED_RUN $CTRL_RUN; do
    probe $R/$run/epoch1 $run/epoch1 test 0 0 $TEST_N
    probe $R/$run/epoch1 $run/epoch1 holdout_shift 0 0 0
done
if [ -f $R/$PPO_RUN/ending.json ]; then
    BEST=$(python3 -c "import json; print(json.load(open('$R/$PPO_RUN/ending.json'))['best_step'])")
    CKPTS="adapter"
    [ "$BEST" != 0 ] && CKPTS="adapter-best $CKPTS"
    for ck in $CKPTS; do
        probe $R/$PPO_RUN/$ck $PPO_RUN/$ck test 0 0 $TEST_N
        probe $R/$PPO_RUN/$ck $PPO_RUN/$ck holdout_shift 0 0 0
        probe $R/$PPO_RUN/$ck $PPO_RUN/$ck dev 0 1 0        # the forced probe, greedy
        probe $R/$PPO_RUN/$ck $PPO_RUN/$ck dev 1 0 $DEV_N   # the 120 dev questions at T=1, as the seed was sampled
    done
fi
