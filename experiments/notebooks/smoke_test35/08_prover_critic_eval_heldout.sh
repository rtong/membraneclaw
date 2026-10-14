#!/usr/bin/env bash
# Held out, for an 08 PPO run's best checkpoint (when it is not step 0) and its last adapter -- the probes 07 ran for
# its own PPO run, so the two compare directly: greedy on the same 120 test questions and all 120 of holdout_shift, the
# forced probe (greedy, the 144 dev questions with a slip) and the 120 dev questions sampled once at T=1. Outputs go to
# <run>/probes-step<N>/ (adapter*/ directories are git-ignored; the probes are the record), one set when the best
# evaluation is the last step (the same weights). The seed's and the control's probes are 07's
# (07_recover_from_errors_runs/, read in place). Adapters folded into the weights for generation (--merge 1).
#   PPO_RUN=ppo-qwen3-1.7b-bo2-a1.5-s0 ./08_prover_critic_eval_heldout.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
R=08_prover_critic_runs
PPO_RUN=${PPO_RUN:?set PPO_RUN to an 08 PPO run directory}
TEST_N=${TEST_N:-120}
DEV_N=${DEV_N:-120}
probe() {  # adapter, out, split, temperature, forced, limit
    local tag="$3-T$4$([ "$6" != 0 ] && echo -n$6)-stop$([ "$5" = 1 ] && echo -forced)"
    [ -f $R/$2/$tag.json ] || $PY 08_prover_critic.py probe --split $3 --stop-at-json 1 --temperature $4 --forced $5 \
        --limit $6 --merge 1 --adapter $1 --out $2
}
[ -f $R/$PPO_RUN/ending.json ] || { echo "$PPO_RUN has not ended"; exit 1; }
read BEST STOP < <(python3 -c "import json; e = json.load(open('$R/$PPO_RUN/ending.json')); print(e['best_step'], e['stopped_at'])")
CKPTS="adapter:$STOP"
if [ "$BEST" != 0 ] && [ "$BEST" != "$STOP" ]; then CKPTS="adapter-best:$BEST $CKPTS"; fi
for pair in $CKPTS; do
    ck=${pair%%:*}
    out=$PPO_RUN/probes-step${pair##*:}
    probe $R/$PPO_RUN/$ck $out test 0 0 $TEST_N
    probe $R/$PPO_RUN/$ck $out holdout_shift 0 0 0
    probe $R/$PPO_RUN/$ck $out dev 0 1 0        # the forced probe, greedy
    probe $R/$PPO_RUN/$ck $out dev 1 0 $DEV_N   # the 120 dev questions at T=1, as the seed was sampled
done
