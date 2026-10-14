#!/usr/bin/env bash
# PPO from 07's seed with 07's configuration, the critic replaced: a separate pretrained model, frozen, read before each
# turn, its heads v (PPO's baseline) and q (the base policy's success) refitted on the last 25 steps at every step; and
# every turn's advantage gains ALPHA x its progress under the prover BoK(pi) (Setlur et al. 2024). The critic, its
# layer and penalties, K and ALPHA default to critic-fit's choice (08_prover_critic_critic_fit.json; run
# 08_prover_critic_critic_fit.sh first). Everything else is 07's ppo-n200-r100-turn-s0 -- the turn as the unit of
# credit, lambda 0.95 per turn, lr 2e-6, KL 0.01, 120 steps with greedy evaluations of 60 dev questions every 40, the
# early stops (reward gate, stall gate at actor steps 40 and 80, no-regression gate, answer collapse) -- and its first
# 25 steps are 07's own (WARM_FROM): the same seed draws the same questions, and in those steps the policy is not
# updated, so their episodes fill the new critic's window and 07's step-0 evaluation is this run's. The run is then
# step for step comparable with 07's: the same questions at every step, evaluations at the same steps.
#   ./08_prover_critic_ppo_run.sh
#   ALPHA=0 RUN=ppo-qwen3-1.7b-a0-s0 ./...        the separate critic alone, no prover (an ablation)
#   SEED_PATH=08_prover_critic_runs/seed-rt-n300-e4/epoch1 WARM_FROM= RUN=ppo-rt-qwen3-4b-bo4-a0.8-s0 ./...
#                                                 from the other 07's seed recipe (08_prover_critic_seed.sh): its own
#                                                 25-step warm-up and step-0 evaluation, no run to take them from
#   RESUME=1 ./...                                continue from last/
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
R07=07_recover_from_errors_runs
choice() { python3 -c "import json; print(json.load(open('08_prover_critic_critic_fit.json'))['choice']['$1'])"; }
CRITIC=${CRITIC:-$(choice critic)}
PROVER_K=${PROVER_K:-$(choice prover_k)}
ALPHA=${ALPHA:-$(choice alpha)}
RUN=${RUN:-ppo-$CRITIC-bo$PROVER_K-a$ALPHA-s0}  # the run directory under 08_prover_critic_runs/
SEED_RUN=${SEED_RUN:-seed-n200-r100}            # 07's seed
SEED_PATH=${SEED_PATH:-$R07/$SEED_RUN/epoch1}   # or 08's own seed, e.g. 08_prover_critic_runs/seed-rt-n300-e4/epoch1
WARM_FROM=${WARM_FROM-$R07/ppo-n200-r100-turn-s0}  # WARM_FROM= (empty) for a seed no earlier run started from
LR=${LR:-2e-6}
REWARD_STOP_DROP=${REWARD_STOP_DROP:-0.08}
STALL_AT=${STALL_AT:-40,80}
REGRESS_MARGIN=${REGRESS_MARGIN:-0.05}
STEPS=${STEPS:-120}
EVAL_EVERY=${EVAL_EVERY:-40}
EVAL_CASES=${EVAL_CASES:-60}
STEP_WATCHDOG=${STEP_WATCHDOG:-3600}    # 07: 1800 and 7200; doubled, for steps and evaluations slowed by another GPU job
EVAL_WATCHDOG=${EVAL_WATCHDOG:-14400}
RESUME=${RESUME:-0}
R=08_prover_critic_runs
BIAS=$(python3 -c "import json; print(round(json.load(open('$SEED_PATH/dev-T1-n120-stop.json'))['overall']['reward'], 4))")
ARGS="--out $RUN --start-from $SEED_PATH --value-init-bias $BIAS --steps $STEPS --eval-every $EVAL_EVERY --eval-cases $EVAL_CASES --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin $REGRESS_MARGIN --reward-stop-drop $REWARD_STOP_DROP --stall-at $STALL_AT --lam 0.95 --adv-unit turn --sample-every 1 --lr $LR --critic $CRITIC --prover-k $PROVER_K --prover-alpha $ALPHA --step-watchdog $STEP_WATCHDOG --eval-watchdog $EVAL_WATCHDOG"
[ -n "$WARM_FROM" ] && ARGS="$ARGS --warm-from $WARM_FROM"
mkdir -p $R/$RUN
if [ "$RESUME" = 1 ] && [ -f $R/$RUN/last/state.pt ]; then
    rm -f $R/$RUN/ending.json
    $PY 08_prover_critic.py ppo $ARGS --resume 1 2>&1 | tee -a $R/$RUN/run.log
else
    $PY 08_prover_critic.py ppo $ARGS 2>&1 | tee -a $R/$RUN/run.log
fi
for attempt in 1 2 3 4 5 6 7 8; do  # a crash resumes from the last evaluation's state (or starts again before one)
    [ -f $R/$RUN/ending.json ] && exit 0
    if [ -f $R/$RUN/last/state.pt ]; then
        $PY 08_prover_critic.py ppo $ARGS --resume 1 2>&1 | tee -a $R/$RUN/run.log
    else
        $PY 08_prover_critic.py ppo $ARGS 2>&1 | tee -a $R/$RUN/run.log
    fi
done
