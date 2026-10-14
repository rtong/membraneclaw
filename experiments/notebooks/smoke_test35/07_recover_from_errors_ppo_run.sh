#!/usr/bin/env bash
# PPO from 07's seed with 06's last configuration -- the one that did not run away: the turn as the unit of credit
# (one assistant message is one action; lambda 0.95 per turn), the actor's step at 2e-6, KL 0.01 to the seed, critic
# warm-up 25, the value head on layer 23 warm-started at the seed's own sampled dev reward, the turn ends and the
# answer's scaffolding out of the actor's loss, Adam eps 1e-6, answers stopped at their closing brace, 3,072 policy
# tokens; the tool gate and the no-regression gate off, the answer-collapse rule on. Only the seed differs from 06's
# ppo-n600-open-turn-lr2e6-s0: 300 episodes, a third of them recoveries. Every step's 8 episodes recorded. Sized for
# Apple silicon, where a step takes ~7 minutes (06's card: 3): greedy evaluations on 60 dev questions every 40 steps
# (06: 120 every 25) and STEPS 120 for a first look (06: 200); RESUME=1 STEPS=<more> continues from last/, and
# EVAL_EVERY=25 EVAL_CASES=120 STEPS=200 is 06's schedule. Runs until a rule, convergence (8 evaluations without a new
# best) or STEPS. On CUDA it waits for the card to have room, as 06's did; on MPS it runs as it is.
# 07's early stops, so a run that is not working ends within hours (PPO_DEFAULTS in the .py has the rules): the reward
# gate (the last 20 steps' sampled reward 0.08 below the warm-up's at 5 steps in a row, from actor step 20), the stall
# gate (at actor steps 40 and 80, that reward not above the warm-up's), 06's no-regression gate on (two greedy
# evaluations 0.05 below step 0), and the answer-collapse rule. A run restarted from scratch takes its step-0
# evaluation from its own eval.jsonl when the start and the evaluation are the same (REUSE_EVAL0=1).
#   ./07_recover_from_errors_ppo_run.sh                                     ppo-n200-r100-turn-s0 from seed-n200-r100
#   SEED_RUN=seed-n300-r0 RUN=ppo-n300-r0-turn-s0 ./...                     the same from the control
#   LR=1e-5 RUN=ppo-n200-r100-turn-lr1e5-s0 ./...                           06's default step (ran away in 06 at 50 steps)
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
SEED_RUN=${SEED_RUN:-seed-n200-r100}     # the seed the run starts from
RUN=${RUN:-ppo-n200-r100-turn-s0}        # the run directory under 07_recover_from_errors_runs/
SEED_CKPT=${SEED_CKPT:-epoch1}
ADV_UNIT=${ADV_UNIT:-turn}
LAM=${LAM:-0.95}
LR=${LR:-2e-6}
SAMPLE_EVERY=${SAMPLE_EVERY:-1}
ANSWER_STOP_REL=${ANSWER_STOP_REL:-0.5}
REWARD_STOP_DROP=${REWARD_STOP_DROP:-0.08}   # the reward gate; 0 turns it off
STALL_AT=${STALL_AT:-40,80}                  # the stall gate's checkpoints, in actor steps; 0 turns it off
REGRESS_MARGIN=${REGRESS_MARGIN:-0.05}       # 06's no-regression gate (06's runs: 1, off)
REUSE_EVAL0=${REUSE_EVAL0:-1}
STEPS=${STEPS:-120}
EVAL_EVERY=${EVAL_EVERY:-40}
EVAL_CASES=${EVAL_CASES:-60}
DEV_N=${DEV_N:-120}                      # the seed's dev probe the value head is warm-started from (07_recover_from_errors_seed.sh)
RESUME=${RESUME:-0}
R=07_recover_from_errors_runs
BIAS=$(python3 -c "import json; print(round(json.load(open('$R/$SEED_RUN/$SEED_CKPT/dev-T1-n$DEV_N-stop.json'))['overall']['reward'], 4))")
ARGS="--out $RUN --start-from $R/$SEED_RUN/$SEED_CKPT --value-init-bias $BIAS --steps $STEPS --eval-every $EVAL_EVERY --eval-cases $EVAL_CASES --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel $ANSWER_STOP_REL --regress-margin $REGRESS_MARGIN --reward-stop-drop $REWARD_STOP_DROP --stall-at $STALL_AT --reuse-eval0 $REUSE_EVAL0 --lam $LAM --adv-unit $ADV_UNIT --sample-every $SAMPLE_EVERY --lr $LR"
wait_card() {
    local smi; smi=$(command -v nvidia-smi || ls /usr/lib/wsl/lib/nvidia-smi 2>/dev/null || true)
    [ -n "$smi" ] || return 0
    while [ "$($smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done
}
wait_card
if [ "$RESUME" = 1 ] && [ -f $R/$RUN/last/state.pt ]; then
    rm -f $R/$RUN/ending.json
    $PY 07_recover_from_errors.py ppo $ARGS --resume 1
else
    $PY 07_recover_from_errors.py ppo $ARGS
fi
for attempt in 1 2 3 4 5 6 7 8; do
    [ -f $R/$RUN/ending.json ] && exit 0
    wait_card
    if [ -f $R/$RUN/last/state.pt ]; then
        $PY 07_recover_from_errors.py ppo $ARGS --resume 1
    else
        $PY 07_recover_from_errors.py ppo $ARGS
    fi
done
