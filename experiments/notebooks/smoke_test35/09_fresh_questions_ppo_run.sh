#!/usr/bin/env bash
# 09: PPO from seed-09-k2 on train_all (06's train + 09's train2 = 3,599 questions, no repeats), constant lr 1e-6
# (LR_ANNEAL=0), 400 steps, 32 questions per step, frozen Qwen3.5-9B critic at layer 24 (ridge 1.0, l2 0.1), err0
# scoring, delta call credit, D37 update (minibatch adv norm, 4 minibatches x 4 epochs, grad clip 0.5, Adam eps 1e-5).
# Auto-selects the MLX stack on macOS (TRAINER=mlx GENERATOR=mlx-batch) and PyTorch on Linux/CUDA (TRAINER=torch
# GENERATOR=torch); any knob can be overridden via environment variables.
#   ./09_fresh_questions_ppo_run.sh
#   RESUME=1 ./09_fresh_questions_ppo_run.sh
#   SEED=1 ./09_fresh_questions_ppo_run.sh
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=${PY:-../../membrane_grpo/.venv/bin/python}
R=09_fresh_questions_runs
SEED=${SEED:-0}
SEED_RUN=${SEED_RUN:-seed-09-k2}
SEED_PATH=${SEED_PATH:-$R/$SEED_RUN/epoch1}
WARM_FROM=${WARM_FROM-}
CRITIC=${CRITIC:-qwen3.5-9b}
CRITIC_LAYER=${CRITIC_LAYER:-24}
CRITIC_RIDGE=${CRITIC_RIDGE:-1.0}
CRITIC_L2=${CRITIC_L2:-0.1}
PROVER_K=${PROVER_K:-4}
ALPHA=${ALPHA:-0}
SPLIT=${SPLIT:-train_all}
PROMPTS=${PROMPTS:-32}
LR=${LR:-1e-6}
LR_ANNEAL=${LR_ANNEAL:-0}
LR_ANNEAL_START=${LR_ANNEAL_START:-0}
STEPS=${STEPS:-400}
STOP_AT=${STOP_AT:-0}
ERR_CALL_CREDIT=${ERR_CALL_CREDIT:-0}
CALL_CREDIT=${CALL_CREDIT:-delta}
EVAL_EVERY=${EVAL_EVERY:-20}
EVAL_CASES=${EVAL_CASES:-120}
PATIENCE=${PATIENCE:-100}
REWARD_STOP_DROP=${REWARD_STOP_DROP:-0.08}
STALL_AT=${STALL_AT:-0}
REGRESS_MARGIN=${REGRESS_MARGIN:-0.05}
SAVE_EVERY=${SAVE_EVERY:-5}
STEP_WATCHDOG=${STEP_WATCHDOG:-14400}
EVAL_WATCHDOG=${EVAL_WATCHDOG:-14400}
ADV_NORM=${ADV_NORM:-minibatch}
MINIBATCHES=${MINIBATCHES:-4}
TARGET_KL=${TARGET_KL:-0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
ADAM_EPS=${ADAM_EPS:-1e-5}
EVAL_GENERATOR=${EVAL_GENERATOR:-}

if [ "$(uname -s)" = Darwin ]; then
    TRAINER=${TRAINER:-mlx}
    GENERATOR=${GENERATOR:-mlx-batch}
    PREFIX_SHARE=${PREFIX_SHARE:-1}
    UPDATE_MICRO=${UPDATE_MICRO:-2}
    FORWARD_BATCH=${FORWARD_BATCH:-2}
    FAST_RULE=${FAST_RULE:-1}
    FAST_RULE_INV=${FAST_RULE_INV:-1}
    UPDATE_MICRO_TOKENS=${UPDATE_MICRO_TOKENS:-12000}
    TAG_STACK=ppo-mlx
else
    TRAINER=${TRAINER:-torch}
    GENERATOR=${GENERATOR:-torch}
    PREFIX_SHARE=${PREFIX_SHARE:-0}
    UPDATE_MICRO=${UPDATE_MICRO:-2}
    FORWARD_BATCH=${FORWARD_BATCH:-8}
    FAST_RULE=${FAST_RULE:-0}
    FAST_RULE_INV=${FAST_RULE_INV:-0}
    UPDATE_MICRO_TOKENS=${UPDATE_MICRO_TOKENS:-0}
    TAG_STACK=ppo
fi
RUN=${RUN:-${TAG_STACK}-qwen3.5-9b-l24-n32-lr1e6c-err0-delta-seed09k2-trainall-s$SEED}
RESUME=${RESUME:-0}

BIAS=${BIAS:-$(python3 -c "import json; print(round(json.load(open('$SEED_PATH/dev-T1-n120-stop.json'))['overall']['reward'], 4))")}
ARGS="--out $RUN --start-from $SEED_PATH --value-init-bias $BIAS --steps $STEPS --eval-every $EVAL_EVERY --eval-cases $EVAL_CASES --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin $REGRESS_MARGIN --reward-stop-drop $REWARD_STOP_DROP --stall-at $STALL_AT --lam 0.95 --adv-unit turn --sample-every 1 --lr $LR --critic $CRITIC --prover-k $PROVER_K --prover-alpha $ALPHA --step-watchdog $STEP_WATCHDOG --eval-watchdog $EVAL_WATCHDOG --adv-norm $ADV_NORM --minibatches $MINIBATCHES --target-kl $TARGET_KL --max-grad-norm $MAX_GRAD_NORM --adam-eps $ADAM_EPS --lr-anneal $LR_ANNEAL --lr-anneal-start $LR_ANNEAL_START --patience $PATIENCE --generator $GENERATOR --trainer $TRAINER --save-every $SAVE_EVERY --seed $SEED --prompts-per-step $PROMPTS --err-call-credit $ERR_CALL_CREDIT --call-credit $CALL_CREDIT --split $SPLIT"
[ "$STOP_AT" != 0 ] && ARGS="$ARGS --stop-at $STOP_AT"
[ "$CRITIC_LAYER" != 0 ] && ARGS="$ARGS --critic-layer $CRITIC_LAYER --critic-ridge $CRITIC_RIDGE --critic-l2 $CRITIC_L2"
[ "$PREFIX_SHARE" != 0 ] && ARGS="$ARGS --prefix-share $PREFIX_SHARE --update-micro $UPDATE_MICRO --forward-batch $FORWARD_BATCH"
[ "$FAST_RULE" != 0 ] && ARGS="$ARGS --fast-rule $FAST_RULE"
[ "$FAST_RULE_INV" != 0 ] && ARGS="$ARGS --fast-rule-inv $FAST_RULE_INV"
[ "$UPDATE_MICRO_TOKENS" != 0 ] && ARGS="$ARGS --update-micro-tokens $UPDATE_MICRO_TOKENS"
[ -n "$WARM_FROM" ] && ARGS="$ARGS --warm-from $WARM_FROM"
[ -n "$EVAL_GENERATOR" ] && ARGS="$ARGS --eval-generator $EVAL_GENERATOR"

mkdir -p "$R/$RUN"
if [ "$RESUME" = 1 ] && { [ -f "$R/$RUN/last/state.pt" ] || [ -f "$R/$RUN/last/state_mlx.json" ]; }; then
    rm -f "$R/$RUN/ending.json"
    $PY 09_fresh_questions.py ppo $ARGS --resume 1 2>&1 | tee -a "$R/$RUN/run.log"
else
    $PY 09_fresh_questions.py ppo $ARGS 2>&1 | tee -a "$R/$RUN/run.log"
fi
for attempt in 1 2 3 4 5 6 7 8; do
    [ -f "$R/$RUN/ending.json" ] && exit 0
    if { [ -f "$R/$RUN/last/state.pt" ] || [ -f "$R/$RUN/last/state_mlx.json" ]; }; then
        $PY 09_fresh_questions.py ppo $ARGS --resume 1 2>&1 | tee -a "$R/$RUN/run.log"
    else
        $PY 09_fresh_questions.py ppo $ARGS 2>&1 | tee -a "$R/$RUN/run.log"
    fi
done
