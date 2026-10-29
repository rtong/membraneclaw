#!/usr/bin/env bash
# 08.5: 08's PPO on the MLX stack -- generation, the update, the reference's log-probs and the separate critic's reads
# all on mlx-lm, and no PyTorch in the process (TRAINER=mlx: train_ppo_mlx under block_torch; 08 generated on MLX and
# trained in PyTorch on MPS). The run: D37 -- 08's configuration (07's diagnosis-line seed, Qwen3-4B critic L27, prover
# Bo4, alpha 0.8, turn credit, KL 0.01) with three of the 37 PPO details (advantages normalised per minibatch, 4
# minibatches x 4 epochs, grad clip 0.5, Adam eps 1e-5, lr 2e-6 annealed linearly to 0), 200 steps, evaluations of 60
# dev questions every 40; seed 0 and 07's first 25 steps (WARM_FROM), so it pairs step by step with 07 and 08's v1.
#   ./08.5_mlx_stack_ppo_run.sh
#   RESUME=1 ./...                                continue from last/
#   RESUME=1 STEPS=400 LR_ANNEAL_START=200 EVAL_EVERY=20 PATIENCE=6 ./...
#                                                 D37 continued from step 200 to convergence: the lr 2e-6 annealed
#                                                 again, to 0 at 400; stop after 6 evaluations without a new best
#   CRITIC=policy ./...                           07's critic instead (the policy's own value head, train_ppo_mlx's
#                                                 critic "policy": 07's hyper-parameters, PPO_DEFAULTS' value_*, critic_*),
#                                                 no prover (ALPHA 0); its 25 warm-up steps replayed from WARM_FROM
#   CRITIC=policy WARM_FROM= PROMPTS=32 LR=1e-6 ERR_CALL_CREDIT=0 CALL_CREDIT=delta STEPS=400 STOP_AT=200 ./...
#                                                 the other 07's experiment 2 (07_error_recovery_diag/EXP2_PROMPT.md)
#                                                 on 07c+D37: 4x the questions per step at half the lr (annealed to 0
#                                                 at 400 as 07c+D37's, stopped at 200), no call credit for a rejected
#                                                 call, delta call credit; its own 25 warm-up steps (32 questions each)
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
R07=07_recover_from_errors_runs
choice() { python3 -c "import json; print(json.load(open('08_prover_critic_critic_fit.json'))['choice']['$1'])"; }  # 08's critic-fit
CRITIC=${CRITIC:-$(choice critic)}
if [ "$CRITIC" = policy ]; then  # 07's critic: nothing of critic-fit's (its layer, ridge and l2 are a separate critic's)
    PROVER_K=${PROVER_K:-4}
    ALPHA=${ALPHA:-0}
else
    PROVER_K=${PROVER_K:-$(choice prover_k)}
    ALPHA=${ALPHA:-$(choice alpha)}
fi
SEED=${SEED:-0}
if [ "$CRITIC" = policy ]; then
    RUN=${RUN:-ppo-mlx-policy-d37-s$SEED}
else
    RUN=${RUN:-ppo-mlx-$CRITIC-bo$PROVER_K-a$ALPHA-d37-s$SEED}  # the run directory under 08.5_mlx_stack_runs/
fi
SEED_RUN=${SEED_RUN:-seed-n200-r100}            # 07's seed
SEED_PATH=${SEED_PATH:-$R07/$SEED_RUN/epoch1}   # or 08's own seed, e.g. 08_prover_critic_runs/seed-rt-n300-e4/epoch1
WARM_FROM=${WARM_FROM-$R07/ppo-n200-r100-turn-s0}  # empty: its own warm-up (needed with another seed)
LR=${LR:-2e-6}
REWARD_STOP_DROP=${REWARD_STOP_DROP:-0.08}
STALL_AT=${STALL_AT:-40,80}
REGRESS_MARGIN=${REGRESS_MARGIN:-0.05}
STEPS=${STEPS:-200}
EVAL_EVERY=${EVAL_EVERY:-40}
EVAL_CASES=${EVAL_CASES:-60}
PATIENCE=${PATIENCE:-8}                 # evaluations without a new best before "converged"
PLATEAU_WINDOW=${PLATEAU_WINDOW:-0}     # >0: stop on a plateau of the last N evaluations' mean (plateau_update)
PLATEAU_PATIENCE=${PLATEAU_PATIENCE:-0}
PLATEAU_DELTA=${PLATEAU_DELTA:-0}
PLATEAU_MIN_STEP=${PLATEAU_MIN_STEP:-0}
LR_ANNEAL_START=${LR_ANNEAL_START:-0}   # 0: anneal from the first actor step; N: from step N to 0 at STEPS
STEP_WATCHDOG=${STEP_WATCHDOG:-3600}    # 07: 1800 and 7200; doubled, for steps and evaluations slowed by another GPU job
EVAL_WATCHDOG=${EVAL_WATCHDOG:-14400}
D37=${D37:-1}
if [ "$D37" = 1 ]; then D_ADV=minibatch; D_MB=4; D_CLIP=0.5; D_EPS=1e-5; D_ANNEAL=1; else D_ADV=none; D_MB=1; D_CLIP=1.0; D_EPS=1e-6; D_ANNEAL=0; fi
ADV_NORM=${ADV_NORM:-$D_ADV}             # none: 03-07's update; batch / minibatch
MINIBATCHES=${MINIBATCHES:-$D_MB}
TARGET_KL=${TARGET_KL:-0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-$D_CLIP}
ADAM_EPS=${ADAM_EPS:-$D_EPS}
LR_ANNEAL=${LR_ANNEAL:-$D_ANNEAL}
GENERATOR=${GENERATOR:-mlx}             # mlx-batch: each round's episodes decoded together (rollout_mlx_batched)
EVAL_GENERATOR=${EVAL_GENERATOR:-}      # empty: GENERATOR's; mlx keeps evaluations comparable after a switch
TRAINER=${TRAINER:-mlx}                 # 08.5: mlx -- the update, the reference and the critic on MLX
SAVE_EVERY=${SAVE_EVERY:-5}
PROMPTS=${PROMPTS:-8}                   # questions (episodes) per step
ERR_CALL_CREDIT=${ERR_CALL_CREDIT:-1}   # 0: a simulate call the tool rejected earns no call credit (the other 07's)
CALL_CREDIT=${CALL_CREDIT:-call}        # delta: turn j earns C[j] - C[j-1] of the calls component (the other 07's)
SPLIT=${SPLIT:-train}                   # training questions: 06_direction_and_compare_data/$SPLIT.jsonl (train_units: questions with hard-to-convert units repeated x3 / x2)
STOP_AT=${STOP_AT:-0}                   # >0: end after this step, the lr annealed on the schedule to STEPS
CRITIC_LAYER=${CRITIC_LAYER:-0}         # a separate critic's layer, ridge and l2 (0: critic-fit's choice for the model;
CRITIC_RIDGE=${CRITIC_RIDGE:-0}         # a model critic-fit never read, e.g. qwen3.5-9b, needs all three)
CRITIC_L2=${CRITIC_L2:-0}
PREFIX_SHARE=${PREFIX_SHARE:-0}         # 1: the shared system prefix once per step / minibatch (mlx_actor_grads_p)
UPDATE_MICRO=${UPDATE_MICRO:-2}         # with PREFIX_SHARE: episodes per suffix micro-batch in the update
FORWARD_BATCH=${FORWARD_BATCH:-8}       # with PREFIX_SHARE: suffixes per forward pass (old / reference log-probs)
FAST_RULE=${FAST_RULE:-0}               # 1: the delta rule's custom-VJP scan in training mode (~1.4x faster update)
FAST_RULE_INV=${FAST_RULE_INV:-0}       # with FAST_RULE: the unit-lower inverse's closed-form VJP too (the rule's backward 10-29% faster)
UPDATE_MICRO_TOKENS=${UPDATE_MICRO_TOKENS:-0}  # with PREFIX_SHARE, >0: a suffix micro-batch holds at most this many padded tokens
                                        # (rows x longest suffix): bounds the update's peak (~12.5 GB + 2.55 GB per 1k)
RESUME=${RESUME:-0}
R=08.5_mlx_stack_runs
BIAS=$(python3 -c "import json; print(round(json.load(open('$SEED_PATH/dev-T1-n120-stop.json'))['overall']['reward'], 4))")
ARGS="--out $RUN --start-from $SEED_PATH --value-init-bias $BIAS --steps $STEPS --eval-every $EVAL_EVERY --eval-cases $EVAL_CASES --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin $REGRESS_MARGIN --reward-stop-drop $REWARD_STOP_DROP --stall-at $STALL_AT --lam 0.95 --adv-unit turn --sample-every 1 --lr $LR --critic $CRITIC --prover-k $PROVER_K --prover-alpha $ALPHA --step-watchdog $STEP_WATCHDOG --eval-watchdog $EVAL_WATCHDOG --adv-norm $ADV_NORM --minibatches $MINIBATCHES --target-kl $TARGET_KL --max-grad-norm $MAX_GRAD_NORM --adam-eps $ADAM_EPS --lr-anneal $LR_ANNEAL --lr-anneal-start $LR_ANNEAL_START --patience $PATIENCE --generator $GENERATOR --trainer $TRAINER --save-every $SAVE_EVERY --seed $SEED --prompts-per-step $PROMPTS --err-call-credit $ERR_CALL_CREDIT --call-credit $CALL_CREDIT --split $SPLIT"
[ "$STOP_AT" != 0 ] && ARGS="$ARGS --stop-at $STOP_AT"
[ "$CRITIC_LAYER" != 0 ] && ARGS="$ARGS --critic-layer $CRITIC_LAYER --critic-ridge $CRITIC_RIDGE --critic-l2 $CRITIC_L2"
[ "$PREFIX_SHARE" != 0 ] && ARGS="$ARGS --prefix-share $PREFIX_SHARE --update-micro $UPDATE_MICRO --forward-batch $FORWARD_BATCH"
[ "$FAST_RULE" != 0 ] && ARGS="$ARGS --fast-rule $FAST_RULE"
[ "$FAST_RULE_INV" != 0 ] && ARGS="$ARGS --fast-rule-inv $FAST_RULE_INV"
[ "$UPDATE_MICRO_TOKENS" != 0 ] && ARGS="$ARGS --update-micro-tokens $UPDATE_MICRO_TOKENS"
[ -n "$WARM_FROM" ] && ARGS="$ARGS --warm-from $WARM_FROM"
[ -n "$EVAL_GENERATOR" ] && ARGS="$ARGS --eval-generator $EVAL_GENERATOR"
[ "$PLATEAU_WINDOW" != 0 ] && ARGS="$ARGS --plateau-window $PLATEAU_WINDOW --plateau-patience $PLATEAU_PATIENCE --plateau-delta $PLATEAU_DELTA --plateau-min-step $PLATEAU_MIN_STEP"
mkdir -p $R/$RUN
if [ "$RESUME" = 1 ] && { [ -f $R/$RUN/last/state.pt ] || [ -f $R/$RUN/last/state_mlx.json ]; }; then
    rm -f $R/$RUN/ending.json
    $PY 08.5_mlx_stack.py ppo $ARGS --resume 1 2>&1 | tee -a $R/$RUN/run.log
else
    $PY 08.5_mlx_stack.py ppo $ARGS 2>&1 | tee -a $R/$RUN/run.log
fi
for attempt in 1 2 3 4 5 6 7 8; do  # a crash resumes from the last evaluation's state (or starts again before one)
    [ -f $R/$RUN/ending.json ] && exit 0
    if { [ -f $R/$RUN/last/state.pt ] || [ -f $R/$RUN/last/state_mlx.json ]; }; then
        $PY 08.5_mlx_stack.py ppo $ARGS --resume 1 2>&1 | tee -a $R/$RUN/run.log
    else
        $PY 08.5_mlx_stack.py ppo $ARGS 2>&1 | tee -a $R/$RUN/run.log
    fi
done
