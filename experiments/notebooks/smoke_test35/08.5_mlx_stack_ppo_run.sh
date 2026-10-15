#!/usr/bin/env bash
# 08.5: 08's PPO on the MLX stack -- generation, the update, the reference's log-probs and the separate critic's reads
# all on mlx-lm (TRAINER=mlx; 08 generated on MLX and trained in PyTorch on MPS). The configuration is 08's D37 run's
# (ppo-qwen3-4b-bo4-a0.8-d37-s0: three of the 37 PPO details, 200 steps, Qwen3-4B critic L27, prover Bo4, alpha 0.8,
# turn credit, 07's diagnosis-line seed, evaluations of 60 dev questions every 40 steps), with another seed (SEED=1):
# 08's runs warm-start from 07's first 25 steps, which needs 07's seed, so this one takes its own 25-step warm-up and
# step-0 evaluation (WARM_FROM empty). Everything else unchanged.
#   ./08.5_mlx_stack_ppo_run.sh
#   RESUME=1 ./...                                continue from last/
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
R07=07_recover_from_errors_runs
choice() { python3 -c "import json; print(json.load(open('08_prover_critic_critic_fit.json'))['choice']['$1'])"; }  # 08's critic-fit
CRITIC=${CRITIC:-$(choice critic)}
PROVER_K=${PROVER_K:-$(choice prover_k)}
ALPHA=${ALPHA:-$(choice alpha)}
SEED=${SEED:-1}
RUN=${RUN:-ppo-mlx-$CRITIC-bo$PROVER_K-a$ALPHA-d37-s$SEED}  # the run directory under 08.5_mlx_stack_runs/
SEED_RUN=${SEED_RUN:-seed-n200-r100}            # 07's seed
SEED_PATH=${SEED_PATH:-$R07/$SEED_RUN/epoch1}   # or 08's own seed, e.g. 08_prover_critic_runs/seed-rt-n300-e4/epoch1
WARM_FROM=${WARM_FROM-}                         # empty: its own warm-up (07's run is seed 0)
LR=${LR:-2e-6}
REWARD_STOP_DROP=${REWARD_STOP_DROP:-0.08}
STALL_AT=${STALL_AT:-40,80}
REGRESS_MARGIN=${REGRESS_MARGIN:-0.05}
STEPS=${STEPS:-200}
EVAL_EVERY=${EVAL_EVERY:-40}
EVAL_CASES=${EVAL_CASES:-60}
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
GENERATOR=${GENERATOR:-mlx}
TRAINER=${TRAINER:-mlx}                 # 08.5: mlx -- the update, the reference and the critic on MLX
SAVE_EVERY=${SAVE_EVERY:-5}
RESUME=${RESUME:-0}
R=08.5_mlx_stack_runs
BIAS=$(python3 -c "import json; print(round(json.load(open('$SEED_PATH/dev-T1-n120-stop.json'))['overall']['reward'], 4))")
ARGS="--out $RUN --start-from $SEED_PATH --value-init-bias $BIAS --steps $STEPS --eval-every $EVAL_EVERY --eval-cases $EVAL_CASES --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin $REGRESS_MARGIN --reward-stop-drop $REWARD_STOP_DROP --stall-at $STALL_AT --lam 0.95 --adv-unit turn --sample-every 1 --lr $LR --critic $CRITIC --prover-k $PROVER_K --prover-alpha $ALPHA --step-watchdog $STEP_WATCHDOG --eval-watchdog $EVAL_WATCHDOG --adv-norm $ADV_NORM --minibatches $MINIBATCHES --target-kl $TARGET_KL --max-grad-norm $MAX_GRAD_NORM --adam-eps $ADAM_EPS --lr-anneal $LR_ANNEAL --generator $GENERATOR --trainer $TRAINER --save-every $SAVE_EVERY --seed $SEED"
[ -n "$WARM_FROM" ] && ARGS="$ARGS --warm-from $WARM_FROM"
mkdir -p $R/$RUN
if [ "$RESUME" = 1 ] && [ -f $R/$RUN/last/state.pt ]; then
    rm -f $R/$RUN/ending.json
    $PY 08.5_mlx_stack.py ppo $ARGS --resume 1 2>&1 | tee -a $R/$RUN/run.log
else
    $PY 08.5_mlx_stack.py ppo $ARGS 2>&1 | tee -a $R/$RUN/run.log
fi
for attempt in 1 2 3 4 5 6 7 8; do  # a crash resumes from the last evaluation's state (or starts again before one)
    [ -f $R/$RUN/ending.json ] && exit 0
    if [ -f $R/$RUN/last/state.pt ]; then
        $PY 08.5_mlx_stack.py ppo $ARGS --resume 1 2>&1 | tee -a $R/$RUN/run.log
    else
        $PY 08.5_mlx_stack.py ppo $ARGS 2>&1 | tee -a $R/$RUN/run.log
    fi
done
