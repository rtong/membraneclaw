#!/usr/bin/env bash
# PPO from this notebook's seed at 06's stable configuration: the turn as the unit of credit (one assistant
# message is one action, lambda 0.95 per turn, every token of a turn carrying its turn's advantage) and the
# actor step a fifth of 05's (lr 2e-6; a turn's advantage is about five times a token's). Otherwise 06's run:
# critic warm-up 25, turn ends and answer scaffolding out of the actor's loss, Adam eps 1e-6, answers stopped at
# their closing brace, KL 0.01 to the seed, value head on layer 23 warm-started at the seed's own twice-sampled
# dev reward, a 3,072-token policy budget, the tool gate off, the answer-collapse rule on, the no-regression
# gate off, every step's 8 episodes recorded. Runs until a rule, convergence (8 evaluations without a new best)
# or STEPS (200). Greedy evals on 120 dev questions every 25 steps; every evaluation checkpoints `last/`, and a
# run the watchdog ends is resumed from it (up to 8 times). Each (re)start waits until the card has room.
#   SEED_RUN=recovery-seed-n<dose> ./07_error_recovery_ppo_run.sh     the run: ppo-n<dose>-turn-lr2e6-s0
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
R=07_error_recovery_runs
SEED_RUN=${SEED_RUN:?set SEED_RUN=recovery-seed-n<the alive dose>}
SEED_CKPT=${SEED_CKPT:-epoch1}
RUN=${RUN:-ppo-${SEED_RUN#recovery-seed-}-turn-lr2e6-s0}
ADV_UNIT=${ADV_UNIT:-turn}
LAM=${LAM:-0.95}
LR=${LR:-2e-6}
STEPS=${STEPS:-200}
BIAS=$(python3 -c "import json; print(round(json.load(open('$R/$SEED_RUN/$SEED_CKPT/dev-T1-x2-stop.json'))['overall']['reward'], 4))")
ARGS="--out $RUN --start-from $R/$SEED_RUN/$SEED_CKPT --value-init-bias $BIAS --steps $STEPS --eval-every 25 --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin 1 --lam $LAM --adv-unit $ADV_UNIT --sample-every 1 --lr $LR"
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
wait_card
echo "$(date +%H:%M) ppo $RUN from $SEED_RUN (value bias $BIAS)"
$PY 07_error_recovery.py ppo $ARGS
for attempt in 1 2 3 4 5 6 7 8; do
    [ -f $R/$RUN/ending.json ] && break
    wait_card
    if [ -f $R/$RUN/last/state.pt ]; then
        $PY 07_error_recovery.py ppo $ARGS --resume 1
    else
        $PY 07_error_recovery.py ppo $ARGS
    fi
done
echo "$(date +%H:%M) ppo done: $(cat $R/$RUN/ending.json 2>/dev/null)"
