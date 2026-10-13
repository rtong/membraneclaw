#!/usr/bin/env bash
# The full run: 04's configuration (critic warm-up 25; turn ends and answer scaffolding out of the
# actor's loss; Adam eps 1e-6; answers stopped at their closing brace; KL 0.01 to the seed; value head
# on layer 23 warm-started at the seed's own sampled dev reward), from this notebook's seed -- the smallest
# SFT dose that is alive, not 05's label seed -- with the policy's token budget at 3,072 (the .py's default
# for 06; 05's was 2,048) and 06's stopping rules: the tool gate off (a seed that barely works starts below
# any line, and the gate made relative to the run's own start stopped two runs within 40 actor steps), the
# answer-collapse rule on (stop when the last 10 steps' answer rate falls below half the warm-up's), the
# no-regression gate off. The unit of credit is what this notebook's third run from the seed changes. With the
# token as the unit (05's GAE), the run at lambda 0.95 learned to skip unit conversion and collapsed -- a skipped
# conversion's bad news arrives with the simulate error hundreds of policy tokens after the convert-or-simulate
# decision, and 0.95 per token discounts it to nothing by the time it reaches that decision -- and the run at
# lambda 1 credited every token of a turn with the whole return and learned to write convert calls, the needless
# ones alike. ADV_UNIT=turn makes the turn the unit: one assistant message is one action, its reward the credits
# that land in it, lambda 0.95 per turn, every token of a turn carrying its turn's advantage (the .py's
# PPO_DEFAULTS has the definition). Every step's 8 episodes are recorded (SAMPLE_EVERY=1, recording only; 05
# recorded every 5th step), so the per-unit-level diagnostics have enough episodes. Runs until a rule, convergence
# (8 evaluations without a new best) or STEPS (default 200: a first look, then RESUME=1 STEPS=500 continues the
# same run from its last checkpoint) stops it. Greedy evals on 120 dev questions every 25 steps; every
# evaluation checkpoints `last/`, and a run the watchdog ends is resumed from it (up to 8 times). Each
# (re)start waits until the card has room (display and games keep 2 GB).
#   ./06_direction_and_compare_ppo_run.sh                                the turn as the unit, lambda 0.95, from the 600 seed: ppo-n600-open-turn-s0
#   LR=2e-6 RUN=ppo-n600-open-turn-lr2e6-s0 ./...                        the same at a fifth of the actor step: the fourth run from the seed
#   ADV_UNIT=token LAM=1 RUN=ppo-n600-open-lam1-s0 ./...                 the token as the unit at lambda 1: the run that over-converted
#   ADV_UNIT=token LAM=0.95 SAMPLE_EVERY=5 RUN=ppo-n600-open-s0 ./...    the token as the unit at 05's lambda: the run that collapsed
#   ADV_UNIT=token LAM=0.95 SAMPLE_EVERY=5 RUN=ppo-n1200-open-s0 SEED_RUN=label-seed-n1200 ./...     the next dose, 05's lambda and unit
#   ADV_UNIT=token LAM=0.95 SAMPLE_EVERY=5 TOOL_STOP_REL=0.9 REGRESS_MARGIN=0.05 RUN=ppo-n<dose>-s0 SEED_RUN=label-seed-n<dose> ./...
#                                                                        the tool gate on: the two runs it stopped
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
RUN=${RUN:-ppo-n600-open-turn-s0}        # the run directory under 06_direction_and_compare_runs/
SEED_RUN=${SEED_RUN:-label-seed-n600}    # the label seed the run starts from: the dose
SEED_CKPT=${SEED_CKPT:-epoch1}           # its checkpoint
ADV_UNIT=${ADV_UNIT:-turn}               # the unit of credit: turn (one assistant message is one action) or token (05's GAE)
LAM=${LAM:-0.95}                         # GAE lambda, per ADV_UNIT (the lambda-1 run: 1 per token)
SAMPLE_EVERY=${SAMPLE_EVERY:-1}          # record every Nth step's 8 episodes in samples.jsonl (recording only; 05: 5)
LR=${LR:-1e-5}                           # the actor's learning rate (the .py's default); the fourth run from the seed: 2e-6, a fifth -- a turn's advantage is about five times a token's
TOOL_STOP=${TOOL_STOP:-0}                # 05's absolute tool gate (call fidelity < 0.9 over 10 steps); off: a seed that barely works starts below it
TOOL_STOP_REL=${TOOL_STOP_REL:-0}        # the tool gate relative to this run's own start: 0.9 stops when the last 10 steps' call fidelity < the warm-up mean - 0.1; 0: off
ANSWER_STOP_REL=${ANSWER_STOP_REL:-0.5}  # the collapse rule: stop when the last 10 steps' answer rate < 0.5 x the warm-up mean
STEPS=${STEPS:-200}                      # the step cap: 200 for a first look (05's 2000 was never reached: convergence stopped it at 244)
RESUME=${RESUME:-0}                      # 1: continue the finished run from last/ up to a higher STEPS (its ending.json is replaced)
REGRESS_MARGIN=${REGRESS_MARGIN:-1}      # 05's no-regression gate: two evaluations below the start by more than this stop the run (1: off)
BIAS=$(python3 -c "import json; print(round(json.load(open('06_direction_and_compare_runs/$SEED_RUN/$SEED_CKPT/dev-T1-x2-stop.json'))['overall']['reward'], 4))")
ARGS="--out $RUN --start-from 06_direction_and_compare_runs/$SEED_RUN/$SEED_CKPT --value-init-bias $BIAS --steps $STEPS --eval-every 25 --tool-stop $TOOL_STOP --tool-stop-rel $TOOL_STOP_REL --answer-stop-rel $ANSWER_STOP_REL --regress-margin $REGRESS_MARGIN --lam $LAM --adv-unit $ADV_UNIT --sample-every $SAMPLE_EVERY --lr $LR"
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2500 ]; do sleep 60; done; }
wait_card
if [ "$RESUME" = 1 ] && [ -f 06_direction_and_compare_runs/$RUN/last/state.pt ]; then
    rm -f 06_direction_and_compare_runs/$RUN/ending.json
    $PY 06_direction_and_compare.py ppo $ARGS --resume 1
else
    $PY 06_direction_and_compare.py ppo $ARGS
fi
for attempt in 1 2 3 4 5 6 7 8; do
    [ -f 06_direction_and_compare_runs/$RUN/ending.json ] && exit 0
    wait_card
    if [ -f 06_direction_and_compare_runs/$RUN/last/state.pt ]; then
        $PY 06_direction_and_compare.py ppo $ARGS --resume 1
    else
        $PY 06_direction_and_compare.py ppo $ARGS
    fi
done
