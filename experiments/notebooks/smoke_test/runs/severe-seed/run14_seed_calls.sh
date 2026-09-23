#!/bin/bash
# 14, step 1: a severe-action seed on top of the tool seed, and the dose that PPO starts from.
#
# Seed: runs/tool-sft-raw (13b's starting point) + 150 of 11's off-distribution records (seeded shuffle),
# on the gold tool episode. Supervised: the three calls (tool_call_target, the tool seed's own target), the
# answer turn's first token, and the value of `action`. No other answer field: root_cause, flags, stage and
# the numbers are context. lr 5e-5, a checkpoint per epoch; dose 0 is runs/tool-sft-raw (probe_epoch0.json).
#
# Per dose, in order, stopping at the first that is chosen or fails:
#   (c) greedy dev numeric_acc >= 0.99                                  -- else stop
#   probe: 150 train cases x 4 samples at temperature 1.0 (10's probe)
#   (d) sampled numeric_acc >= 0.90, 13b's own --numeric-stop            -- else stop
#   (b) severe said on <= 10% of the samples whose answer is not severe  -- else stop
#   (a) severe said at least once on a sample whose answer is severe     -- then this dose is chosen
# A larger dose only moves further, so a dose that fails (b)-(d) ends the search.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
OUT=runs/severe-seed
SEED=runs/tool-seed-severe-150-calls

$PY ppo_ac.py --prompt-tool 1 --sft-only 1 --sft-tool 1 --sft-seed-from runs/tool-sft-raw/adapter \
    --sft-cases data/seed_cases.jsonl --sft-max-cases 150 --sft-labels-only 2 --sft-slots action \
    --sft-anchor-open 1 --sft-anchor-calls 1 --sft-epochs 4 --sft-lr 5e-5 --max-vram-mib 10500 \
    --out $SEED > $OUT/calls_sft.log 2>&1
echo "calls sft exit $?" >> $OUT/status

for k in 1 2 3 4; do
    $PY ppo_ac.py --prompt-tool 1 --eval-only 1 --resume-from $SEED/epoch$k --eval-batch 16 --max-vram-mib 10500 \
        --out $OUT/dev_calls_epoch$k.json > $OUT/dev_calls_epoch$k.log 2>&1
    echo "dev calls_epoch$k exit $?" >> $OUT/status
    $PY -c "import json,sys; o=json.load(open('$OUT/dev_calls_epoch$k.json'))['overall']; sys.exit(o['numeric_acc'] < 0.99)" \
        || { echo "calls_epoch$k fails (c); stop" >> $OUT/status; break; }

    $PY ppo_ac.py --prompt-tool 1 --eval-only 1 --resume-from $SEED/epoch$k --eval-split train --eval-cases 150 \
        --eval-samples 4 --eval-temperature 1.0 --eval-batch 16 --max-vram-mib 10500 \
        --out $OUT/probe_calls_epoch$k.json > $OUT/probe_calls_epoch$k.log 2>&1
    echo "probe calls_epoch$k exit $?" >> $OUT/status
    verdict=$($PY -c "
import json
d = json.load(open('$OUT/probe_calls_epoch$k.json')); o, p = d['overall'], d['probe']
fp = (p['severe_emitted'] - p['severe_correct']) / (p['samples'] - p['severe_slots'])
print('fail-d' if o['numeric_acc'] < 0.90 else 'fail-b' if fp > 0.10 else 'chosen' if p['severe_correct'] >= 1 else 'next')")
    echo "calls_epoch$k: $verdict" >> $OUT/status
    [ "$verdict" = next ] || break
done
echo "calls all done" >> $OUT/status
