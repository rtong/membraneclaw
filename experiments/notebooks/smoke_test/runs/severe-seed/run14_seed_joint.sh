#!/bin/bash
# 14, step 1: the tool seed and a severe-action seed as one SFT from the raw model.
#
# One shuffled pass over (i) every train case exactly as runs/tool-sft-raw was trained -- the three gold
# calls, nothing else -- and (ii) 150 of 11's off-distribution records (seeded shuffle) as gold tool
# episodes, supervised on the calls, the answer turn's first token and the value of `action`. No other
# answer field is supervised anywhere. lr 1e-4 and 3 epochs as the tool seed; a checkpoint per epoch.
#
# Per epoch: greedy dev (with the calls teacher-forced), then 150 train cases x 4 samples at temperature 1.0.
# Chosen: the smallest epoch that passes all of
#   (a) severe said at least once on a sample whose answer is severe
#   (b) severe said on <= 10% of the samples whose answer is not severe
#   (c) greedy dev numeric_acc >= 0.99
#   (d) sampled numeric_acc >= 0.90
#   (e) teacher-forced call entropy on dev <= 2x runs/tool-sft-raw's (dev_toolsftraw.json)
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
OUT=runs/severe-seed
SEED=runs/tool-seed-joint-150

dev() {  # name, run directory
    $PY ppo_ac.py --prompt-tool 1 --eval-only 1 --resume-from "$2" --eval-batch 16 --max-vram-mib 10500 \
        --out $OUT/dev_$1.json > $OUT/dev_$1.log 2>&1
    echo "dev $1 exit $?" >> $OUT/status
}
probe() {  # name, run directory
    $PY ppo_ac.py --prompt-tool 1 --eval-only 1 --resume-from "$2" --eval-split train --eval-cases 150 \
        --eval-samples 4 --eval-temperature 1.0 --eval-batch 16 --max-vram-mib 10500 \
        --out $OUT/probe_$1.json > $OUT/probe_$1.log 2>&1
    echo "probe $1 exit $?" >> $OUT/status
}

dev toolsftraw runs/tool-sft-raw

$PY ppo_ac.py --prompt-tool 1 --sft-only 1 --sft-tool 1 --sft-tool-replay 1 \
    --sft-cases data/seed_cases.jsonl --sft-max-cases 150 --sft-labels-only 2 --sft-slots action \
    --sft-anchor-open 1 --sft-anchor-calls 1 --sft-epochs 3 --sft-lr 1e-4 --max-vram-mib 10500 \
    --out $SEED > $OUT/joint_sft.log 2>&1
echo "joint sft exit $?" >> $OUT/status

for k in 1 2 3; do
    dev joint_epoch$k $SEED/epoch$k
    pre=$($PY -c "
import json
base = json.load(open('$OUT/dev_toolsftraw.json'))['overall']['call_entropy']
o = json.load(open('$OUT/dev_joint_epoch$k.json'))['overall']
print('fail-c' if o['numeric_acc'] < 0.99 else 'fail-e' if o['call_entropy'] > 2 * base else 'ok')")
    echo "joint_epoch$k dev: $pre" >> $OUT/status
    [ "$pre" = ok ] || continue
    probe joint_epoch$k $SEED/epoch$k
    verdict=$($PY -c "
import json
d = json.load(open('$OUT/probe_joint_epoch$k.json')); o, p = d['overall'], d['probe']
fp = (p['severe_emitted'] - p['severe_correct']) / (p['samples'] - p['severe_slots'])
print('fail-d' if o['numeric_acc'] < 0.90 else 'fail-b' if fp > 0.10 else 'fail-a' if p['severe_correct'] < 1 else 'pass')")
    echo "joint_epoch$k probe: $verdict" >> $OUT/status
    if [ "$verdict" = pass ]; then echo "joint chosen: epoch$k" >> $OUT/status; break; fi
done
echo "joint all done" >> $OUT/status
