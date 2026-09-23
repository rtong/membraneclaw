#!/bin/bash
# 14, step 1: a severe-action seed on top of the tool seed, at three doses, and the probes that pick one.
#
# Seed: runs/tool-sft-raw (13b's starting point) + 150 of 11's off-distribution records (seeded shuffle),
# loss on the value of `action` in the answer turn of the gold tool episode, plus the answer turn's first
# token (`--sft-anchor-open 1`: "answer now" rather than "call again"; no task content). root_cause is not
# supervised: on the tool line organic_fouling is already emitted. lr 1e-4 as in 10/11, no dead weight.
# Doses: epoch 1, 2, 3 of one pass; dose 0 is runs/tool-sft-raw itself (probe_epoch0.json).
#
# Probe per dose: 150 train cases x 4 samples at temperature 1.0 (10's probe), plus greedy dev.
# Choice: the smallest dose that
#   (a) says the severe action at least once on a sample whose answer is severe,
#   (b) has not collapsed the slot: severe said on at most 10% of the samples whose answer is not severe,
#   (c) keeps the calls and the answer: greedy dev numeric_acc >= 0.99.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
OUT=runs/severe-seed
SEED=runs/tool-seed-severe-150-anchor

$PY ppo_ac.py --prompt-tool 1 --sft-only 1 --sft-tool 1 --sft-seed-from runs/tool-sft-raw/adapter \
    --sft-cases data/seed_cases.jsonl --sft-max-cases 150 --sft-labels-only 2 --sft-slots action \
    --sft-anchor-open 1 --sft-epochs 3 --sft-lr 1e-4 --max-vram-mib 13000 --out $SEED > $OUT/anchor_sft.log 2>&1
echo "anchor sft exit $?" >> $OUT/status

probe() {  # name, run directory
    $PY ppo_ac.py --prompt-tool 1 --eval-only 1 --resume-from "$2" --eval-split train --eval-cases 150 \
        --eval-samples 4 --eval-temperature 1.0 --eval-batch 16 --max-vram-mib 13000 \
        --out $OUT/probe_$1.json > $OUT/probe_$1.log 2>&1
    echo "probe $1 exit $?" >> $OUT/status
}
dev() {  # name, run directory
    $PY ppo_ac.py --prompt-tool 1 --eval-only 1 --resume-from "$2" --max-vram-mib 13000 \
        --out $OUT/dev_$1.json > $OUT/dev_$1.log 2>&1
    echo "dev $1 exit $?" >> $OUT/status
}

for k in 1 2 3; do dev anchor_epoch$k $SEED/epoch$k; done
for k in 1 2 3; do probe anchor_epoch$k $SEED/epoch$k; done
echo "anchor all done" >> $OUT/status
