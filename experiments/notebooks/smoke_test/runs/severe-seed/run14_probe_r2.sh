#!/bin/bash
# 14, step 1 (continued): the probes and dev passes run14_seed.sh had left, same commands, probes at
# --eval-batch 16. Waits for the dev pass already on the card (epoch1) before starting.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../membrane_grpo/.venv/bin/python
OUT=runs/severe-seed
SEED=runs/tool-seed-severe-150

while kill -0 3170197 2>/dev/null; do sleep 5; done
[ -f $OUT/dev_epoch1.json ] && echo "dev epoch1 wrote" >> $OUT/status || echo "dev epoch1 missing" >> $OUT/status

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

for k in 1 2 3; do probe epoch$k $SEED/epoch$k; done
for k in 2 3; do dev epoch$k $SEED/epoch$k; done
echo "all done" >> $OUT/status
