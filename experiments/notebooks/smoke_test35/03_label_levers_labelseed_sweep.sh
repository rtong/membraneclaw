#!/bin/sh
# 03_label_levers, the label-seed dose sweep: the stage seed (calls, the answer's `{` and the stage
# value on the 400 train cases) plus k records of each of the seven causes from
# smoke_test/data/seed_cases.jsonl, labelled on the stage, root_cause and action values. Each seed is
# probed at temperature 1 on all of dev (2 samples a case) and greedy on dev; dose 0 is the stage
# seed itself under the same probe.
# Skips what already exists. Every job capped at 3000 MiB.
cd "$(dirname "$0")"
PY=../../membrane_grpo/.venv/bin/python
R=03_label_levers_runs
ALL=scaling,oxidation_damage,organic_fouling,colloidal_fouling,biofouling,compaction,mechanical_leak
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
probe() {  # adapter, out prefix, log
  [ -e "$2-probe-T1.json" ] || $PY 03_label_levers.py eval --adapter $1 --split dev --temperature 1 --samples 2 \
    --stop-at-json 1 --batch 8 --out $2-probe-T1.json --max-vram-mib 3000 >> $3 2>&1
  [ -e "$2-dev.json" ] || $PY 03_label_levers.py eval --adapter $1 --split dev --stop-at-json 1 --batch 8 \
    --out $2-dev.json --max-vram-mib 3000 >> $3 2>&1
}
[ -e $R/tool-seed-stage/epoch3-probe-T1.json ] || $PY 03_label_levers.py eval \
  --adapter $R/tool-seed-stage/epoch3 --split dev --temperature 1 --samples 2 --stop-at-json 1 \
  --batch 8 --out $R/tool-seed-stage/epoch3-probe-T1.json --max-vram-mib 3000 >> $R/tool-seed-stage.log 2>&1
for K in ${DOSES:-4 2 8}; do
  S=$R/label-seed-pc$K
  [ -e "$S/epoch3/adapter_model.safetensors" ] || $PY 03_label_levers.py labelseed --out label-seed-pc$K \
    --causes $ALL --per-cause $K --replay-slots stage --grad-checkpoint 1 --max-vram-mib 3000 >> $S.log 2>&1
  probe $S/epoch3 $S/epoch3 $S.log
done
