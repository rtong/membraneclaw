#!/usr/bin/env bash
# 09: seed-09-k2 -- 07's 300-episode recipe (200 of 06's train references + 100 train recoveries, one epoch at lr 1e-4,
# LoRA r16) mixed with 09_fresh_questions_data/dose_k2.jsonl (16 train2 episodes: k=2 for each of the 8 units 07's seed
# converted right at <= 6%), then its step-0 held-out probes and alive checks. Runs on CUDA or Apple silicon MPS.
#   ./09_fresh_questions_seed.sh
#   DOSE=09_fresh_questions_data/dose_k4.jsonl SEED_RUN=seed-09-k4 ./09_fresh_questions_seed.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
if [ "$(uname -s)" = Darwin ]; then
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.85}
fi
PY=${PY:-../../membrane_grpo/.venv/bin/python}
R=09_fresh_questions_runs
SEED_RUN=${SEED_RUN:-seed-09-k2}
DOSE=${DOSE:-09_fresh_questions_data/dose_k2.jsonl}
[ -f $R/$SEED_RUN/epoch1/adapter_model.safetensors ] || $PY 09_fresh_questions.py seed --out $SEED_RUN --label full     --reasoning 1 --epochs 1 --lr 1e-4 --accum 8 --lora-r 16 --limit 200 --recover 100     --extra "$DOSE" --extra-split train2
[ -f $R/$SEED_RUN/probes-step0/dev-T1-n120-stop.json ] || PPO_RUN=$SEED_RUN CKPTS="epoch1:0" ./09_fresh_questions_eval_heldout.sh
cp $R/$SEED_RUN/probes-step0/dev-T1-n120-stop.json $R/$SEED_RUN/epoch1/dev-T1-n120-stop.json
$PY 09_fresh_questions.py alive $R/$SEED_RUN/probes-step0/dev-T1-n120-stop.episodes.jsonl || true
$PY 09_fresh_questions_tools/units_args09.py $SEED_RUN=$R/$SEED_RUN/probes-step0/dev-T1-n120-stop.episodes.jsonl
