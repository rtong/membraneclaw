#!/usr/bin/env bash
# The other notebook 07's seed (Rui Tong's, commit effa2e4), trained here: the first 300 of 06's reference episodes,
# every fourth replaced by its own question's recovery episode (225 + 75: 36 unconverted inputs, 39 x10 / x0.1 factor
# slips; his recovery data, converted: 08_prover_critic_data/train_recovery_effa2e4.jsonl), one epoch at lr 1e-4,
# LoRA r16 -- his settings. On the same 120 dev questions at T=1 his seed beat 07's (0.490 against 0.430, paired
# -0.060, 95% -0.118 to -0.001), so 08 takes its recipe: this seed is probed as 07's were (the 120 dev questions at
# T=1, the forced probe) and 08's PPO starts from it.
#   ./08_prover_critic_seed.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
R=08_prover_critic_runs
S=${S:-seed-rt-n300-e4}
DEV_N=${DEV_N:-120}
[ -f $R/$S/epoch1/adapter_model.safetensors ] || $PY 08_prover_critic.py seed --out $S --label full --reasoning 1 \
    --epochs 1 --lr 1e-4 --accum 8 --lora-r 16 --limit 300 --recovery-every 4
[ -f $R/$S/epoch1/dev-T1-n$DEV_N-stop.json ] || $PY 08_prover_critic.py probe --split dev --stop-at-json 1 \
    --temperature 1 --limit $DEV_N --merge 1 --adapter $R/$S/epoch1 --out $S/epoch1
$PY 08_prover_critic.py alive $R/$S/epoch1/dev-T1-n$DEV_N-stop.episodes.jsonl || true
[ -f $R/$S/epoch1/dev-T0-stop-forced.json ] || $PY 08_prover_critic.py probe --split dev --stop-at-json 1 \
    --temperature 0 --forced 1 --merge 1 --adapter $R/$S/epoch1 --out $S/epoch1
