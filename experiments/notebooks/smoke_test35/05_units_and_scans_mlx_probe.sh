#!/usr/bin/env bash
# Untrained models on dev, through mlx-lm on Apple silicon: Qwen3.5-4B in 4-bit weights and
# Qwen3.5-0.8B as stored. Greedy, 2,048 policy tokens, float32 activations. Each model is run under
# probe's stop rule (the first {...} of a call-free turn) and then under the answer-object rule, which
# generates again only the episodes the first rule ended on a brace that was not an answer, plus 8 of
# the others as a check that carrying them over is exact.
#   PY      a python with mlx-lm and transformers
#   MLX_4B  the 4-bit model: $PY -m mlx_lm convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 4 --mlx-path <dir>
# The simulators need ../../../.venv-watertap, as for every other command of this notebook.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
MLX_4B=${MLX_4B:?the directory of the 4-bit Qwen3.5-4B}
$PY 05_units_and_scans.py mlx-check --split dev
for run in "mlx-raw-4b-4bit $MLX_4B" "mlx-raw-08b Qwen/Qwen3.5-0.8B"; do
    set -- $run
    $PY 05_units_and_scans.py mlx-probe --model "$2" --split dev --out "$1" --stop-at-json 1
    $PY 05_units_and_scans.py mlx-probe --model "$2" --split dev --out "$1" --stop-at-json 2 \
        --reuse "05_units_and_scans_runs/$1/dev-T0-stop.state.jsonl" --verify 8
done
