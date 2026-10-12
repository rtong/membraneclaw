#!/usr/bin/env bash
# The diagnostic this notebook opens with: 05's label seed, under 05's own tool declarations, on
# holdout_shift with every limit restated by 05's first train sentence, and on test with every limit
# restated by holdout_shift's unseen sentence. Greedy, answers stopped at their closing brace. The
# numbers, the units and every other sentence stay as they are. A probe of the sentence, not a
# benchmark: the splits themselves are never changed, and nothing here is trained on.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=../../membrane_grpo/.venv/bin/python
SEED05=05_units_and_scans_runs/label-seed-pc2/epoch3
$PY 06_direction_and_compare.py probe --split holdout_shift --swap shared --stop-at-json 1 --max-new-tokens 2048 \
    --tools-json 05_units_and_scans_tools.json --adapter $SEED05 --out seed05-swap
$PY 06_direction_and_compare.py probe --split test --swap holdout --stop-at-json 1 --max-new-tokens 2048 \
    --tools-json 05_units_and_scans_tools.json --adapter $SEED05 --out seed05-swap
