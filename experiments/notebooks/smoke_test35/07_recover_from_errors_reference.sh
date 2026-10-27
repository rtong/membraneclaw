#!/usr/bin/env bash
# 07's one new kind of data: the recovery episodes. For every train and dev question with an input that needs
# converting, one such input (drawn per question) is left unconverted; when every simulate call of that turn errs,
# the simulator's responses are kept and the fix follows -- a line naming the input, its number and the unit the
# question gives it in, the convert_units call, the simulate turn again, the check turn, the answer (06's). The train
# episodes feed the seed; the dev ones only give the forced probe its starting turns. The questions and the blind
# reference episodes are 06's, read in place. Then the harness checks (token-for-token against the chat template,
# the fix line included) and the data checks.
#   ./07_recover_from_errors_reference.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
W=../../../.venv-watertap/bin/python    # the simulators
PY=../../membrane_grpo/.venv/bin/python # the tokenizer (check-harness); its tool workers run under $W
for S in train dev; do
    [ -f 07_recover_from_errors_data/${S}_recovery.jsonl ] || $W 07_recover_from_errors.py reference --recover 1 --split $S
done
$PY 07_recover_from_errors.py check-harness --split dev --n 60
python3 07_recover_from_errors.py check --decontam 0
