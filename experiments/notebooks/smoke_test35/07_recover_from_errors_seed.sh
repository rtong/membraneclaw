#!/usr/bin/env bash
# The seed, small and fixed in advance -- not swept. 06 went 300 -> 600 -> 1,200 episodes each time PPO did not
# activate a dose; 300 was already alive by its rule. 07 stays at 300 and changes what is in it: the first N=200 of
# 06's reference episodes (06's shuffle) plus M=100 recovery episodes of other questions, one epoch, 06's SFT settings
# (every token of the calls and the answer labelled, the comparison lines, lr 1e-4, one step per 8 episodes, LoRA r16);
# a recovery episode's slipped turns are context, never labelled. The control is the same 300 with no recovery
# episodes -- 06's own 300 seed, retrained here. Each is probed on dev two ways, sized for Apple silicon (where a
# batch of 16 sampled episodes takes minutes, not seconds): 120 dev questions (the same 120 for every policy) sampled
# once at T=1 -- judged by `alive`, 07's recovery label included; its reward warm-starts PPO's value head -- and the
# forced probe, greedy: every dev question with a slip (144) started from its slipped turns. Probes fold the adapter
# into the weights for generation (--merge 1).
#   ./07_recover_from_errors_seed.sh               the seed and the control, and their probes
#   N=200 M=100 ./07_recover_from_errors_seed.sh   (the default) another split of the dose: seed-n<N>-r<M>, control seed-n<N+M>-r0
#   CONTROL=0 ./...                                the seed only
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
R=07_recover_from_errors_runs
N=${N:-200}; M=${M:-100}
DEV_N=${DEV_N:-120}
RUNS="seed-n$N-r$M:$N:$M"
[ "${CONTROL:-1}" = 1 ] && RUNS="$RUNS seed-n$((N + M))-r0:$((N + M)):0"
for spec in $RUNS; do
    IFS=: read -r run limit recover <<< "$spec"
    [ -f $R/$run/epoch1/adapter_model.safetensors ] || $PY 07_recover_from_errors.py seed --out $run --label full --reasoning 1 \
        --epochs 1 --lr 1e-4 --accum 8 --lora-r 16 --limit $limit --recover $recover
    [ -f $R/$run/epoch1/dev-T1-n$DEV_N-stop.json ] || $PY 07_recover_from_errors.py probe --split dev --stop-at-json 1 \
        --temperature 1 --limit $DEV_N --merge 1 --adapter $R/$run/epoch1 --out $run/epoch1
    $PY 07_recover_from_errors.py alive $R/$run/epoch1/dev-T1-n$DEV_N-stop.episodes.jsonl || true
    [ -f $R/$run/epoch1/dev-T0-stop-forced.json ] || $PY 07_recover_from_errors.py probe --split dev --stop-at-json 1 \
        --temperature 0 --forced 1 --merge 1 --adapter $R/$run/epoch1 --out $run/epoch1
done
