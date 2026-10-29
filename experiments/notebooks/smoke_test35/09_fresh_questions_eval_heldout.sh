#!/usr/bin/env bash
# 09: the four held-out probes (greedy 120 test, greedy 120 holdout_shift, greedy forced recovery on 144 dev questions,
# and 120 dev questions sampled once at T=1) into 09_fresh_questions_runs/<run>/probes-step<N>/.
# Auto-selects MLX generation on macOS and merged PyTorch generation on Linux/CUDA.
#   ./09_fresh_questions_eval_heldout.sh
#   CKPTS="adapter-step200:200" PPO_RUN=ppo-... ./09_fresh_questions_eval_heldout.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=${PY:-../../membrane_grpo/.venv/bin/python}
R=09_fresh_questions_runs
if [ "$(uname -s)" = Darwin ]; then
    DEFAULT_RUN=ppo-mlx-qwen3.5-9b-l24-n32-lr1e6c-err0-delta-seed09k2-trainall-s0
    GENERATOR=${GENERATOR:-mlx}
    MERGE=${MERGE:-0}
else
    DEFAULT_RUN=ppo-qwen3.5-9b-l24-n32-lr1e6c-err0-delta-seed09k2-trainall-s0
    GENERATOR=${GENERATOR:-torch}
    MERGE=${MERGE:-1}
fi
PPO_RUN=${PPO_RUN:-$DEFAULT_RUN}
TEST_N=${TEST_N:-120}
DEV_N=${DEV_N:-120}
probe() {  # adapter, out, split, temperature, forced, limit
    local tag="$3-T$4$([ "$6" != 0 ] && echo -n$6)-stop$([ "$5" = 1 ] && echo -forced)"
    [ -f "$R/$2/$tag.json" ] || $PY 09_fresh_questions.py probe --split "$3" --stop-at-json 1 --temperature "$4"         --forced "$5" --limit "$6" --generator "$GENERATOR" --merge "$MERGE" --adapter "$1" --out "$2"
}
if [ -z "${CKPTS:-}" ]; then
    while [ ! -f "$R/$PPO_RUN/ending.json" ]; do sleep 60; done
    while pgrep -f "09_fresh_questions.py ppo --out $PPO_RUN" > /dev/null; do sleep 30; done
    read BEST STOP < <(python3 -c "import json; e = json.load(open('$R/$PPO_RUN/ending.json')); print(e['best_step'], e['stopped_at'])")
    CKPTS="adapter:$STOP"
    if [ "$BEST" != 0 ] && [ "$BEST" != "$STOP" ]; then CKPTS="adapter-best:$BEST $CKPTS"; fi
fi
for pair in $CKPTS; do
    ck=${pair%%:*}
    out=$PPO_RUN/probes-step${pair##*:}
    probe "$R/$PPO_RUN/$ck" "$out" test 0 0 "$TEST_N"
    probe "$R/$PPO_RUN/$ck" "$out" holdout_shift 0 0 0
    probe "$R/$PPO_RUN/$ck" "$out" dev 0 1 0
    probe "$R/$PPO_RUN/$ck" "$out" dev 1 0 "$DEV_N"
done
$PY 09_fresh_questions_tools/rescore_probes.py $(for pair in $CKPTS; do echo "$R/$PPO_RUN/probes-step${pair##*:}"; done)
