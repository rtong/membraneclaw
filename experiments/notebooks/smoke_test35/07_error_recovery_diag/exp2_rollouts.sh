#!/usr/bin/env bash
# Experiment 2 (user 2026-10-19): plain Monte-Carlo credit (lambda 0.95 on every turn, as the control) with four times
# the rollouts per step -- 32 distinct questions instead of 8 -- at half the learning rate (Adam normalises the step:
# 4x the episodes at lr / sqrt(4) keeps the useful drift per step and halves the random swing), and the per-turn call
# increments (--call-credit delta) as in experiment 1. Otherwise the n300-maskend run's configuration
# (ppo-n300-maskend-turn-lr2e6-err0-s0, the control), fresh from its seed, to step 200 and no further.
# Pre-registered gate: at step 200, dev T=1 x2 (-err0) paired against the seed, the reward difference's 95% interval
# above 0. Then the comparisons with the control. Going past 200 needs the user's go-ahead.
# Settings (environment): PY (training python), MAX_VRAM (MiB cap, default 13000), ROLLOUT_CHUNK (default 8),
# CRITIC_CHUNK (tokens, default 250000; 0 keeps the critic's window on the GPU), WAIT_MIB (wait until the card has
# less than this many MiB used before each GPU job; unset: no wait), NVSMI (nvidia-smi path).
#   nohup setsid bash 07_error_recovery_diag/exp2_rollouts.sh > 07_error_recovery_runs/exp2_rollouts.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=${PY:-$(cd ../../membrane_grpo/.venv/bin && pwd)/python}
MAX_VRAM=${MAX_VRAM:-13000}
ROLLOUT_CHUNK=${ROLLOUT_CHUNK:-8}
CRITIC_CHUNK=${CRITIC_CHUNK:-250000}
NVSMI=${NVSMI:-nvidia-smi}
R=07_error_recovery_runs
D=07_error_recovery_diag
SEED=$R/recovery-seed-n300-maskend/epoch1
CTRL=ppo-n300-maskend-turn-lr2e6-err0-s0
RUN=ppo-n300-maskend-turn-lr1e6-err0-delta-n32-s0
LOG=$R/ppo-$RUN.log
ARGS="--out $RUN --start-from $SEED --value-init-bias 0.4566 --steps 200 --eval-every 25 --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin 1 --lam 0.95 --adv-unit turn --sample-every 1 --lr 1e-6 --adv-center 0 --err-call-credit 0 --call-credit delta --prompts-per-step 32 --rollout-chunk $ROLLOUT_CHUNK --critic-chunk-tokens $CRITIC_CHUNK --max-vram-mib $MAX_VRAM"
wait_card() { [ -z "${WAIT_MIB:-}" ] && return; while [ "$($NVSMI --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt "$WAIT_MIB" ]; do sleep 60; done; }
last_step() { python3 -c "import json; print(json.load(open('$R/$RUN/ending.json'))['stopped_at'])" 2>/dev/null; }

# what the run needs, checked first
for f in $SEED/adapter_model.safetensors ../../../.venv-watertap/bin/python 07_error_recovery_data/dev_recovery_pr33.jsonl $R/$CTRL/step200/dev-T1-x2-stop-err0.json; do
    [ -e $f ] || { echo "missing $f"; exit 1; }
done
$PY -c "import torch, transformers, peft, fla; assert torch.cuda.is_available(); print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'fla', fla.__version__, torch.cuda.get_device_name(0))" || exit 1

for attempt in 1 2 3 4 5 6 7 8; do
    [ -f $R/$RUN/ending.json ] && break
    wait_card
    RES=""; [ -f $R/$RUN/last/state.pt ] && RES="--resume 1"
    echo "$(date '+%m-%d %H:%M') ppo $RUN, attempt $attempt ${RES:-(fresh)}" | tee -a $LOG
    $PY 07_error_recovery.py ppo $ARGS $RES >> $LOG 2>&1
done
N=$(last_step)
echo "$(date '+%m-%d %H:%M') $RUN ended at ${N:-?}: $(cat $R/$RUN/ending.json 2>/dev/null | tr -d '\n')"
python3 - $R/$RUN <<'EOF'
import json, statistics as st, sys
ev = {r["step"]: r for r in map(json.loads, open(f"{sys.argv[1]}/eval.jsonl"))}
m = [json.loads(l) for l in open(f"{sys.argv[1]}/metrics.jsonl")]
if 0 in ev:
    print(f"environment check: greedy eval at step 0 {ev[0]['reward']:.4f} (the control on the reference machine: 0.5954)")
print("greedy eval: " + " ".join(f"{s}:{e['reward']:.3f}" for s, e in sorted(ev.items())) + "   (control: 0:0.595 100:0.620 200:0.834)")
print(f"seconds per step: median {st.median(r['step_seconds'] for r in m):.0f} (generation {st.median(r['gen_seconds'] for r in m):.0f})")
EOF
[ "${N:-0}" = 200 ] && [ -f $R/$RUN/adapter-step200/adapter_model.safetensors ] || { echo "no step-200 checkpoint: gate failed"; exit 1; }
A=$R/$RUN/adapter-step200
if [ ! -f $R/$RUN/step200/dev-T1-x2-stop.json ]; then
    wait_card; echo "$(date '+%m-%d %H:%M') probe x2"
    $PY 07_error_recovery.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 --adapter $A --out $RUN/step200
fi
[ -f $R/$RUN/step200/dev-T1-x2-stop-err0.json ] || $PY $D/rescore_err0.py $R/$RUN/step200 dev-T1-x2-stop
if [ ! -f $R/$RUN/step200/dev-T0-stop-forced.json ]; then
    wait_card; echo "$(date '+%m-%d %H:%M') forced probe"
    $PY 07_error_recovery.py probe --split dev --temperature 0 --stop-at-json 1 --forced 07_error_recovery_data/dev_recovery_pr33.jsonl --adapter $A --out $RUN/step200
fi
[ -f $R/$RUN/step200/dev-T0-stop-forced-err0.json ] || $PY $D/rescore_err0.py $R/$RUN/step200 dev-T0-stop-forced
wait_card; echo "$(date '+%m-%d %H:%M') decision probe"
(cd $D && ONLY_STEPS=25,50,75,100,125,150,175,200 $PY decision_probe.py decisions.jsonl n32=$RUN:100)
echo "== gate: step 200 vs the seed (dev T=1 x2, -err0)"
$PY $D/compare_runs.py $SEED $R/$RUN/step200 dev-T1-x2-stop-err0 seed n32
echo "== step 200 vs the control's step 200"
$PY $D/compare_runs.py $R/$CTRL/step200 $R/$RUN/step200 dev-T1-x2-stop-err0 new200 n32
echo "== forced probe"
$PY $D/forced_report.py new200=$R/$CTRL/step200 n32-200=$R/$RUN/step200
echo "== route swings: mean |change of route log-odds| between checkpoints (sign changes), steps 25-200"
python3 - $D/decisions.jsonl <<'EOF'
import json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
G = ("train L1 2+ units", "train L2", "dev L1 2+ units", "dev L0")
for key in ("new", "n32"):
    sel = sorted((r for r in rows if r["run"] == key and 25 <= r["step"] <= 200), key=lambda r: r["step"])
    out = []
    for g in G:
        xs = [r["route"][g]["log_odds"] for r in sel if g in r["route"]]
        d = [b - a for a, b in zip(xs, xs[1:])]
        out.append(f"{g} {st.mean(abs(x) for x in d):.2f} ({sum(x * y < 0 for x, y in zip(d, d[1:]))} flips)" if d else f"{g} -")
    l0 = [r["route"]["dev L0"]["p_convert"] for r in sel if "dev L0" in r["route"]]
    print(f"{key:5} " + " | ".join(out) + (f" | dev L0 p(convert) mean {st.mean(l0):.3f}" if l0 else ""))
EOF
(cd $D && $PY window_mix.py n32 $RUN 0 200) 2>/dev/null
echo "$(date '+%m-%d %H:%M') experiment 2 done (stopped at 200)"
