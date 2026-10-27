#!/usr/bin/env bash
# User 2026-10-17 「好的，没问题」: the run from recovery-seed-n300-maskend with err0 (ppo-n300-maskend-turn-lr2e6-err0-s0)
# continued from its last/ (step 200: adapter, critic, Adam moments, window, rng) to 400 in a copy, nothing else changed
# (--err-call-credit 0, no centring).
# Stop rule (pre-registered in the reply before the go-ahead): at an evaluation N > 200, greedy decision < 0.65 (the
# evals score under the new reward, so decision, not reward) or the training window N-25..N-1 truncated >= 0.30 ->
# collapse-stop.json and the run is stopped there (last/ = N). Then, at 400 or at the stop: dev T=1 x2 (+ the -err0
# re-score) and PR #33's forced probe on that step's adapter, to pair with step 200.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
D=07_error_recovery_diag
R=07_error_recovery_runs
RUN=ppo-n300-maskend-turn-lr2e6-err0-s0
C=$RUN-to400
LOG=$R/ppo-maskend-err0-to400.log
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
collapsed() {  # prints the reason and returns 0 once a checkpointed evaluation meets the collapse rule
    python3 - "$R/$C" "$LOG" <<'EOF'
import json, re, sys
d, log = sys.argv[1], open(sys.argv[2]).read()
ev = [json.loads(l) for l in open(f"{d}/eval.jsonl")]
m = [json.loads(l) for l in open(f"{d}/metrics.jsonl")]
for e in ev:
    n = e["step"]
    if n <= 200 or not re.search(rf"eval @{n}\b", log):   # the eval line is printed after save_state
        continue
    w = [r["truncated"] for r in m if n - 25 <= r["step"] < n]
    tr = sum(w) / len(w) if w else 0.0
    if e["decision_correct"] < 0.65 or (len(w) == 25 and tr >= 0.30):
        rec = {"step": n, "eval_decision": e["decision_correct"], "eval_reward": e["reward"], "window_truncated": tr}
        open(f"{d}/collapse-stop.json", "w").write(json.dumps(rec) + "\n")
        print(json.dumps(rec)); sys.exit(0)
sys.exit(1)
EOF
}
STEP=$($PY -c "import torch; print(torch.load('$R/$RUN/last/state.pt', map_location='cpu', weights_only=False)['step'])" 2>/dev/null)
[ "$STEP" = 200 ] || { echo "$(date +%H:%M) last/ is at step $STEP, not 200"; exit 1; }
if [ ! -d $R/$C ]; then
    rsync -a --exclude step200 --exclude ending.json $R/$RUN/ $R/$C/
    cp $R/$RUN/ending.json $R/$C/ending-step200.json
    python3 -c "import json; p='$R/$C/config.json'; c=json.load(open(p)); c.update(steps=400); open(p,'w').write(json.dumps(c, indent=2) + '\n')"
fi
ARGS="--out $C --start-from $R/recovery-seed-n300-maskend/epoch1 --value-init-bias 0.4566 --steps 400 --eval-every 25 --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin 1 --lam 0.95 --adv-unit turn --sample-every 1 --lr 2e-6 --adv-center 0 --err-call-credit 0"
for attempt in 1 2 3 4 5 6 7 8; do
    [ -f $R/$C/ending.json ] || [ -f $R/$C/collapse-stop.json ] && break
    wait_card
    echo "$(date +%H:%M) ppo $C attempt $attempt (resume from last/)"
    $PY 07_error_recovery.py ppo $ARGS --resume 1 &
    P=$!
    while kill -0 $P 2>/dev/null; do
        sleep 60
        if why=$(collapsed); then
            echo "$(date +%H:%M) collapse rule: $why -- stopping python $P"
            kill $P; sleep 15; kill -0 $P 2>/dev/null && kill -9 $P
            break
        fi
    done
    wait $P 2>/dev/null
done
N=$(python3 -c "import json, os; f='$R/$C/collapse-stop.json'; print(json.load(open(f))['step'] if os.path.exists(f) else json.load(open('$R/$C/ending.json'))['stopped_at'])" 2>/dev/null || echo 400)
[ -f $R/$C/collapse-stop.json ] && echo "$(date +%H:%M) stopped by the rule: $(cat $R/$C/collapse-stop.json)"
A=$R/$C/adapter-step$N
[ -f $A/adapter_model.safetensors ] || { echo "$(date +%H:%M) no adapter-step$N; ending: $(cat $R/$C/ending.json 2>/dev/null | tr -d '\n')"; exit 1; }
if [ ! -f $R/$C/step$N/dev-T1-x2-stop.json ]; then
    wait_card; echo "$(date +%H:%M) probe x2 $C/step$N"
    $PY 07_error_recovery.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 --adapter $A --out $C/step$N
fi
[ -f $R/$C/step$N/dev-T1-x2-stop-err0.json ] || $PY $D/rescore_err0.py $R/$C/step$N dev-T1-x2-stop
if [ ! -f $R/$C/step$N/dev-T0-stop-forced.json ]; then
    wait_card; echo "$(date +%H:%M) forced probe $C/step$N"
    $PY 07_error_recovery.py probe --split dev --temperature 0 --stop-at-json 1 --forced 07_error_recovery_data/dev_recovery_pr33.jsonl \
        --adapter $A --out $C/step$N
fi
echo "$(date +%H:%M) new to400 done"
