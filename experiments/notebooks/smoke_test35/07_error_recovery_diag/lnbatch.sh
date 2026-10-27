#!/usr/bin/env bash
# User 2026-10-19 「好」: --loss-norm batch (one token mean over the batch's trainable tokens) from the new PPO's step-200
# state (ppo-n300-maskend-turn-lr2e6-err0-s0/last/: adapter, critic, Adam moments, window, rng), 100 steps to 300, in a
# copy. Everything else as the new run's own 200 -> 400 continuation (the control, -to400): same flags, same python rng,
# so the same questions in the same order. The control's stop rule, pre-registered: at an evaluation N > 200, greedy
# decision < 0.65 or the window N-25..N-1 truncated >= 0.30 -> collapse-stop.json, stopped there.
# Then: the decision probe at 225/250/275/300 (KL by kind at 250, 300, as the control), and dev T=1 x2 (+ -err0 re-score)
# on this run's last step and on the control's same step, paired.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
R=07_error_recovery_runs
D=07_error_recovery_diag
RUN=ppo-n300-maskend-turn-lr2e6-err0-s0
CTRL=$RUN-to400
C=$RUN-from200-lnbatch
LOG=$R/ppo-lnbatch.log
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
[ "$STEP" = 200 ] || { echo "$(date +%H:%M) $RUN/last/ is at step $STEP, not 200"; exit 1; }
if [ ! -d $R/$C ]; then
    rsync -a --exclude step200 --exclude ending.json $R/$RUN/ $R/$C/
    cp $R/$RUN/ending.json $R/$C/ending-step200.json
    python3 -c "import json; p='$R/$C/config.json'; c=json.load(open(p)); c.update(steps=300, loss_norm='batch'); open(p,'w').write(json.dumps(c, indent=2) + '\n')"
fi
ARGS="--out $C --start-from $R/recovery-seed-n300-maskend/epoch1 --value-init-bias 0.4566 --steps 300 --eval-every 25 --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin 1 --lam 0.95 --adv-unit turn --sample-every 1 --lr 2e-6 --adv-center 0 --err-call-credit 0 --loss-norm batch"
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
python3 - $R/$C $R/$CTRL <<'EOF'
import json, sys
a, b = ({r["step"]: r for r in map(json.loads, open(f"{d}/metrics.jsonl"))} for d in sys.argv[1:])
keys = ("reward_mean", "adv_mean", "kl_mean", "policy_tokens_mean")
print("step 200 (same state, same rng; only the update differs):", "  ".join(f"{k} {a[200][k]:.4f}/{b[200][k]:.4f}" for k in keys))
EOF
N=$(python3 -c "import json, os; f='$R/$C/collapse-stop.json'; print(json.load(open(f))['step'] if os.path.exists(f) else json.load(open('$R/$C/ending.json'))['stopped_at'])" 2>/dev/null || echo 300)
[ -f $R/$C/collapse-stop.json ] && echo "$(date +%H:%M) stopped by the rule: $(cat $R/$C/collapse-stop.json)"
[ -f $R/$C/adapter-step$N/adapter_model.safetensors ] || { echo "$(date +%H:%M) no adapter-step$N; ending: $(cat $R/$C/ending.json 2>/dev/null | tr -d '\n')"; exit 1; }
wait_card; echo "$(date +%H:%M) decision probe $C"
(cd $D && ONLY_STEPS=225,250,275,300 $PY decision_probe.py decisions.jsonl lnb=$C:250,300)
for d in $C $CTRL; do
    if [ ! -f $R/$d/step$N/dev-T1-x2-stop.json ]; then
        wait_card; echo "$(date +%H:%M) probe x2 $d/step$N"
        $PY 07_error_recovery.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 --adapter $R/$d/adapter-step$N --out $d/step$N
    fi
    [ -f $R/$d/step$N/dev-T1-x2-stop-err0.json ] || $PY $D/rescore_err0.py $R/$d/step$N dev-T1-x2-stop
done
echo "== $C step $N vs the control's step $N"
$PY $D/compare_runs.py $R/$CTRL/step$N $R/$C/step$N dev-T1-x2-stop-err0 control lnbatch
echo "== $C step $N vs the new run's step 200"
$PY $D/compare_runs.py $R/$RUN/step200 $R/$C/step$N dev-T1-x2-stop-err0 new200 lnbatch
echo "$(date +%H:%M) lnbatch done"
