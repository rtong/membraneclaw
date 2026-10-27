#!/usr/bin/env bash
# Experiment 3 (user 2026-10-21): the actor's lambda 0.5 on every turn, the critic's targets still lambda 0.95
# (--actor-lam 0.5, decoupled GAE). Otherwise experiment 1's configuration (delta call credit, lr 2e-6, 8 prompts per
# step, the control's flags), fresh from the n300-maskend seed, to step 200 and no further.
# Why: first-turn TD alone (td0, td05) halved the first turn's advantage noise and the route still swung as much as the
# control, every question group moving together; the later turns, whose tool calls kept Monte-Carlo credit, may carry it.
# Pre-registered gate: at step 200, dev T=1 x2 (-err0) paired against the seed, the reward difference's 95% interval
# above 0. Primary readout: the route swings 25-200 (train and dev L1 2+ units at most half the control's, dev L0
# p(convert) no higher). Added readout: each swing split into its common part (the mean change over the four dev
# groups) and its differential part (the change of dev L1 2+ minus dev L0). Going past 200 needs the user's go-ahead.
#   nohup setsid bash 07_error_recovery_diag/exp3_actorlam.sh > 07_error_recovery_runs/exp3_actorlam.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
R=07_error_recovery_runs
D=07_error_recovery_diag
SEED=$R/recovery-seed-n300-maskend/epoch1
CTRL=ppo-n300-maskend-turn-lr2e6-err0-s0
TD0=ppo-n300-maskend-turn-lr2e6-err0-delta-td0-s0
TD05=ppo-n300-maskend-turn-lr2e6-err0-delta-td05-s0
RUN=ppo-n300-maskend-turn-lr2e6-err0-delta-alam05-s0
LOG=$R/ppo-$RUN.log
ARGS="--out $RUN --start-from $SEED --value-init-bias 0.4566 --steps 200 --eval-every 25 --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin 1 --lam 0.95 --adv-unit turn --sample-every 1 --lr 2e-6 --adv-center 0 --err-call-credit 0 --call-credit delta --actor-lam 0.5"
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
last_step() { python3 -c "import json; print(json.load(open('$R/$RUN/ending.json'))['stopped_at'])" 2>/dev/null; }
paired() { $PY $D/compare_runs.py $1 $2 dev-T1-x2-stop-err0 $3 $4; }

for attempt in 1 2 3 4 5 6 7 8; do
    [ -f $R/$RUN/ending.json ] && break
    wait_card
    RES=""; [ -f $R/$RUN/last/state.pt ] && RES="--resume 1"
    echo "$(date '+%m-%d %H:%M') ppo $RUN, attempt $attempt ${RES:-(fresh)}" | tee -a $LOG
    $PY 07_error_recovery.py ppo $ARGS $RES >> $LOG 2>&1
done
N=$(last_step)
echo "$(date '+%m-%d %H:%M') $RUN ended at ${N:-?}: $(cat $R/$RUN/ending.json 2>/dev/null | tr -d '\n')"
[ "${N:-0}" = 200 ] && [ -f $R/$RUN/adapter-step200/adapter_model.safetensors ] || { echo "no step-200 checkpoint: gate failed"; echo '{"passed": false, "error": "no step-200 checkpoint"}' > $R/$RUN/gate200.json; exit 1; }
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
(cd $D && ONLY_STEPS=25,50,75,100,125,150,175,200 $PY decision_probe.py decisions.jsonl alam05=$RUN:100)

echo "== gate: step 200 vs the seed (dev T=1 x2, -err0)"
paired $SEED $R/$RUN/step200 seed alam05 > $R/$RUN/gate200.txt 2>&1
cat $R/$RUN/gate200.txt
python3 - $R/$RUN/gate200.txt $R/$RUN/gate200.json <<'EOF'
import json, re, sys
t = open(sys.argv[1]).read()
m = re.search(r"- seed, reward\s+([+-][\d.]+)\s+95% interval ([+-][\d.]+) to ([+-][\d.]+)", t)
rec = {"diff": float(m[1]), "lo": float(m[2]), "hi": float(m[3])} if m else {"error": "no reward line"}
rec["passed"] = bool(m) and rec["lo"] > 0
open(sys.argv[2], "w").write(json.dumps(rec) + "\n"); print("gate:", json.dumps(rec))
EOF
echo "== step 200 vs the control's step 200"; paired $R/$CTRL/step200 $R/$RUN/step200 new200 alam05
echo "== step 200 vs td05's step 200"; paired $R/$TD05/step200 $R/$RUN/step200 td05 alam05 | grep -E "^   alam05 - td05"
echo "== step 200 vs td0's step 200"; paired $R/$TD0/step200 $R/$RUN/step200 td0 alam05 | grep -E "^   alam05 - td0"
echo "== forced probe"
$PY $D/forced_report.py new200=$R/$CTRL/step200 td0-200=$R/$TD0/step200 td05-200=$R/$TD05/step200 alam05-200=$R/$RUN/step200
echo "== route swings, steps 25-200: mean |change of route log-odds| between checkpoints (sign changes); common = mean change over the four dev groups, differential = change of dev L1 2+ minus dev L0"
python3 - $D/decisions.jsonl <<'EOF'
import json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
G = ("train L1 2+ units", "train L2", "dev L1 2+ units", "dev L0")
DEV = ("dev L0", "dev L1 one unit", "dev L1 2+ units", "dev L2")
for key in ("new", "td0", "td05", "alam05"):
    sel = sorted((r for r in rows if r["run"] == key and 25 <= r["step"] <= 200), key=lambda r: r["step"])
    if len(sel) < 2:
        print(f"{key:6} -"); continue
    out = []
    for g in G:
        xs = [r["route"][g]["log_odds"] for r in sel]
        d = [b - a for a, b in zip(xs, xs[1:])]
        out.append(f"{g} {st.mean(abs(x) for x in d):.2f} ({sum(x * y < 0 for x, y in zip(d, d[1:]))})")
    dd = [[b["route"][g]["log_odds"] - a["route"][g]["log_odds"] for g in DEV] for a, b in zip(sel, sel[1:])]
    common = [st.mean(x) for x in dd]
    diff = [x[2] - x[0] for x in dd]
    same = sum(len({v > 0 for v in x if abs(v) > 0.05}) == 1 for x in dd)
    l0 = [r["route"]["dev L0"]["p_convert"] for r in sel]
    print(f"{key:6} " + " | ".join(out) + f" | dev L0 p(convert) {st.mean(l0):.3f} | common {st.mean(abs(x) for x in common):.2f}"
          f" differential {st.mean(abs(x) for x in diff):.2f} | all four dev groups one way {same}/{len(dd)}")
    print("       dev L1 2+ p(convert): " + " ".join(f"{r['step']}:{r['route']['dev L1 2+ units']['p_convert']:.2f}" for r in sel))
EOF
(cd $D && $PY window_mix.py alam05 $RUN 0 200) 2>/dev/null
echo "$(date '+%m-%d %H:%M') experiment 3 done (stopped at 200)"
