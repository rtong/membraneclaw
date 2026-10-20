#!/usr/bin/env bash
# Experiment 1 (user 2026-10-19): the first turn -- the convert-or-simulate decision -- credited by the critic's value of
# the state it led to (--first-turn-lam 0: r_0 + V(s_1) - V(s_0); and 0.5), with the per-turn call increments
# (--call-credit delta) so that r_0 does not depend on later calls. Otherwise the n300-maskend run's configuration
# (ppo-n300-maskend-turn-lr2e6-err0-s0, the control), fresh from its seed. Pre-registered:
#   gate 0  td_precheck.py on the control's step-200 state (informational here: both lambdas run either way);
#   gate    at step 200, dev T=1 x2 (-err0) paired against the seed: the reward difference's 95% interval above 0,
#           or the run stops at 200;
#   rule    from 200 to 300: at an evaluation N > 200, greedy decision < 0.65 or training truncation >= 0.30 over
#           steps N-25..N-1 -> collapse-stop.json, stopped at N.
# Order: td0 to 200, td05 to 200, then each run that passed its gate on to 300. Probes at 200 and at the last step;
# the decision probe on every checkpoint; comparisons with the seed, the control and each other at the end.
# Waits for the --loss-norm batch run's chain (LNB_PID) first. Log: 07_error_recovery_runs/exp1_td.log
#   nohup setsid bash 07_error_recovery_diag/exp1_td.sh > 07_error_recovery_runs/exp1_td.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
R=07_error_recovery_runs
D=07_error_recovery_diag
LNB_PID=${LNB_PID:-110946}
SEED=$R/recovery-seed-n300-maskend/epoch1
CTRL=ppo-n300-maskend-turn-lr2e6-err0-s0
BASE="--start-from $SEED --value-init-bias 0.4566 --eval-every 25 --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin 1 --lam 0.95 --adv-unit turn --sample-every 1 --lr 2e-6 --adv-center 0 --err-call-credit 0 --call-credit delta"
declare -A LAMS=([td0]=0 [td05]=0.5)
run_name() { echo "ppo-n300-maskend-turn-lr2e6-err0-delta-$1-s0"; }
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
collapsed() {  # <run dir> <log>: prints the reason and returns 0 once a checkpointed evaluation N > 200 meets the rule
    python3 - "$1" "$2" <<'EOF'
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
last_step() { python3 -c "import json, os; d='$R/$1'; f=d+'/collapse-stop.json'; print(json.load(open(f))['step'] if os.path.exists(f) else json.load(open(d+'/ending.json'))['stopped_at'])" 2>/dev/null; }
train() {  # <run> <steps> <lam>: up to 8 attempts, each resuming from last/ when there is one
    local RUN=$1 STEPS=$2 LAM=$3 LOG=$R/ppo-$1.log
    for attempt in 1 2 3 4 5 6 7 8; do
        { [ -f $R/$RUN/ending.json ] || [ -f $R/$RUN/collapse-stop.json ]; } && break
        wait_card
        local RES=""; [ -f $R/$RUN/last/state.pt ] && RES="--resume 1"
        echo "$(date '+%m-%d %H:%M') ppo $RUN to $STEPS, attempt $attempt ${RES:-(fresh)}" | tee -a $LOG
        $PY 07_error_recovery.py ppo --out $RUN $BASE --first-turn-lam $LAM --steps $STEPS $RES >> $LOG 2>&1 &
        local P=$!
        while kill -0 $P 2>/dev/null; do
            sleep 60
            if why=$(collapsed $R/$RUN $LOG 2>/dev/null); then
                echo "$(date '+%m-%d %H:%M') collapse rule: $why -- stopping python $P" | tee -a $LOG
                kill $P; sleep 15; kill -0 $P 2>/dev/null && kill -9 $P
                break
            fi
        done
        wait $P 2>/dev/null
    done
}
probes() {  # <run> <step>: dev T=1 x2 and the forced probe on adapter-step<step>, each re-scored -err0
    local RUN=$1 N=$2 A=$R/$1/adapter-step$2
    [ -f $A/adapter_model.safetensors ] || { echo "no $A"; return 1; }
    if [ ! -f $R/$RUN/step$N/dev-T1-x2-stop.json ]; then
        wait_card; echo "$(date '+%m-%d %H:%M') probe x2 $RUN/step$N"
        $PY 07_error_recovery.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 --adapter $A --out $RUN/step$N
    fi
    [ -f $R/$RUN/step$N/dev-T1-x2-stop-err0.json ] || $PY $D/rescore_err0.py $R/$RUN/step$N dev-T1-x2-stop
    if [ ! -f $R/$RUN/step$N/dev-T0-stop-forced.json ]; then
        wait_card; echo "$(date '+%m-%d %H:%M') forced probe $RUN/step$N"
        $PY 07_error_recovery.py probe --split dev --temperature 0 --stop-at-json 1 --forced 07_error_recovery_data/dev_recovery_pr33.jsonl --adapter $A --out $RUN/step$N
    fi
    [ -f $R/$RUN/step$N/dev-T0-stop-forced-err0.json ] || $PY $D/rescore_err0.py $R/$RUN/step$N dev-T0-stop-forced
}
decisions() {  # <key> <run> <steps, comma list>
    wait_card; echo "$(date '+%m-%d %H:%M') decision probe $1 at $3"
    (cd $D && ONLY_STEPS=$3 $PY decision_probe.py decisions.jsonl $1=$2:100)
}
paired() {  # <A dir> <B dir> <labelA> <labelB> [stem]: compare_runs; prints its paired lines
    $PY $D/compare_runs.py $1 $2 ${5:-dev-T1-x2-stop-err0} $3 $4
}
gate200() {  # <run>: 0 if the step-200 reward difference against the seed has its 95% interval above 0
    paired $SEED $R/$1/step200 seed $1 > $R/$1/gate200.txt 2>&1
    python3 - $R/$1/gate200.txt $R/$1/gate200.json <<'EOF'
import json, re, sys
t = open(sys.argv[1]).read()
m = re.search(r"- seed, reward\s+([+-][\d.]+)\s+95% interval ([+-][\d.]+) to ([+-][\d.]+)", t)
rec = {"diff": float(m[1]), "lo": float(m[2]), "hi": float(m[3])} if m else {"error": "no reward line"}
rec["passed"] = bool(m) and rec["lo"] > 0
open(sys.argv[2], "w").write(json.dumps(rec) + "\n"); print(json.dumps(rec))
sys.exit(0 if rec["passed"] else 1)
EOF
}

while kill -0 $LNB_PID 2>/dev/null; do sleep 60; done
echo "$(date '+%m-%d %H:%M') the --loss-norm batch chain is done; experiment 1 starts"
if [ ! -f $D/td_precheck.json ]; then
    wait_card; echo "$(date '+%m-%d %H:%M') gate 0: td_precheck"
    (cd $D && $PY td_precheck.py) || echo "td_precheck FAILED -- continuing (informational)"
fi
# Leg 1: each lambda to 200, probes, decisions, gate
for K in td0 td05; do
    RUN=$(run_name $K)
    train $RUN 200 ${LAMS[$K]}
    N=$(last_step $RUN)
    echo "$(date '+%m-%d %H:%M') $RUN ended at ${N:-?}: $(cat $R/$RUN/collapse-stop.json $R/$RUN/ending.json 2>/dev/null | tr -d '\n')"
    if [ "${N:-0}" = 200 ] && probes $RUN 200; then
        decisions $K $RUN 25,50,75,100,125,150,175,200
        if gate200 $RUN; then echo "$(date '+%m-%d %H:%M') $RUN passed the step-200 gate"; else echo "$(date '+%m-%d %H:%M') $RUN FAILED the step-200 gate: stops at 200"; fi
    else
        echo "$(date '+%m-%d %H:%M') $RUN has no step-200 checkpoint: gate failed"
        echo '{"passed": false, "error": "no step-200 checkpoint"}' > $R/$RUN/gate200.json
    fi
done
# Leg 2: the runs that passed, 200 -> 300
for K in td0 td05; do
    RUN=$(run_name $K)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$R/$RUN/gate200.json')).get('passed') else 1)" || continue
    if [ -f $R/$RUN/ending.json ] && [ "$(last_step $RUN)" = 200 ] && [ ! -f $R/$RUN/ending-step200.json ]; then
        mv $R/$RUN/ending.json $R/$RUN/ending-step200.json
        python3 -c "import json; p='$R/$RUN/config.json'; c=json.load(open(p)); c.update(steps=300); open(p,'w').write(json.dumps(c, indent=2) + '\n')"
    fi
    train $RUN 300 ${LAMS[$K]}
    N=$(last_step $RUN)
    echo "$(date '+%m-%d %H:%M') $RUN ended at ${N:-?}: $(cat $R/$RUN/collapse-stop.json $R/$RUN/ending.json 2>/dev/null | tr -d '\n')"
    [ -n "$N" ] && [ "$N" -gt 200 ] && probes $RUN $N && decisions $K $RUN $(seq -s, 225 25 $N)
done
# Comparisons
echo "== comparisons ($(date '+%m-%d %H:%M'))"
for K in td0 td05; do
    RUN=$(run_name $K)
    [ -f $R/$RUN/step200/dev-T1-x2-stop-err0.json ] || continue
    echo "-- $RUN step 200 vs the control's step 200"; paired $R/$CTRL/step200 $R/$RUN/step200 new200 $K | grep -E "^   $K - new200"
    N=$(last_step $RUN)
    if [ "${N:-0}" -gt 200 ] && [ -f $R/$RUN/step$N/dev-T1-x2-stop-err0.json ]; then
        echo "-- $RUN step $N vs its step 200"; paired $R/$RUN/step200 $R/$RUN/step$N ${K}200 ${K}$N | grep -E "^   ${K}$N - ${K}200"
        if [ ! -f $R/$CTRL-to400/step$N/dev-T1-x2-stop-err0.json ]; then probes $CTRL-to400 $N; fi
        echo "-- $RUN step $N vs the control's step $N"; paired $R/$CTRL-to400/step$N $R/$RUN/step$N new$N $K | grep -E "^   $K - new$N"
    fi
done
A=$(run_name td0); B=$(run_name td05)
if [ -f $R/$A/step200/dev-T1-x2-stop-err0.json ] && [ -f $R/$B/step200/dev-T1-x2-stop-err0.json ]; then
    echo "-- td05 vs td0, step 200"; paired $R/$A/step200 $R/$B/step200 td0 td05
fi
echo "-- forced probe"
$PY $D/forced_report.py new200=$R/$CTRL/step200 $(for K in td0 td05; do RUN=$(run_name $K); for d in $R/$RUN/step*/; do [ -f $d/dev-T0-stop-forced.episodes.jsonl ] && echo "$K-$(basename $d)=$d"; done; done)
echo "-- route swings: mean |change of route log-odds| between checkpoints, and sign changes"
python3 - $D/decisions.jsonl <<'EOF'
import json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
G = ("train L1 2+ units", "train L2", "dev L1 2+ units", "dev L0")
for key in ("new", "td0", "td05"):
    rs = sorted((r for r in rows if r["run"] == key), key=lambda r: r["step"])
    for lo, hi in ((25, 200), (200, 300)):
        sel = [r for r in rs if lo <= r["step"] <= hi]
        out = []
        for g in G:
            xs = [r["route"][g]["log_odds"] for r in sel if g in r["route"]]
            d = [b - a for a, b in zip(xs, xs[1:])]
            flips = sum(x * y < 0 for x, y in zip(d, d[1:]))
            out.append(f"{g} {st.mean(abs(x) for x in d):.2f} ({flips} flips)" if d else f"{g} -")
        l0 = [r["route"]["dev L0"]["p_convert"] for r in sel if "dev L0" in r["route"]]
        print(f"{key:5} {lo}-{hi}: " + " | ".join(out) + (f" | dev L0 p(convert) mean {st.mean(l0):.3f}" if l0 else ""))
EOF
for K in td0 td05; do (cd $D && $PY window_mix.py $K $(run_name $K) 0 300) 2>/dev/null; done
echo "$(date '+%m-%d %H:%M') experiment 1 done"
