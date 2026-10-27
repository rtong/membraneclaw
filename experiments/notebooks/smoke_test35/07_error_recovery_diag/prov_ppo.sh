#!/usr/bin/env bash
# The provenance run (user 2026-10-23, "好的，继续": the strict rule, every unit including % -> fraction): the
# control's configuration (n300-maskend seed, turn advantage, lr 2e-6, err0, call credit "call") plus --provenance 1 --
# an input the question states in another unit proves its point only when a convert_units call of an earlier turn
# returned it, so an input converted in the head earns no evidence (the calls, values and format components unchanged).
# Fresh from the seed, to step 200 and no further.
# Why: from step 100 the PPO runs pass L1 inputs converted in the head, the reward pays both routes alike (L1 one
# unit, within question, -0.03 to -0.06) and the route drifts; their training samples re-scored with provenance give
# the convert-first route back its margin on L1 2+ units (+0.16 to +0.19) and part of it on L1 one unit (about +0.1).
# The critic's initial bias is the seed's dev T=1 x2 reward under this run's reward (err0 + provenance), as the
# control's 0.4566 was its dev T=1 x2 reward under err0.
# Pre-registered gate: at step 200, dev T=1 x2 scored err0 + provenance, paired against the seed scored the same way,
# the reward difference's 95% interval above 0 (held-out probes only when it passes). Readouts: prov_primary.py (its
# route swings against the control's, its own route contrast); test and holdout_shift T=1 x2 against the control's
# step 200 under the old reward (err0) and under err0 + provenance; the forced probe. Going past 200 needs the user.
# Needs the trainer with PROVENANCE (the flag is put in once the seed held-out probes no longer use the trainer).
#   nohup setsid bash 07_error_recovery_diag/prov_ppo.sh > 07_error_recovery_runs/prov_ppo.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
PY=$(cd ../../membrane_grpo/.venv/bin && pwd)/python
R=07_error_recovery_runs
D=07_error_recovery_diag
SEED=$R/recovery-seed-n300-maskend/epoch1
CTRL=ppo-n300-maskend-turn-lr2e6-err0-s0
ALAM=ppo-n300-maskend-turn-lr2e6-err0-delta-alam05-s0
RUN=ppo-n300-maskend-turn-lr2e6-err0-prov-s0
LOG=$R/ppo-$RUN.log
grep -q "^PROVENANCE = False" 07_error_recovery.py || { echo "the trainer has no PROVENANCE flag"; exit 1; }
[ -f $SEED/dev-T1-x2-stop-err0-prov.json ] || $PY $D/rescore_prov.py $SEED dev-T1-x2-stop || exit 1
BIAS=$(python3 -c "import json; print(f\"{json.load(open('$SEED/dev-T1-x2-stop-err0-prov.json'))['overall']['reward']:.4f}\")")
ARGS="--out $RUN --start-from $SEED --value-init-bias $BIAS --steps 200 --eval-every 25 --tool-stop 0 --tool-stop-rel 0 --answer-stop-rel 0.5 --regress-margin 1 --lam 0.95 --adv-unit turn --sample-every 1 --lr 2e-6 --adv-center 0 --err-call-credit 0 --provenance 1"
wait_card() { while [ "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 60; done; }
last_step() { python3 -c "import json; print(json.load(open('$R/$RUN/ending.json'))['stopped_at'])" 2>/dev/null; }
probe() {  # split, T, samples
    local tag=$1-T$2$([ "$3" -gt 1 ] && echo -x$3)-stop
    if [ ! -f $R/$RUN/step200/$tag.json ]; then
        wait_card; echo "$(date '+%m-%d %H:%M') probe $tag"
        $PY 07_error_recovery.py probe --split $1 --stop-at-json 1 --temperature $2 --samples $3 --adapter $A --out $RUN/step200
    fi
    [ -f $R/$RUN/step200/$tag.json ] || { echo "$(date '+%m-%d %H:%M') no $tag"; exit 1; }
    [ -f $R/$RUN/step200/$tag-err0.json ] || $PY $D/rescore_err0.py $R/$RUN/step200 $tag
    [ -f $R/$RUN/step200/$tag-err0-prov.json ] || $PY $D/rescore_prov.py $R/$RUN/step200 $tag
}
rescored() {  # dir, tag: the err0 + provenance re-scoring of a probe already there
    [ -f $1/$2-err0-prov.json ] || $PY $D/rescore_prov.py $1 $2
}

echo "$(date '+%m-%d %H:%M') value-init bias $BIAS (the seed's dev T=1 x2 reward, err0 + provenance)"
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
probe dev 1 2
echo "== gate: step 200 vs the seed (dev T=1 x2, err0 + provenance)"
$PY $D/compare_runs.py $SEED $R/$RUN/step200 dev-T1-x2-stop-err0-prov seed prov200 > $R/$RUN/gate200.txt 2>&1
cat $R/$RUN/gate200.txt
python3 - $R/$RUN/gate200.txt $R/$RUN/gate200.json <<'EOF'
import json, re, sys
t = open(sys.argv[1]).read()
m = re.search(r"- seed, reward\s+([+-][\d.]+)\s+95% interval ([+-][\d.]+) to ([+-][\d.]+)", t)
rec = {"diff": float(m[1]), "lo": float(m[2]), "hi": float(m[3])} if m else {"error": "no reward line"}
rec["passed"] = bool(m) and rec["lo"] > 0
open(sys.argv[2], "w").write(json.dumps(rec) + "\n"); print("gate:", json.dumps(rec))
EOF
if [ ! -f $R/$RUN/step200/dev-T0-stop-forced.json ]; then
    wait_card; echo "$(date '+%m-%d %H:%M') forced probe"
    $PY 07_error_recovery.py probe --split dev --temperature 0 --stop-at-json 1 --forced 07_error_recovery_data/dev_recovery_pr33.jsonl --adapter $A --out $RUN/step200
fi
[ -f $R/$RUN/step200/dev-T0-stop-forced-err0.json ] || $PY $D/rescore_err0.py $R/$RUN/step200 dev-T0-stop-forced
wait_card; echo "$(date '+%m-%d %H:%M') decision probe"
(cd $D && ONLY_STEPS=25,50,75,100,125,150,175,200 $PY decision_probe.py decisions.jsonl prov=$RUN:100)
rescored $R/$CTRL/step200 dev-T1-x2-stop
echo "== dev: step 200 vs the control's step 200, under the old reward (err0)"
$PY $D/compare_runs.py $R/$CTRL/step200 $R/$RUN/step200 dev-T1-x2-stop-err0 new200 prov200
echo "== dev: step 200 vs the control's step 200, both under err0 + provenance"
$PY $D/compare_runs.py $R/$CTRL/step200 $R/$RUN/step200 dev-T1-x2-stop-err0-prov new200 prov200 | grep -E "^   prov200 - new200"
echo "== forced probe"
$PY $D/forced_report.py new200=$R/$CTRL/step200 alam05-200=$R/$ALAM/step200 prov200=$R/$RUN/step200
echo "== primary readout"
(cd $D && $PY prov_primary.py $RUN prov)
(cd $D && $PY window_mix.py prov $RUN 0 200) 2>/dev/null
python3 -c "import json, sys; sys.exit(0 if json.load(open('$R/$RUN/gate200.json'))['passed'] else 1)" || { echo "$(date '+%m-%d %H:%M') gate failed: no held-out probes"; exit 1; }
for split in test holdout_shift; do
    probe $split 1 2
    rescored $R/$CTRL/step200 $split-T1-x2-stop
    rescored $SEED $split-T1-x2-stop
    echo "== $split: step 200 vs the control's step 200, under the old reward (err0)"
    $PY $D/compare_runs.py $R/$CTRL/step200 $R/$RUN/step200 $split-T1-x2-stop-err0 new200 prov200
    echo "== $split: step 200 vs the control's step 200, both under err0 + provenance"
    $PY $D/compare_runs.py $R/$CTRL/step200 $R/$RUN/step200 $split-T1-x2-stop-err0-prov new200 prov200 | grep -E "^   prov200 - new200"
    echo "== $split: step 200 vs the seed, both under err0 + provenance"
    $PY $D/compare_runs.py $SEED $R/$RUN/step200 $split-T1-x2-stop-err0-prov seed prov200 | grep -E "^   prov200 - seed"
done
echo "$(date '+%m-%d %H:%M') provenance run done (stopped at 200)"
