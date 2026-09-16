set -euo pipefail
cd ~/grpo-seed/experiments/membrane_grpo
V=~/MembraneClaw/experiments/membrane_grpo/.venv/bin/python
grep -q sampled_cause_hist eval.py || { echo "PROBES_ABORTED: eval.py lacks sampled_cause_hist"; exit 1; }
EVAL="--backend hf --model Qwen/Qwen3-1.7B --split dev --max-tokens 640 --prompt-version v2-oracle-num --mode sample -k 8 --batch-size 32"
for name in frozen seed-shuffled seed-true; do
  rm -rf runs/probe-t1-$name
  if [ "$name" = frozen ]; then A=""; else A="--adapter runs/$name/adapter"; fi
  echo "== $(date -Is) probe $name"
  $V eval.py $EVAL $A --run-name probe-t1-$name
done
echo "== $(date -Is) PROBES_DONE"
