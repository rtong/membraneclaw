set -euo pipefail
LOG=~/grpo-seed-logs/p11.log
# Wait for the first pipeline. If it died without finishing, stop rather than
# run the correct seed against a missing comparison.
until grep -q P11_ALL_DONE "$LOG"; do
  if ! pgrep -f "grpo-seed-logs/p11.sh" >/dev/null; then
    sleep 5; grep -q P11_ALL_DONE "$LOG" && break
    echo "P11B_ABORTED: p11.sh exited without P11_ALL_DONE"; exit 1
  fi
  sleep 60
done

cd ~/grpo-seed && git pull -q --ff-only && git log --oneline -1
cd experiments/membrane_grpo
V=~/MembraneClaw/experiments/membrane_grpo/.venv/bin/python
BASE="--model Qwen/Qwen3-1.7B --steps 200 --prompts-per-step 4 --group-size 8 --max-new-tokens 640 --temperature 1.0 --lr 1e-05 --weight-decay 0.0 --clip-eps 0.2 --beta 0.0 --inner-epochs 1 --micro-batch 1 --normalize token --lora-r 16 --weights MAIN --seed 0 --dtype bfloat16 --split train --prompt-version v2-oracle-num --eval-every 25 --eval-cases 200 --eval-batch 32 --eval-split dev --eval-max-tokens 640"
EVAL="--backend hf --model Qwen/Qwen3-1.7B --split dev --max-tokens 640 --prompt-version v2-oracle-num"

echo "== $(date -Is) seed-true"
$V seed_sft.py --shuffle-labels 0 --out runs/seed-true
$V eval.py $EVAL --mode greedy --batch-size 16 --adapter runs/seed-true/adapter --run-name paired-seed-true
echo P11B_SEED_DONE

echo "== $(date -Is) q3-oracle-main-trueseed-s0 (from the correct seed)"
$V grpo_scratch.py $BASE --init-adapter runs/seed-true/adapter --out runs/q3-oracle-main-trueseed-s0
$V eval.py $EVAL --mode greedy --batch-size 16 --adapter runs/q3-oracle-main-trueseed-s0/adapter --run-name paired-q3-oracle-main-trueseed-s0
echo P11B_GRPO_DONE

echo "== $(date -Is) temperature-1.0 label probes"
$V eval.py $EVAL --mode sample -k 8 --batch-size 32 --run-name probe-t1-frozen
$V eval.py $EVAL --mode sample -k 8 --batch-size 32 --adapter runs/seed-shuffled/adapter --run-name probe-t1-seed-shuffled
$V eval.py $EVAL --mode sample -k 8 --batch-size 32 --adapter runs/seed-true/adapter --run-name probe-t1-seed-true
echo "== $(date -Is) P11B_ALL_DONE"
