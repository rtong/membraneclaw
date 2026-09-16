set -euo pipefail
cd ~/grpo-seed/experiments/membrane_grpo
V=~/MembraneClaw/experiments/membrane_grpo/.venv/bin/python
EVAL="--backend hf --model Qwen/Qwen3-1.7B --max-tokens 640 --batch-size 16 --prompt-version v3 --mode greedy"
SEED=runs/seed-nb11-v3/adapter

echo "== $(date -Is) baseline: the seeded policy itself, under v3"
$V eval.py $EVAL --split dev --adapter $SEED --run-name paired-seed-nb11-v3
echo P12_BASELINE_DONE

echo "== $(date -Is) GRPO 400 steps from it"
$V grpo_scratch.py --model Qwen/Qwen3-1.7B --steps 400 --prompts-per-step 4 --group-size 8 \
  --max-new-tokens 640 --temperature 1.0 --lr 1e-05 --weight-decay 0.0 --clip-eps 0.2 --beta 0.0 \
  --inner-epochs 1 --micro-batch 1 --normalize token --lora-r 16 --weights MAIN --seed 0 \
  --dtype bfloat16 --split train --prompt-version v3 --eval-every 25 --eval-cases 200 \
  --eval-batch 32 --eval-split dev --eval-max-tokens 640 \
  --init-adapter $SEED --out runs/q3-v3-seeded-s0
echo P12_GRPO_DONE

for split in dev test holdout_shift; do
  echo "== $(date -Is) final adapter on $split"
  $V eval.py $EVAL --split $split --adapter runs/q3-v3-seeded-s0/adapter --run-name paired-q3-v3-seeded-s0-$split
done
echo "== $(date -Is) P12_ALL_DONE"
