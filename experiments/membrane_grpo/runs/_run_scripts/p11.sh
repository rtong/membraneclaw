set -euo pipefail
cd ~/grpo-seed/experiments/membrane_grpo
V=~/MembraneClaw/experiments/membrane_grpo/.venv/bin/python
BASE="--model Qwen/Qwen3-1.7B --steps 200 --prompts-per-step 4 --group-size 8 --max-new-tokens 640 --temperature 1.0 --lr 1e-05 --weight-decay 0.0 --clip-eps 0.2 --beta 0.0 --inner-epochs 1 --micro-batch 1 --normalize token --lora-r 16 --weights MAIN --seed 0 --dtype bfloat16 --split train --prompt-version v2-oracle-num --eval-every 25 --eval-cases 200 --eval-batch 32 --eval-split dev --eval-max-tokens 640"
EVAL="--backend hf --model Qwen/Qwen3-1.7B --split dev --mode greedy --max-tokens 640 --batch-size 16 --prompt-version v2-oracle-num"

echo "== $(date -Is) seed-shuffled"
$V seed_sft.py --shuffle-labels 1 --out runs/seed-shuffled
$V eval.py $EVAL --adapter runs/seed-shuffled/adapter --run-name paired-seed-shuffled
echo P11_SEED_DONE

echo "== $(date -Is) q3-oracle-main-s0 (fresh adapter)"
$V grpo_scratch.py $BASE --out runs/q3-oracle-main-s0
$V eval.py $EVAL --adapter runs/q3-oracle-main-s0/adapter --run-name paired-q3-oracle-main-s0
echo P11_FRESH_DONE

echo "== $(date -Is) q3-oracle-main-shufseed-s0 (from the shuffled seed)"
$V grpo_scratch.py $BASE --init-adapter runs/seed-shuffled/adapter --out runs/q3-oracle-main-shufseed-s0
$V eval.py $EVAL --adapter runs/q3-oracle-main-shufseed-s0/adapter --run-name paired-q3-oracle-main-shufseed-s0
echo "== $(date -Is) P11_ALL_DONE"
