#!/usr/bin/env bash
# The candidate critics (08_prover_critic.py CRITICS, and 07's own critic's features as "policy") fitted and compared on
# 07's sampled PPO episodes: every step of ppo-n200-r100-turn-s0 (train questions; replayed through the simulators so
# that each episode's text and score are the rollout's) and the seed's and the control's 120 dev questions sampled at
# T=1 (held out by question). Each candidate is read once over every episode (its features cached under
# 08_prover_critic_runs/critic-fit/, so a second call only refits); the heads' penalties come from cross-validation by
# question, and the candidates are compared the way PPO will use them -- fitted on the 25 steps before a step, tested
# on that step -- then the paper's prover criterion picks K. Writes 08_prover_critic_critic_fit.json.
#   ./08_prover_critic_critic_fit.sh
#   CANDIDATES=policy,qwen3-1.7b ./...      fewer candidates (the 4B reads ~2.5x slower than the 1.7B)
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
PY=../../membrane_grpo/.venv/bin/python
CANDIDATES=${CANDIDATES:-policy,qwen3-1.7b,qwen3.5-2b,qwen3-4b}
mkdir -p 08_prover_critic_runs/critic-fit
$PY 08_prover_critic.py critic-fit --candidates "$CANDIDATES" 2>&1 | tee -a 08_prover_critic_runs/critic-fit/critic-fit.log
