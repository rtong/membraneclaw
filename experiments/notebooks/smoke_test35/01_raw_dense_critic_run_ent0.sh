#!/bin/sh
# 01_raw_dense_critic, run ent0-s0: base-s0's configuration with the entropy bonus off.
# The critic's layer, learning rate and starting value are the ones the notebook's probe chose:
# hidden_states[23], lr = 0.02 / ||h||_1 = 9.55e-5, bias = mean return over scored tokens.
cd "$(dirname "$0")"
mkdir -p 01_raw_dense_critic_runs
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 exec ../../membrane_grpo/.venv/bin/python 01_raw_dense_critic.py \
  --run ent0-s0 --entropy-coef 0 \
  --value-layer 23 --value-lr 9.549180223364499e-05 --value-init-bias 0.1583 \
  --max-vram-mib 13000 \
  > 01_raw_dense_critic_runs/ent0-s0.log 2>&1
