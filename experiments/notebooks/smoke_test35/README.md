# smoke_test35 — actor-critic PPO on Qwen3.5-0.8B

The v2 prompt, unchanged; the reward paid per field at the token that writes it; a learned value
head read off the policy's own hidden state. Task, grader and evaluation are imported from
[`../../membrane_grpo`](../../membrane_grpo) (`task/`, `reward.py`, `eval.py`), never copied.

Each notebook owns one `.py` file with its prefix, and everything a notebook's runs produce sits
beside it under the same prefix: launch scripts (`NN_*_run*.sh`), run directories
(`NN_*_runs/<run>/` with `config.json`, `metrics.jsonl`, `eval.jsonl`, `best.json`, `ending.json`,
sampled episodes), measurements (`NN_*_probe.json`, `NN_*_checks.json`) and figures (`NN_*.png`).
No code is shared between notebooks. Adapter weights, value heads, resume checkpoints and the probes'
hidden-state caches are not in git.

| notebook | question | answer |
| --- | --- | --- |
| [`01_raw_dense_critic.ipynb`](01_raw_dense_critic.ipynb) | Does the per-field-credit critic work on the raw model, with no tool and no seed? | No. Held out by prompt, no hidden layer predicts the return beyond position (every layer is -0.008 to -0.014 worse than position alone); a training run stopped on its critic gate at step 75 with the diagnosis collapsed to one label. The raw model gets the numbers right 1% of the time, and every 1.7B critic that worked had the numbers right first. |
| [`02_tool_seed_ppo.ipynb`](02_tool_seed_ppo.ipynb) | With a calculator and a seed that learns only to call it, does it work -- and where does PPO converge? | Yes. On the seed, hidden states add +0.011 to +0.015 over position; in the converged run (`kl001-s0`, 700 steps) the critic is ahead of the clock by +0.02 to +0.03 from step 100. Held out on test: exact match 0.005 -> 0.285, cause 0.290 -> 0.825 (McNemar p = 3e-17 / 1e-26); holdout_shift exact match 0.020 -> 0.400. The remaining cause error is one table row: `(down, flat, up, lead)` read as `scaling` on all 29 test cases. |
| [`03_label_levers.ipynb`](03_label_levers.ipynb) | What kept labels out of PPO's reach, and what brings them back? | Four settings, one per run. A **stage seed** (02's seed wrote a cause into `stage` on 164/200 answers, and PPO could lock in), the answer's scaffolding frozen and **Adam eps 1e-6** (the format stopped drifting), a **label seed** of 4 off-distribution records per cause (every cause sampled; `scaling` had never been said), and a correct **severe action paid 3x** (0/37 -> 37/37). Test exact match 0.285 -> 0.380 -> 0.805 -> **0.985**, cause 0.995; holdout_shift 0.940. What is left are flag thresholds on 3 of 200 test cases. |

## Running

```sh
cd experiments/notebooks/smoke_test35
../../membrane_grpo/.venv/bin/python 02_tool_seed_ppo.py seed --out tool-seed-anchor --anchor-open
./02_tool_seed_ppo_run_kl001.sh            # the converged run; resumes itself after a stall
./02_tool_seed_ppo_eval_heldout.sh          # test / holdout_shift / train, and the seed-only control
../../membrane_grpo/.venv/bin/python 03_label_levers.py seed --out tool-seed-stage --slots stage
./03_label_levers_labelseed_sweep.sh        # label seeds of 2 / 4 / 8 records per cause, probed at T=1
./03_label_levers_run_pc4_severe3.sh        # the final run (also _run_eps.sh, _run_pc4.sh)
./03_label_levers_eval_heldout_pc4_severe3.sh
```

The notebooks read finished runs and cached measurements; they retrain nothing.
