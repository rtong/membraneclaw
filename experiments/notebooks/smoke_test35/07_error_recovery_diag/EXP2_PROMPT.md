# Notebook 07, experiment 2: four times the rollouts, Monte-Carlo credit

You are helping with an RL experiment series. It trains Qwen3.5-0.8B with PPO to answer water-treatment questions with simulator tools, and you run one experiment on your own GPU machine.

- **Repo:** `MembraneClaw`, branch `experiment/smoke-test35-07-error-recovery` (PR #37).
- **Code:** the commit after `027ac5f` or later.
- **Directory:** work in `experiments/notebooks/smoke_test35/` (`$ST` below).
- **Who to report to:** the person who sent you this.

## What it tests

The trainer credits each turn with GAE over turns, λ 0.95: that is, nearly Monte-Carlo. The first turn decides whether to convert the question's units before simulating, and its credit is about the whole episode's return minus the critic's estimate.

- **The noise:** on the same question with the same first call, reward has sd 0.22–0.33, while the effect of the decision is −0.19 to +0.30. So each step's push on the decision has signal/noise about 0.2, and the decision random-walks.
- **This experiment** keeps the Monte-Carlo credit and averages the noise down: 32 distinct questions per step instead of 8. Noise per step falls as 1/√N.
- **Why the lr halves too:** Adam normalises the step, so with noisy gradients a step moves about lr whatever the batch size. Four times the rollouts at lr/√4 keeps the useful drift per step and halves the random swing.
- **Delta call credit:** `--call-credit delta` (per-turn increments of the calls component) is included, as in the parallel experiment 1. Under Monte-Carlo credit it changes per-turn rewards in only about 3% of episodes.

**Compared with the control** (`ppo-n300-maskend-turn-lr2e6-err0-s0`, trained on another machine), only these differ:

| flag | control | here |
|---|---|---|
| `--prompts-per-step` | 8 | 32 |
| `--lr` | 2e-6 | 1e-6 |
| `--call-credit` | call | delta |

Plus memory chunking, which gives the same updates. It runs from the same seed, to step 200.

## Rules

1. **Fixed task.** The task prompt is fixed. Never reword the questions or the prompt, and never precompute anything into them.
2. **Read-only areas.**
   - Benchmark D1–D7 is eval-only.
   - `experiments/auto-evaluate/`, `experiments/membrane_grpo/` and `toy_mdp` are read-only. You only use a python environment like membrane_grpo's.
3. **No code changes.** Don't change `07_error_recovery.py`, the flags, or the scripts. If something doesn't work, stop and report.
4. **Gates.** The gate is pre-registered: if it fails, stop and report. Don't go past step 200.
5. **Git.** Don't commit or push to this branch. Return results as described at the end.
6. **Processes.** Never edit a bash script while it is running. Stop processes by PID.

## Setup

1. **Code.** Check out the branch at the commit after `027ac5f` or later: `git fetch && git checkout experiment/smoke-test35-07-error-recovery && git pull`.
2. **Seed weights.** `.safetensors` files are not in git. You will be sent `adapter_model.safetensors`, about 22 MB. Put it at `$ST/07_error_recovery_runs/recovery-seed-n300-maskend/epoch1/adapter_model.safetensors`; that folder's `adapter_config.json` is already in git.
3. **Training environment.** Python 3.13 with:
   - torch 2.13 (CUDA), transformers 5.15.0, peft 0.20.0;
   - flash-linear-attention (`fla`) 0.5.2, which Qwen3.5's linear-attention layers need on the GPU.

   `experiments/membrane_grpo/requirements.txt` lists the rest, but not fla. The script expects `experiments/membrane_grpo/.venv/bin/python`; otherwise set `PY=/path/to/python`.
4. **Simulator environment** at the repo root: `.venv-watertap`, with watertap 1.7.0, idaes-pse 2.12.0 and pyomo 6.10.1, plus the IDAES solvers (`idaes get-extensions`, for ipopt). The trainer starts its tool workers as `<repo>/.venv-watertap/bin/python 07_error_recovery.py worker`.
5. **Model.** `Qwen/Qwen3.5-0.8B` from Hugging Face; it downloads on first use.
6. **CPU checks.** All must pass:
   ```
   cd $ST
   CUDA_VISIBLE_DEVICES= $PY 07_error_recovery_diag/templates/test_first_turn_lam.py
   CUDA_VISIBLE_DEVICES= $PY 07_error_recovery_diag/templates/test_loss_norm.py
   ```

## GPU settings

These are environment variables for the script:

- **`MAX_VRAM`:** the per-process cap in MiB. The default 13000 is for a shared 16 GB card. On a dedicated card, use about the card's size minus 1–2 GB.
- **`ROLLOUT_CHUNK`** (default 8: the 32 prompts are generated 8 at a time) and **`CRITIC_CHUNK`** (default 250000 tokens). These keep 32 episodes per step within 16 GB. On a card of 40 GB or more you may set `ROLLOUT_CHUNK=16` or `32`, and `CRITIC_CHUNK=0`.
  - Chunking doesn't change the updates. The critic's chunked fit is the same full-window update; a CPU test checks this.
  - Chunked generation draws different random numbers. That is fine: it is a fresh run.
- **`WAIT_MIB`:** if the card is shared, the script waits before each GPU job until less than this many MiB are used. Leave it unset on a dedicated card.
- **`NVSMI`:** the nvidia-smi path, if it is not on PATH.

## Run

```
cd $ST
MAX_VRAM=... ROLLOUT_CHUNK=... CRITIC_CHUNK=... nohup setsid bash 07_error_recovery_diag/exp2_rollouts.sh > 07_error_recovery_runs/exp2_rollouts.log 2>&1 < /dev/null &
```

The script does the following, in order:
1. It checks its inputs and the environment.
2. It trains `ppo-n300-maskend-turn-lr1e6-err0-delta-n32-s0` to step 200, with up to 8 attempts that resume from `last/` if the watchdog ends a run. The training log is `07_error_recovery_runs/ppo-<run>.log`.
3. At step 200:
   - dev T=1 x2 with the err0 re-score;
   - the forced probe (recovery from a slipped call) with the err0 re-score;
   - the decision probe on checkpoints 25–200 (key `n32`).
4. It prints the comparisons and stops.

**Watch these:**
- **Environment check:** the greedy eval at step 0, the first line in the log after about 30 minutes, should be close to the control's 0.5954. A small difference from a different GPU is fine. If it is off by more than 0.05, stop and report: simulators, model or seed are probably wrong.
- **Speed:** after about step 30, compute seconds per step from `metrics.jsonl` (`step_seconds`). Report the projected time to step 200. The control took about 3 minutes per step with 8 episodes on a 16 GB card.
- **Memory:** if a run fails with an OOM, report it, with the log.

## Pre-registered gate and readouts

**Gate.** At step 200, dev T=1 x2 err0 paired against the seed: the 95% interval of the reward difference must lie above 0 (the "seed n32" comparison block). For reference, the control's step 200 was +0.264 (+0.222 to +0.306).

**Primary readout: does the decision stop random-walking?** The script prints a route-swing table.
- It shows mean |change of route log-odds| between consecutive checkpoints, and sign changes, for train L1 2+ units / train L2 / dev L1 2+ units / dev L0, over 25–200.
- The control ("new") has 2.45 / 1.65 / 2.59 / 1.26.
- **Success:** at most half of the control's on train and dev L1 2+, and dev L0 p(convert) no higher than the control's.

**Secondary readouts:**
- Step 200 against the control's step 200 (dev T=1 x2 err0, paired). Note that the control's step 200 is its best checkpoint, a peak.
- The forced probe against the control's step 200.
- `window_mix`: per window, the convert-first share by group.

## Return

Send back the following.

**Files:**
1. **The run directory** `$ST/07_error_recovery_runs/ppo-n300-maskend-turn-lr1e6-err0-delta-n32-s0/`:
   - everything except `*.safetensors`;
   - plus `adapter-step200/adapter_model.safetensors`.
2. **The decision-probe lines:** the `n32` lines appended to `07_error_recovery_diag/decisions.jsonl`, for example `grep '"run": "n32"'`.
3. **The logs:** `07_error_recovery_runs/exp2_rollouts.log` and the training log.

**A short summary:**
- your GPU and settings;
- time per step;
- the step-0 environment check;
- the gate result;
- the route-swing table;
- anything unusual.
