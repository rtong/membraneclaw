# Notebook 07, experiment 1: the first turn credited by the critic (λ_0 = 0 and 0.5)

You run experiment 1 on the GPU machine where the code lives.

- **Repo:** `/home/bayan/MembraneClaw`, branch `experiment/smoke-test35-07-error-recovery` (open PR #37).
- **Code:** the trainer flags are committed in `027ac5f`; the driver script and these prompts are in the commit after it.
- **Directory:** work in `ST=/home/bayan/MembraneClaw/experiments/notebooks/smoke_test35`.
- **Python:** `../../membrane_grpo/.venv/bin/python`, run from `$ST`.
- **Language:** report to the user in Chinese.

## What it tests

The PPO here credits each turn with GAE over turns, λ 0.95, so the first turn's advantage is about G − V(s_0). The first turn holds the convert-or-simulate decision, and it gets blamed or rewarded for everything that happens later.

- **The data:** on the same question with the same first call, reward has sd 0.22–0.33, while the route effect is −0.19 (L0) to +0.30 (L1 2+ units).
- **The consequence:** the route's per-step push has signal/noise about 0.2, and its log-odds swing 2–5 per 25 steps for the whole run.

Experiment 1 makes two changes; all else is identical to the control `ppo-n300-maskend-turn-lr2e6-err0-s0`:

1. **`--call-credit delta`:** turn j earns C[j] − C[j−1] of the weighted calls component, so a turn's reward depends only on that turn and the ones before it.
2. **`--first-turn-lam`:**
   - At 0, the first turn's advantage is r_0 + V(s_1) − V(s_0). The decision is judged by the critic's value of the state it led to, the first tool response; that value is learnt by regression over many episodes and doesn't depend on this episode's later luck.
   - At 0.5, half of the second turn's advantage is added.
   - Other turns and the critic's targets are unchanged.

There are two runs: λ_0 = 0 (`td0`) and λ_0 = 0.5 (`td05`). Each starts fresh from the seed.

**The risk is the critic.** Its explained variance is about 0.35. And for L0, the harm of an unneeded conversion is visible in the first tool response only 27% of the time.

## Rules

1. **Fixed task.** The task prompt is fixed. Never reword the questions or the prompt, and never precompute anything into them.
2. **Read-only areas.**
   - Benchmark D1–D7 is eval-only.
   - `experiments/auto-evaluate/`, `experiments/membrane_grpo/` and `toy_mdp` are read-only. You only use membrane_grpo's `.venv` python.
3. **GPU.** The card has 16 GB and is shared with the display.
   - Leave 2 GB free. PPO keeps its default `--max-vram-mib 13000`.
   - Every GPU job waits until the card has less than 2500 MiB used; the driver does this.
   - Run one GPU job at a time.
4. **Processes.**
   - Never edit a bash script while it is running, including `exp1_td.sh` once launched.
   - Never edit `07_error_recovery.py` while the driver may still restart a run from it.
   - Stop processes by PID. Never use `pgrep -f` with the script's own name.
5. **Gates.** Every gate is pre-registered. When one fails, stop and report. Don't improvise another run, and don't change parameters.
6. **Git.** Don't commit or push unless the user asks. Never commit files you didn't create: OPERATIONS.md, deploy/, mem-deploy/ and smoke_test/ belong to someone else.
7. **Notebooks.** Don't edit the notebook (.ipynb). Notebooks are written later, from finished runs only.
8. **Failures.** If a run fails (an OOM, or a crash after 8 attempts), report it with the log. Don't raise the memory cap.

## Already running: do not touch

The `--loss-norm batch` branch: PID 110946, log `07_error_recovery_runs/ppo-lnbatch.log`.

- **What's left:** it ends around 03:00 on 2026-10-20. After training it runs its probes, including dev T=1 x2 on the control's continuation at step 300: `ppo-n300-maskend-turn-lr2e6-err0-s0-to400/step300`. You need that file.
- **Ordering:** the driver below waits for this PID before starting.
- **Report:** put the final comparison lines from this log in your report.

## Your steps

1. **Re-run the CPU tests.** All must pass.
   ```
   cd $ST
   CUDA_VISIBLE_DEVICES= ../../membrane_grpo/.venv/bin/python 07_error_recovery_diag/templates/test_first_turn_lam.py
   CUDA_VISIBLE_DEVICES= ../../membrane_grpo/.venv/bin/python 07_error_recovery_diag/templates/test_loss_norm.py
   CUDA_VISIBLE_DEVICES= ../../membrane_grpo/.venv/bin/python 07_error_recovery_diag/templates/test_call_credit.py
   ```
2. **Read `07_error_recovery_diag/exp1_td.sh`** and `07_error_recovery_diag/td_precheck.py` before launching.
3. **Launch the driver:**
   ```
   cd $ST
   nohup setsid bash 07_error_recovery_diag/exp1_td.sh > 07_error_recovery_runs/exp1_td.log 2>&1 < /dev/null &
   ```
4. **Know what the driver does,** in order:
   1. It waits for lnbatch.
   2. **Gate 0, the pre-check** (`td_precheck.py`, about 20–30 minutes on the GPU).
      - It runs on the control's step-200 state and its training samples of steps 200–249.
      - It computes each episode's first-turn advantage three ways: MC (as trained), TD0 and TD0.5.
      - It writes `07_error_recovery_diag/td_precheck.json`.
      - Here it is informational: both λ_0 run either way.
   3. `td0` to step 200, then `td05` to step 200. Each takes about 13–14 h with evals.
   4. At each step 200:
      - dev T=1 x2 with the err0 re-score, and the forced probe with the err0 re-score;
      - the decision probe on checkpoints 25–200 (keys `td0`, `td05`);
      - **the step-200 gate.**
   5. Each run that passed continues in place to step 300. Past 200 the collapse rule applies. Then probes at the last step, and the decision probe on 225–300.
   6. Comparisons, printed at the end of the log.
5. **Check the first steps of `td0`** after its warm-up (steps 25 onward, about 1.5 h after it starts):
   - The log lines look normal (reward, calls, KL), with no OOM.
   - `metrics.jsonl` has `adv_turn0_mean` and `adv_turn0_std`.
   - If something is wrong, stop the driver and then the python by PID, and report.
6. **Report** after gate 0, after each step-200 gate, and at the end.

## Pre-registered criteria

**Gate 0** (`td_precheck.json`), for reporting:
- A variant passes if, in L0 and L1 2+ units, its convert-first minus simulate-first contrast has the reference's sign and at least half its size (references: L0 −0.189, L1 2+ units +0.299), and its pooled within-cell sd is below MC's.
- **How to read it.**
  - Under TD0, the L0 contrast is partly mechanical. A right first simulate call earns its call credit at once, in r_0, while convert-first earns 0 there. So the L0 contrast can come out right even if the critic is useless.
  - The L1 2+ contrast is the real test of the critic: there r_0 is 0 on both routes, and only V(s_1) can tell them apart.
  - Report the table, the critic's explained variance, and V(s_1) after an erring vs a clean first response.

**Step-200 gate,** for each run: dev T=1 x2 err0 paired against the seed (`recovery-seed-n300-maskend/epoch1`). The 95% interval of the reward difference must lie above 0, or the run stops at 200.

**Collapse rule, 200–300:** at an evaluation N > 200, stop if greedy decision < 0.65, or if training truncation over steps N−25 to N−1 is ≥ 0.30.

**Primary readout: does the route stop random-walking?**
- Measure mean |change of route log-odds| between consecutive checkpoints, and the number of sign changes, for train L1 2+ units / train L2 / dev L1 2+ units / dev L0.
- The control ("new" in `decisions.jsonl`) has, for 25–200: 2.45 / 1.65 / 2.59 / 1.26; for 200–300: 1.93 / 1.08 / 2.10 / 1.14.
- **Success:** at most half of the control's on train and dev L1 2+, and dev L0 p(convert) no higher than the control's. The driver prints this table.

**Secondary readouts:**
- **Step 200** against the control's step 200. Note that the control's step 200 is its best checkpoint, a peak.
- **Step 300** against its own step 200, and against the control's step 300.
- **td05 against td0** at step 200, paired.
- **Forced probe** (recovery) against the control's step 200.
- `window_mix`: per window, the convert-first share by group.
- `adv_turn0_std` over training.

## Timing

About 2 days in total:
- about 20–30 minutes for gate 0;
- 2 × ~16 h to step 200, including probes;
- about 9 h for each run that continues to 300.

## Report (in Chinese)

At each point, include:
- the CPU test output;
- the gate 0 table;
- each gate, passed or failed, with its numbers;
- the readout tables against the control;
- lnbatch's final results;
- the actual time each stage took;
- anything that looked wrong.

Don't commit, and don't start anything else.
