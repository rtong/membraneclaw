# smoke_test35 — actor-critic PPO on Qwen3.5-0.8B

The reward paid per field at the token that writes it; a learned value head read off the policy's
own hidden state. In `01`-`03` the task is the v2 prompt, unchanged, and task, grader and evaluation
are imported from [`../../membrane_grpo`](../../membrane_grpo) (`task/`, `reward.py`, `eval.py`),
never copied. From `04` the task is natural-language questions answered with the deployed WaterTAP
tools; its generator, gold answers and grader live in `04_natural_watertap.py`, and `05` carries a
copy of them with one tool and one reasoning line added. The MembraneClaw
benchmark (`auto-evaluate/benchmarks`) is evaluation-only: nothing is generated, seeded or tuned from it.

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
| [`04_natural_watertap.ipynb`](04_natural_watertap.ipynb) | The real task: natural-language questions answered with `simulate_ro` and `simulate_swro_system`, graded by code. Can the series' recipe -- seed, then PPO -- learn it on 0.8B? | 1,800 generated questions in six families, every gold answer a real simulation (12/12 identical to the deployed server), 0 prompts sharing an 8-gram with the benchmark. The raw model scores 0.003 (it calls a tool that does not exist). A blind seed on the scaffolding plus a label seed of 26 records reaches test reward 0.833 (decisions 0.812); PPO converges at 500 steps and adds +0.011 reward, +0.026 decisions -- not significant (McNemar p = 0.41). Left: unit conversions done in the head (level-2 questions cover the proving points ~half the time) and scan judgement (pressure scans 0.575); holdout_shift 0.66. |
| [`05_units_and_scans.ipynb`](05_units_and_scans.ipynb) | 04's two limits, measured item by item from its answers: limits copied unconverted (91 of 102), arguments converted in the head 1-2.5% off (33 of 63 level-2 questions), and scans asked for the highest setting answered with the first (5 of 13). Do a conversion tool and an aggregation line fix them? | Yes, in the seed. A **`convert_units` tool** (served here, not on the deployed server yet; 04's factor for A in LMH/bar was 1000x off, corrected, 1,680 of 1,800 questions unchanged) and **one line** naming the passing candidates and the objective's pick. The label seed alone: test reward 0.833 -> **0.916**, decisions 0.812 -> **0.908**, whole answer right 0.696 -> 0.817, level-2 proving simulations 0.51 -> 0.95. PPO from it added nothing (test +0.004, McNemar p = 1.00; whole answer 0.817 -> 0.846) and its tool gate stopped it at step 244 as the calls began to go; best checkpoint test 0.920 / holdout_shift 0.739 (04: 0.844 / 0.657). Left: misread simulated values and comparisons, and holdout_shift's flipped limit directions. An **untrained Qwen3.5-4B** on dev (through mlx-lm on Apple silicon) is not the missing piece: reward 0.292 under the answer contract (it answers in prose; the seed: 0.928), its conclusion read from that prose right on 0.583 of the questions (the seed: 0.933), a third of its episodes lost to calls in the wrong units, and on the 134 questions it gets through it is right on 126 against the seed's 131. Untrained 0.8B: 0.003. |
| [`06_direction_and_compare.ipynb`](06_direction_and_compare.ipynb) | 05's seed wrote the comparison operator the wrong way round on 98 of 655 holdout_shift items -- all of them under the two limit sentences train never uses -- and on test compared or copied a right number wrongly in 26 of its 37 wrong verdicts. Is the sentence the cause, and do more sentences in train and a comparison tool fix it? And seeded the method's way, with the smallest alive SFT dose, does PPO activate the seed? | **The sentence is the cause**: 05's seed with holdout_shift's limits restated by a train sentence writes 0 of the 98 operators wrong; with test's limits restated by the unseen sentence, 231 wrong. **One more sentence per limit in train, and a `check_limits` tool** whose response is the comparison lines (both tools served here, not on the deployed server yet). The method's seed, 600 reference episodes: test 0.816, whole answer right 0.775, holdout_shift 0.802 / 0.717. The next dose, 1200 episodes, one epoch of SFT before any PPO: test **0.925** / **0.879**, holdout_shift **0.880** / **0.850** (05's PPO: 0.920 / 0.846 and 0.739 / 0.517). **Judged greedily, PPO activated neither.** From the 600 seed: at lambda 0.95 per token the run collapsed by learning to skip the unit conversion (the choice's bad news is a simulate error a median 431 tokens later, weighted 0.95^431, and the critic at the choice is blind to the question's unit level); at lambda 1 it wrote convert calls needed or not; with the turn as the unit of credit every sampled measure improved for 50 actor steps (reward 0.722 -> 0.804, whole answer right 0.63 -> 0.70) and then the policy ran away, a turn's advantage being 5x a token's at the same actor step; at a fifth of the step it did not run away, and sampled at T=1 on all 240 dev questions its last checkpoint beats the seed, 0.830 against 0.702 (paired difference +0.128, 95% interval +0.095 to +0.161), while its greedy evaluations stayed at 0.809-0.848 against 0.879 at the start: PPO moved what the policy samples, not its greedy path. From the 1200 seed at lambda 0.95: evaluations 0.947 -> 0.788, a unit slipped in one simulate argument and the erroring call retried rather than fixed, up to 20 in a row. Every best checkpoint is step 0. Left: recovery from an erroring call, which the reference episodes never show. |
| [`07_recover_from_errors.ipynb`](07_recover_from_errors.ipynb) | 06's runs were capped by one error: a simulate call fed the question's number in the question's unit, the simulator erring, and the policy retrying rather than fixing -- 06's reference episodes never err, so no seed had seen a fix. Do recovery episodes in a small seed (no dose sweep) teach the fix, and does PPO then activate it? (Run on Apple silicon: MPS.) | **The seed carries the fix**: 300 episodes (200 of 06's references + 100 recovery episodes: one input left unconverted, the simulator's real errors, then a one-line fix naming the input and its unit, the conversion and the rest); started from the slipped turns of 144 dev questions it recovers in **0.812** of them against the control's 0.285 (06's 300 references; exact repeats per episode 0.67 against 4.97). **PPO activated it**: from that seed with 06's last configuration, the series' first run whose greedy evaluations rose above step 0 and stayed there -- **0.586 -> 0.658 -> 0.648 -> 0.673** at steps 0/40/80/120 (60 dev questions, best = last), decision right 0.550 -> 0.683; in training, recovery after an erroring call 0.372 -> 0.539 while the share that erred stayed (PPO grew the fix, not the avoidance). On MPS, Qwen3.5's triangular solve (0.229 of 0.234 s per layer) replaced by a recursive-doubling inverse: forward 5.8 -> 1.35 s, forward+backward 25.6 -> 6.0 s. The run's held-out probes are queued behind 08. |
| [`07_error_recovery.ipynb`](07_error_recovery.ipynb) | The same error as the other 07: 06's runs were capped by a simulate call fed a number in the wrong unit and retried rather than fixed. Given recovery episodes at the smallest alive dose, does PPO activate the fix -- and what stops it after? (Run on CUDA. The two 07s' probes differ -- the adapter folded into bf16 or not, 120 dev questions once or 240 twice, 06's reward or err0 -- so their numbers are not comparable as they stand; 08 compares the two seed recipes on the same questions.) | **PPO activated it.** Recovery episodes (one slip -- an unconverted input or a x10 factor -- the tool's real error, the fix; the slipped turn shown, not trained) as every fourth episode of the smallest alive dose (150 not alive, 300 alive). Step 200 against its seed, every dev question sampled twice at T=1, paired: **+0.253** (+0.219 to +0.288) from the first seed, **+0.264** (+0.222 to +0.306) from the corrected one (the omission's `<\|im_end\|>` unlabelled, and no call credit for a simulate call the tool rejects); test 0.416 -> **0.712** and 0.449 -> **0.714**, holdout_shift 0.363 -> **0.607** and 0.349 -> **0.608**, gains at every unit level. Not separated: decisions among the erring episodes went 0.17 -> 0.31, about what 06's PPO did without recovery episodes. **Step 200 is the peak.** Whether the first call converts is one token shared by every question; its log-odds random-walk from step 50, all question groups together, and continued to 400 the run converts first less on L1 and falls (-0.045). Five changes to the credit (a batch token mean, first-turn TD at lambda 0 and 0.5, actor lambda 0.5, provenance) each passed its gate against the seed, and none beat the control at the same step or stopped the walk: from step 100 the reward barely tells the L1 routes apart, and provenance changed what the simulate branch does (the stated number, rejected, then converted), not the swing. Left: the update's noise at 8 questions a step (`09_questions_per_step`). |

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

# 04 (the simulator commands need ../../../.venv-watertap; check runs anywhere)
W=../../../.venv-watertap/bin/python
$W 04_natural_watertap.py dump-tools        # the served tool declarations
$W 04_natural_watertap.py generate          # 1,800 questions, gold from the simulators (~7 min, 8 workers)
$W 04_natural_watertap.py elasticity        # the gold margins
python3 04_natural_watertap.py check        # reward sanity, balance, decontamination
$W 04_natural_watertap.py xcheck --split test   # 12 gold points against the deployed MCP
../../membrane_grpo/.venv/bin/python 04_natural_watertap.py reference   # blind reference episodes on train
./04_natural_watertap_seed_verdict2.sh      # the scaffold seed (and its dev evaluation)
./04_natural_watertap_labelseed_sweep.sh    # label seeds of 2 / 4 / 8 records per answer class, at T=1
./04_natural_watertap_ppo_run.sh            # PPO from the k=2 label seed; resumes itself after a stall
./04_natural_watertap_eval_heldout.sh       # test / holdout_shift, and the seed-only control

# 05 (04's commands, on 05_units_and_scans.py; the data is regenerated with the corrected A factor)
$W 05_units_and_scans.py dump-tools           # the two served declarations + convert_units
$W 05_units_and_scans.py generate             # 1,800 questions; check reports which are 04's byte for byte
python3 05_units_and_scans.py check
$W 05_units_and_scans.py reference            # blind references: convert, simulate every candidate, answer
../../membrane_grpo/.venv/bin/python 05_units_and_scans.py check-harness
./05_units_and_scans_seed.sh                  # the label seed (k=2) and its greedy / T=1 dev probes
./05_units_and_scans_ppo_run.sh               # PPO from it (stopped by its tool gate at step 244)
./05_units_and_scans_eval_heldout.sh          # best checkpoint and seed on test / holdout_shift
PY=<python with mlx-lm> MLX_4B=<4-bit Qwen3.5-4B> ./05_units_and_scans_mlx_probe.sh   # untrained 4B / 0.8B on dev, on Apple silicon

# 06 (05's commands, on 06_direction_and_compare.py; train drawn again from the wider sentence bank)
$W 06_direction_and_compare.py dump-tools         # the two served declarations + convert_units + check_limits
$W 06_direction_and_compare.py generate            # train drawn again (one more sentence per limit); dev / test / holdout_shift are 05's byte for byte
python3 06_direction_and_compare.py check
$W 06_direction_and_compare.py reference           # blind references: convert, simulate every candidate, check the limits, answer
../../membrane_grpo/.venv/bin/python 06_direction_and_compare.py check-harness
./06_direction_and_compare_diagnose.sh             # 05's seed with holdout_shift's sentence swapped in and out: is it the sentence?
./06_direction_and_compare_seed.sh                 # the dose sweep: SFT on the first N reference episodes, N doubling from 10, each probed at T=1 and judged alive
DOSE=<dose> ./06_direction_and_compare_seed.sh     # a dose's twice-sampled dev probe (the value head's warm start)
./06_direction_and_compare_ppo_run.sh              # PPO from the 600 seed with the turn as the unit of credit, lambda 0.95 per turn, tool gate off (ppo-n600-open-turn-s0); its header lists the knobs of the other runs
LR=2e-6 RUN=ppo-n600-open-turn-lr2e6-s0 ./06_direction_and_compare_ppo_run.sh                                      # the same at a fifth of the actor step
for S in 175 200; do ../../membrane_grpo/.venv/bin/python 06_direction_and_compare.py probe --split dev --stop-at-json 1 --temperature 1 --samples 2 \
    --adapter 06_direction_and_compare_runs/ppo-n600-open-turn-lr2e6-s0/adapter-step$S --out ppo-n600-open-turn-lr2e6-s0/step$S; done   # its steps 175 and 200 sampled on dev as the seed was
ADV_UNIT=token LAM=1 RUN=ppo-n600-open-lam1-s0 ./06_direction_and_compare_ppo_run.sh                               # the token as the unit at lambda 1: the run that over-converted
ADV_UNIT=token LAM=0.95 SAMPLE_EVERY=5 RUN=ppo-n600-open-s0 ./06_direction_and_compare_ppo_run.sh                  # the token at 05's lambda 0.95: the run that collapsed
ADV_UNIT=token LAM=0.95 SAMPLE_EVERY=5 RUN=ppo-n1200-open-s0 SEED_RUN=label-seed-n1200 ./06_direction_and_compare_ppo_run.sh     # the next dose
./06_direction_and_compare_eval_heldout.sh         # the seed and a run's best (when not step 0) on test / holdout_shift; SEED_RUN= PPO_RUN= for the others

# 07 (06's questions and blind references, read in place; runs on CUDA or Apple silicon's MPS)
./07_recover_from_errors_reference.sh              # the recovery episodes of train and dev (one input slipped, the simulator's errors, the fix), the harness and data checks
./07_recover_from_errors_seed.sh                   # the seed (200 of 06's reference episodes + 100 recovery episodes, one epoch) and the control (06's 300); dev at T=1, alive, the forced probe
./07_recover_from_errors_ppo_run.sh                # PPO from the seed with 06's last configuration (turn credit, lambda 0.95 per turn, lr 2e-6, tool gate off)
SEED_RUN=seed-n300-r0 RUN=ppo-n300-r0-turn-s0 ./07_recover_from_errors_ppo_run.sh   # the same from the control
./07_recover_from_errors_eval_heldout.sh           # seed, control and the run's checkpoints on test / holdout_shift; the forced probe on the run's checkpoints

# 07, error recovery (06's questions and clean references, read in place; CUDA)
$W 07_error_recovery.py recovery                    # a recovery episode for every train question: one slip, the tool's real error, the fix
./07_error_recovery_seed.sh                         # the dose ladder from 150, every 4th episode a recovery episode (300 alive), and 06's seed of that dose on the same probe
07_error_recovery_diag/seed_variants.sh             # three more 300 seeds, among them recovery-seed-n300-maskend (the omission's <|im_end|> unlabelled)
SEED_RUN=recovery-seed-n300 ./07_error_recovery_ppo_run.sh                        # PPO from the 300 seed, 200 steps (ppo-n300-turn-lr2e6-s0)
SEED_RUN=recovery-seed-n300-maskend ERR_CREDIT=0 ./07_error_recovery_ppo_run.sh   # from the corrected seed, no call credit for a rejected call: the later runs' control
07_error_recovery_diag/probes.sh                    # the step-200 probes; the later runs and their probes: 07_error_recovery_diag/README.md

# 08.5 (08's PPO with no PyTorch: everything on mlx-lm, Apple silicon; setup, versions, memory and the infra switches: 08.5_mlx_stack_RECIPE.md)
08.5_mlx_stack_tools/launch_9b.sh                  # PPO with the Qwen3.5-9B critic (layer 24), shared prefix, the delta rule's custom-VJP scan, the update's token cap
CKPTS="adapter-step200:200" PPO_RUN=ppo-mlx-qwen3.5-9b-l24-n32-lr1e6-err0-delta-s0 ./08.5_mlx_stack_eval_heldout.sh   # a checkpoint's held-out probes
../../membrane_grpo/.venv/bin/python 08.5_mlx_stack_tools/rescore_probes.py <probe dir> ...                           # both scorings (default / err0), offline
```

The notebooks read finished runs and cached measurements; they retrain nothing.
