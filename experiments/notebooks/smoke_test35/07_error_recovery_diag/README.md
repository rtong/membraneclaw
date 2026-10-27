# 07: launch scripts, analysis scripts and the diagnosis of the route decision

Scripts that import `07_error_recovery.py` and read or write `07_error_recovery_runs/`. Each one's header or docstring
gives its usage. The notebook (`../07_error_recovery.ipynb`) reads their outputs; the GPU-only ones (the decision probe,
the attribution) are read from the files they wrote.

Runs and probes (besides `../07_error_recovery_seed.sh`, the dose ladder, and `../07_error_recovery_ppo_run.sh`, the
first three PPO runs)
- `seed_variants.sh`: the three other 300-episode seeds (unconverted slips only; the omission's `<|im_end|>` unlabelled;
  both), with their dev T=1 x2 and forced probes.
- `probes.sh`: the probes the runs' own scripts do not make (dev T=1 x2 and forced on the first three runs' step 200,
  forced on the seeds, test and holdout_shift on the two uncentred step-200s).
- `seed_heldout.sh`: the two seeds on test and holdout_shift.
- `to400.sh`: the err0 run continued from its step 200 to 400. `lnbatch.sh`: the same branch with `--loss-norm batch`,
  to 300.
- `exp1_td.sh` (with `td_precheck.py`, its gate 0): first-turn TD at lambda 0 and 0.5, with per-turn call credit.
  `exp3_actorlam.sh`: actor lambda 0.5 on every turn. `prov_ppo.sh`: provenance (with `prov_primary.py`, its primary
  readout). `exp2_rollouts.sh`: four times the questions per step (notebook 08).
- `EXP1_PROMPT.md`, `EXP2_PROMPT.md`, `templates/`: the hand-off prompts for experiments 1 and 2, and the CPU tests of
  the trainer flags (each checks the flag off against the trainer before it).

Probe re-scoring and comparisons
- `rescore_err0.py`: re-scores a saved probe under `--err-call-credit 0` (writes the `-err0` files in the run dirs).
- `rescore_prov.py`: the same under `--err-call-credit 0 --provenance 1` (the `-err0-prov` files).
- `compare_runs.py`: two probes of the same questions, paired per question (reward, decisions, recovery; 95% intervals).
- `forced_report.py`: the forced probe (PR #33's 144 dev unit slips, greedy) and how each retry was made.
- `route_returns.py`: training samples per window and first call, with each route's mean return.

Why "convert first" is lost after step 200
- `diag_common.py`: fixed probe prompts, the route tokens ("=" then "convert" vs "=s"), and rebuilding a stored
  training sample into the Episode the trainer built (rewards and masks reproduced exactly).
- `decision_probe.py` -> `decisions.jsonl`: on every saved checkpoint, teacher-forced log-odds of a first convert_units
  call by question group, P(line break after the aggregation line), and the per-token log ratio to the seed by turn kind.
- `attribution.py` -> `attr_{A,orig,new}.json`: from a run's step-200 state (adapter, critic, Adam moments) and its real
  batches of steps 200-224, the first-order change each part of the update makes to the route log-odds, by token kind
  and by advantage part (task reward, KL penalty, centring).
- `route_attr.py` -> `ra_{A,orig,new}.jsonl`: the route tokens' part of it per episode (question group, first call,
  advantage, trainable tokens); `ra_summary.py` sums it up.
- `route_contrast.py`: the within-question reward of converting first minus simulating first, per question group and
  50-step phase, the runs that drew the same questions pooled.
- `window_mix.py`: per 25-step window, each group's convert-first share and reward by first call, with the change of
  the route log-odds.
- `pad_check.py`: token log-probs from the padded batch vs the episode alone vs its prefix (differences are bf16 noise).

The attribution is first order (no PPO clip, the step-200 policy and Adam preconditioner held fixed): its signs agree
with the actual change of the route log-odds in all three runs, its magnitudes do not (0.3x to 8x).
