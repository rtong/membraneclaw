# 07: analysis scripts and the diagnosis of the route decision

Scripts that import `07_error_recovery.py` and read `07_error_recovery_runs/`. Each one's docstring gives its usage.

Probe re-scoring and comparisons
- `rescore_err0.py`: re-scores a saved probe under `--err-call-credit 0` (writes the `-err0` files in the run dirs).
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
- `window_mix.py`: per 25-step window, each group's convert-first share and reward by first call, with the change of
  the route log-odds.
- `pad_check.py`: token log-probs from the padded batch vs the episode alone vs its prefix (differences are bf16 noise).

The attribution is first order (no PPO clip, the step-200 policy and Adam preconditioner held fixed): its signs agree
with the actual change of the route log-odds in all three runs, its magnitudes do not (0.3x to 8x).
