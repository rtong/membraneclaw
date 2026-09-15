# The runs from before the five fixes

Kept because they are the evidence for the defects, not because they are results.
Both are 200 steps of Qwen3-1.7B, MAIN and ABLATE weights, and both were produced
by the version of `ppo_ac.py` at commit `c2ab487`.

What they demonstrate, measurable straight out of `metrics.jsonl`:

* **The clipped surrogate never clipped anything.** `inner_epochs=1` means the
  only gradient pass has `logp == logp_old`, so `rho` is identically 1 and
  `min(rho*A, clip(rho)*A)` is just `A`. Across all 400 steps of the two runs,
  `ratio_mean` has min and max both exactly `1.000000000` and `clip_frac` is
  exactly `0.0`. These are vanilla policy gradient with a learned baseline.

* **Per-token credit assignment never happened.** At `lam=1`, `A_t = R - V(s_t)`
  and `R` is constant within a sequence, so every per-token difference comes from
  `V`. Median within-sequence `adv_std` is 0.011 against a per-sequence
  `|adv_mean|` of 0.172 — 6.4%. The other 93.6% is a scalar broadcast to every
  token, which is exactly what GRPO does and is the thing this method exists to
  do differently.

* **The critic was a near-constant.** Median `value_ev` +0.005, `value_std` 0.012
  against a `value_mean` near 0.32, `|dV|` 0.0033 per step. The head received one
  update per rollout batch.

`main-s0/` is metrics only, recovered from git — its adapter and figures were
overwritten on disk when the re-run started. `ablate-s0/` is complete apart from
the figures, which are regenerated from `metrics.jsonl` by notebook 02 and were
dropped rather than kept stale.

Do not read these as a baseline for the fixed runs. They are the same task and
the same model, but the algorithm differs in four places, so the comparison that
matters is defect-by-defect, not reward-by-reward.

## `paired/`

The three greedy evaluations behind the pre-fix round's McNemar test — the frozen
policy, and the `MAIN` and `ABLATE` adapters as they stood at commit `c2ab487`.
Each carries a full `per_case` block, so that test is re-checkable without
re-running anything.

    frozen -> MAIN      0.235 -> 0.290   13 gained,  2 lost   p = 0.0074
    frozen -> ABLATE    0.235 -> 0.285   13 gained,  3 lost   p = 0.0213
    MAIN   -> ABLATE    0.290 -> 0.285    0 gained,  1 lost   p = 1.0000

Moved here rather than deleted because the adapters that produced them have been
overwritten by the re-runs, so these files are the only remaining record of what
those policies did case by case.
