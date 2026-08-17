# membrane_grpo

GRPO on a 0.5B instruct model, over a structured membrane-troubleshooting task
with a deterministic reward.

**Scale: this is a smoke test.** One short run of a few hundred steps on a Mac
mini, not a converged training run. Every claim it produces is scoped to that.
The point is to run the full loop — prompt, sampled completions, deterministic
reward, policy update — by hand, and to get a measured example of reward rising
while held-out accuracy does not.

This picks up where [`../toy_mdp`](../toy_mdp) left off. That project derived
REINFORCE and PPO-clip by hand on a six-state MDP; this one keeps the domain and
the hand-derived-gradient habit, and swaps the tabular policy for a language
model.

## The task

Given two RO operating records — a baseline and a current reading — compute
three normalized percent changes, threshold each into a trend flag, and read a
root cause and corrective action off a decision table. Answer as one JSON object.

```json
{
  "normalized_flow_change_pct": -20.6,
  "salt_passage_change_pct": 24.3,
  "dp_change_pct": 17.8,
  "flags": {"flow": "down", "salt_passage": "up", "dp": "up"},
  "stage": "tail",
  "root_cause": "scaling",
  "action": "acid_clean_low_ph"
}
```

Everything needed is in the prompt: the correction formula, the thresholds, the
table, the action lookup. The task is **closed-book** on purpose — a 0.5B model
has no reliable RO domain knowledge, and an experiment about RL should not be
bottlenecked on knowledge the base model never had. What is measured is whether
RL improves execution of a stated procedure.

The reward is computed by `task/decision_table.py` and nothing else. No LLM
judge, no fuzzy matching.

Data is synthetic and every parameter is hand-picked for teaching. See
[`data/DATA_CARD.md`](data/DATA_CARD.md) — this is not a diagnostic tool.

## Status

| Phase | | |
| --- | --- | --- |
| P0 | scaffold, throughput probe | done — **on the Mac; void, see below** |
| P1 | data generator, frozen splits | done, and machine-independent |
| P2 | reward function + adversarial baselines | done |
| P0' | re-measure on `anton` (CUDA, vLLM available) | next, needs SSH |
| P3 | frozen baseline, go/no-go on cold-start validity | |
| P4 | `grpo_scratch.py` — hand-written GRPO | |
| P5 | short run + probe-reward control | |
| P7 | curves and memo | |

Training moved to `anton`, a CUDA box on the tailnet, after P2. The task layer
and the frozen data carry over untouched — that is what the standard-library-only
constraint bought. The P0 measurements below do not: they describe MPS on a Mac
mini and are kept as a record of how the design was arrived at, not as limits
that still apply. `probe_throughput.py` has a CUDA path and gets re-run there.

## Measured limits (P0)

Mac mini, M4-class, 16 GB unified memory, torch 2.13 / MPS, Qwen2.5-0.5B-Instruct
in bf16. Raw numbers in `runs/probe/throughput.json`; regenerate with
`.venv/bin/python probe_throughput.py`.

| | |
| --- | --- |
| Prompt length | 902 tokens (p50), 904 (p95) — nearly constant, the template dominates |
| Correct answer length | 70 tokens compact, 103 pretty-printed |
| Completion budget | 192 tokens — ~2x the pretty-printed answer |
| Generation, batch 32 | 290 tok/s, 0.67 s/sequence |
| Update (fwd+bwd, LoRA r=16), batch 1 | 1.29 s/sequence — the fastest measured |
| Update memory, batch 1 | 1.08 GiB retained; batch 4 OOMs on transients |
| Projected step, 4 prompts x 8 completions | ~62 s, or 58 steps/hour |
| Step split | update 41 s, generation 21 s |

Four findings that changed the design, all of which would have been wrong if
guessed:

**Generation throughput has a cliff between batch 12 and 16.** Not a taper — a
discontinuity, reproducible to within 2% across repeats:

| batch | 8 | 12 | **16** | 24 | 32 | 48 |
| --- | --- | --- | --- | --- | --- | --- |
| tok/s | 83 | 110 | **245** | 271 | 287 | 308 |
| s/sequence | 2.32 | 1.75 | **0.78** | 0.71 | 0.67 | 0.62 |

Batch 16 finishes 33% more work in *40% less wall time* than batch 12. Below the
cliff decoding is dispatch-bound; above it the matmuls are large enough to
saturate. The consequence for GRPO is concrete: never sample a group of 8 on its
own. Batch several prompts' groups into one call and stay at 16 or above. Taking
the obvious route of "group size 8, so batch 8" would have cost 3x throughput.

**The output head, not the model, is the memory bottleneck.** Qwen2.5-0.5B
carries a 151,936-token vocabulary on an 896-dimensional hidden state, so the LM
head is 136M parameters — 27% of the model — and the logits tensor dwarfs
everything else. At batch 4 over a full 1,096-token sequence that is 1.2 GiB in
bf16, ~2.5 GiB once upcast for the log-softmax, and the backward pass wants a
gradient buffer the same size again. The first probe run died there.

**So `logits_to_keep` is mandatory, not an optimisation.** Restricting the head
to completion positions cuts the logits by 5.7x, and it is what the algorithm
wants anyway: prompt tokens are given rather than sampled, contribute no
policy-gradient term, and their logits are never read. Computing log-probs as
`chosen - logsumexp` instead of gathering from a full `log_softmax` saves two
more full-size tensors.

**Gradient checkpointing buys nothing here.** Measured, not assumed — it was
verified engaged (`is_gradient_checkpointing == True` on the PEFT wrapper and on
the decoder layers) and made no difference to either the OOM threshold or the
timing. It shrinks decoder activations, and the decoder activations are not what
is large.

**The update's OOM ceiling is not a binding constraint.** Worth stating plainly,
because it looks like one. In a clean process a training step *retains* only
1.08 GiB — 0.93 of that is the weights. What fails at batch 4 is transient
allocation during the backward: MPS asks for 19.7 GiB, which is above this
machine's 16 GB and above the ~20 GiB ceiling it reaches by spilling into
swap-backed shared memory. But batch 1 is also the fastest configuration
measured, at 1.29 s/sequence against 1.44 s at batch 2 — batching the update
trades memory for speed and, on this backend, does not get the speed. Nothing is
lost by running at batch 1, and a larger-memory machine would move the wall
without moving the optimum.

The lever that would actually shorten a step is prompt length, not memory. The
update's forward pass covers the full 1,096-token sequence, of which 904 tokens
— 82% — are a prompt that is nearly identical across every case. Compressing the
template is worth roughly 1.5x on both phases, and is deliberately deferred
until after the frozen baseline: it would mean bumping `PROMPT_VERSION`, and
changing the template while a baseline is being established is how frozen
evaluations stop being comparable.

One caveat on the memory column: MPS reports `driver_allocated_memory`, which is
the allocator pool including cached blocks from earlier measurements. It marks
where things fall over; it is not a true per-step peak, and it runs far above
what is actually retained.

## What not solving the problem is worth (P2)

Priced before training, because a reward function is only as good as the
cheapest way to score well on it. `python3 baselines.py --split dev`:

| strategy | reward | easy | hard | exact match | valid |
| --- | --- | --- | --- | --- | --- |
| `empty` / `prose` / `schema_template` | 0.000 | | | 0.00 | 0.00 |
| `constant` — valid JSON, same guess every time | **0.245** | 0.236 | 0.262 | 0.00 | 1.00 |
| `copy_stage_only` | 0.263 | 0.254 | 0.280 | 0.00 | 1.00 |
| `skip_correction` — everything right but the TCF | **0.890** | 1.000 | 0.690 | 0.65 | 1.00 |
| `oracle_verbose` — correct, fenced, one extra key | 0.900 | | | 1.00 | 1.00 |
| `oracle` | 1.000 | | | 1.00 | 1.00 |

Two numbers to keep in view for the rest of the project:

**0.245 is the floor, not zero.** A model that learns nothing except to emit
schema-valid JSON with a fixed guess collects a quarter of the available reward
— 0.10 for format outright, and the rest from luck on a balanced seven-way label
and a two-way stage. Any reward curve has to be read against this line.

**0.890 with exact match 0.65 is the signature of `skip_correction`.** If the
run parks near those two numbers together, the model has learned to do the
arithmetic without the temperature correction — and its exact match is 0.65
because that is precisely the `easy` fraction of the dev split. This is the
sharpest prediction the project has, and it is the one that would make
prediction 3 concrete.

The probe weights make a third point on their own. Under `PROBE`,
`skip_correction` scores 0.930 while `oracle_verbose` scores 0.650: a strategy
that gets the diagnosis wrong on 12% of cases beats one that gets every field
right but adds a stray key. That inversion is the misspecification the control
is built to demonstrate, and it is visible before a single gradient step.

## Pre-registered predictions

Written before the first training run, and to be scored honestly afterwards even
where wrong — the same convention as `toy_mdp`, which records a baseline
variance-reduction result that failed to reproduce as a speedup.

1. **Format saturation dominates the reward curve.** Schema validity climbs from
   its cold-start value to near 1.0, total reward rises substantially, and
   `root_cause` exact match moves much less.
2. **The probe-reward control separates the two outright.** Same task, same
   model, same code — only the reward weights change, de-emphasising
   `root_cause`. Reward should climb *faster* while held-out exact match stays
   flatter.
3. **The `hard` tier does not move.** Any gain in exact match comes from `easy`
   cases. A 0.5B model will not learn to evaluate `1.03 ** (25 - T)` from
   reinforcement on the final answer.
4. **pass@1 rises while pass@8 is flat or falls**, with diversity collapsing
   alongside. If so, the reward gain is the sampling distribution narrowing onto
   answers the base model could already produce — not new capability. This is
   the main thesis of the memo.
5. **A visible fraction of groups will have zero reward variance**, giving
   identically zero advantages and no gradient. Logged as `adv_zero_frac`.

The `holdout_shift` slices exist to test 3 and 4 under distribution shift and are
never trained on.

## Layout

| Path | |
| --- | --- |
| `task/decision_table.py` | the answer key: 17 symptom combinations, 7 causes |
| `task/generate.py` | backward case construction; the only grader |
| `task/prompt.py` | the frozen template, `PROMPT_VERSION` |
| `task/schema.py` | lenient parser (the reward gate) and strict validator |
| `probe_throughput.py` | P0 measurements |
| `data/` | frozen splits, `SHA256SUMS`, `DATA_CARD.md` |
| `test_*.py` | tests for each of the above |

## Running

The task layer is standard-library only, so the data and its tests need no
install:

```bash
python3 -m pytest
```

Regenerating the data is byte-reproducible from the seed, and deliberately
breaks `test_checksums_match_the_files_on_disk` until `SHA256SUMS` is rewritten:

```bash
python3 -m task.generate
```

Anything touching the model needs the experiment's own venv, which does not use
the repository's `.venv` and its much heavier WaterTAP stack:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python probe_throughput.py
```
