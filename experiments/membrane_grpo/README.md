# membrane_grpo

GRPO on a 0.5B instruct model, over a structured membrane-troubleshooting task
with a deterministic reward.

**Scale: this is a smoke test.** Two 200-step runs, not converged training.
Every claim is scoped to that. The point was to run the full loop — prompt,
sampled completions, deterministic reward, policy update — by hand, and to get
a measured example of reward rising while held-out accuracy does not.

Both were achieved. Held-out reward rose 0.086 → 0.279 while `root_cause`
accuracy read 0.145 at all nine evaluation points, against a 1/7 = 0.143 chance
floor — and a second run, identical but for the reward weights, reports the same
policy as a 19x improvement instead of a 3.2x one. [`MEMO.md`](MEMO.md) is the
write-up; `runs/grpo_curves.png` is the picture.

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
| P0 | scaffold, throughput probe | done — **on the Mac; superseded, see below** |
| P1 | data generator, frozen splits | done, and machine-independent |
| P2 | reward function + adversarial baselines | done |
| P0' | re-measure on `anton` (CUDA) | done |
| P3a | `eval.py`, 9B reference run | done — **found a task-design bug, see below** |
| P3b | prompt v2 | done — validated against the 9B |
| P3c | the frozen 0.5B baseline | done — go/no-go passed |
| P4 | `grpo_scratch.py` — hand-written GRPO | done — **exit gate met** |
| P5 | short run + probe-reward control | done |
| P7 | curves and memo | done — [`MEMO.md`](MEMO.md) |
| P8 | model selection, then three weight sets on Qwen3-1.7B | done — the diagnosis moves |
| P9 | seed replication of MAIN and ABLATE | done — **P8's explanation does not replicate** |
| P10 | the oracle decomposition, no training | done — **the bottleneck is not arithmetic** |

P8 onwards is why the memo has three parts rather than one. P8 found a reward
weighting that appeared to move held-out `root_cause` 0.295 → 0.430 at p < 1e-4;
P9 re-ran it at seed 42 and got 0.340, with the seed accounting for more of the
difference than the weighting; P10 stopped training altogether and asked what
the ceiling was, which turned out to be far below where the runs were aiming.

Training moved to `anton`, a CUDA box on the tailnet, after P2. The task layer
and the frozen data carry over untouched — that is what the standard-library-only
constraint bought. The P0 measurements below do not: they describe MPS on a Mac
mini and are kept as a record of how the design was arrived at, not as limits
that still apply. `probe_throughput.py` has a CUDA path and gets re-run there.

## What the 9B reference run found (P3a)

The point of pointing a much larger model at the task first was to ask whether
the task is well posed before blaming a 0.5B for failing it. It is not, quite —
and that is worth more than a clean number would have been.

Qwen3.5-9B-AWQ, greedy, 40 dev cases, both versions on the same cases with the
same decoding — `python3 prompt_ab.py --model qwen3.5-9b -n 40`, artifact in
`runs/prompt-ab/`:

| | v1 | v2 |
| --- | --- | --- |
| validity | 1.000 | 1.000 |
| reward | 0.624 | **0.750** |
| exact match | **0.000** | **0.325** |
| all three flags | 0.450 | 0.550 |
| cause accuracy | 0.675 | 0.650 |
| action accuracy | 0.525 | 0.675 |
| completion tokens | 104 | **531** |

Numeric accuracy within 0.5 pp, per field — this is where the two versions
actually differ:

| field | v1 | v2 | v1 p90 error | v2 p90 error |
| --- | --- | --- | --- | --- |
| `normalized_flow_change_pct` | 0.375 | 0.675 | 17.7 | 10.1 |
| `salt_passage_change_pct` | 0.125 | 0.900 | 111.3 | 0.7 |
| `dp_change_pct` | 0.025 | **1.000** | 51.1 | **0.00** |

Under v2 the median absolute error on all three fields is **0.00**: when it is
right it is exact, and the failures are outright mistakes rather than drift.

**The model was not doing the arithmetic — it was estimating.** Under v1 it
emits perfectly formed JSON with plausible-looking numbers. On one case it got
every flag, the cause and the action right while all three numbers were wrong;
its estimates happened to land in the right threshold bands. On another it put
salt passage at 16.3 against a true 159.8. It did not even apply its own quoted
thresholds consistently: it labelled a flow change of 10.5 as `flat` when the
prompt states `up if >= +10`.

**Reasoning mode is unusable here**, exactly as `agent/config.py` in the parent
repo records: the model spends the entire budget inside `<think>` and returns
empty content. All eight cases failed the gate with `empty` — the exploratory
artifact is in `runs/9b-v1-exploratory/v1_thinking_n8.json`.

**Letting it show terse work is what fixes the arithmetic**, at 5.1x the
completion length. Both halves of the prompt have to change; editing only the
user turn does nothing, because the system prompt's "reply with exactly one JSON
object and no other text" wins and the model goes straight to JSON. That was
measured the hard way — the first attempt at this comparison changed the user
turn alone and reported no effect at all.

### A real ambiguity in the task, not a model failure

`dp_change_pct` was the worst field under v1 at 0.025, and it was the *simplest*
of the three computations — a sum and a percent change, no ratio, no correction.
That is what gave the diagnosis away. On the first case inspected the model
answered 30.2 where the key says 17.8, and `0.28 / 0.93 = 30.1`: the change
measured against **the anomalous stage's own dp** rather than the train total.
The prompt does state `dp(t) = dp_lead(t) + dp_tail(t)`, but the record also
announces which stage the anomaly sits in, and the generator puts the whole dp
change on that stage, so the two readings diverge sharply. A reasonable reader
picks the wrong one.

Printing the total in v2 took that field from **0.025 to 1.000** — the clearest
evidence available that this was an ambiguity in the task rather than a limit of
the model.

**Loosening the tolerance would not have helped.** Widening the band barely
moves either version, because the errors are bimodal rather than noisy:

| all three numbers within | 0.5 pp | 1.0 pp | 2.0 pp | 5.0 pp |
| --- | --- | --- | --- | --- |
| v1 | 0.000 | 0.000 | 0.000 | 0.050 |
| v2 | 0.600 | 0.650 | 0.675 | 0.775 |

### What v2 changes

Fixing the task, not accommodating the model:

1. system and user prompts both permit terse working, JSON last;
2. the record states the dp **total** alongside the per-stage values;
3. intermediate values keep four significant figures — salt passage is a percent
   change *of a ratio*, and rounding the ratio early was costing it accuracy.

`parse_answer` changes to match: with working allowed it selects the object
carrying the most answer keys rather than the first one it finds, since a stray
object in the arithmetic would otherwise be graded as the answer.

### Two results v2 did not flatter

**Cause accuracy went slightly down**, 0.675 to 0.650, while every other metric
rose. Under v1 the model was diagnosing qualitatively — reading the direction of
travel and picking a plausible cause — and that is surprisingly effective. Under
v2 it computes first and then derives the cause from its own flags, so a failed
computation now drags the diagnosis down with it. This is the component cascade
described in `reward.py`, showing up as a measured regression: **being made to
compute can hurt the diagnosis on cases where the computation fails.** It is
also why `diagnostics["cause_given_flags"]` exists.

**The hard tier is still a wall.** v2 exact match splits `easy=0.462` against
`hard=0.071`. Even a 9B showing its working mostly cannot evaluate
`1.03 ** (25 - T)` correctly. The tier is doing exactly the job it was designed
for, and prediction 3 — that RL will move the easy tier and not the hard one —
now has a reference point well above anything a 0.5B will reach.

**Working costs 5.1x the tokens**, 104 to 531. The P0' step estimates assumed a
192-token completion budget, so they need redoing against roughly 640 before the
training run is sized.

Changing `PROMPT_VERSION` was cheap here and would not have been later — no
frozen baseline depended on v1, which is precisely why the reference run came
before the baseline rather than after it.

## The frozen baseline, and the go/no-go (P3c)

Qwen2.5-0.5B-Instruct, prompt v2, dev split (`sha256 94b32d05…`), seed 0.
Artifacts in `runs/baseline-0.5b-v2/`. The test split remains sealed.

| greedy, pass@1 | dev | holdout_shift |
| --- | --- | --- |
| reward | **0.086** | 0.098 |
| exact match | 0.000 | 0.000 |
| validity (gate) | **1.000** | 1.000 |
| schema valid | 0.005 | — |
| cause accuracy | **0.145** | 0.160 |
| numeric (of 3) | 0.000 | 0.000 |
| completion tokens | 108 | — |

Three things this pins down.

**Cause accuracy is chance.** 0.145 against 1/7 = 0.143. The base model has no
diagnostic ability on this task whatsoever; 76% of its reward (0.065 of 0.086)
is the `root_cause` component paying out on luck.

**It scores worse than a constant guesser.** `baselines.constant` earns 0.245;
the model earns 0.086. It loses because its flag values are out of vocabulary —
it writes the *threshold text* into the flag field, `"flow": "<-10"`, and copies
the threshold constants into the numbers. `flags.bad_value` fires 595 times
across 200 cases, almost exactly three per case, and `stage` is out of
vocabulary on 199 of 200. So schema validity is 0.005 and the format component
is worth 0.0005. A model that learned nothing except to emit the right
vocabulary would nearly triple this score.

**v2's benefit does not transfer.** The 0.5B ignores the instruction to show
working and goes straight to JSON at 108 tokens — the same failure v1 produced
in the 9B. Whatever it gains from v2 is not the arithmetic.

### The go/no-go: sampled diversity

The question that decides whether the experiment can run at all is not accuracy,
it is whether a group of completions carries any reward variance. Without it
GRPO's advantages are identically zero and there is no gradient. T=1.0, k=8, 200
dev cases:

| | |
| --- | --- |
| zero-variance groups | **0.160** |
| unique answers per group of 8 | **5.54** |
| distinct-4 | 0.727 |
| validity under sampling | **0.665** |
| pass@8 | **0.000** |

**84% of groups carry usable gradient.** Prediction 5 does not block the run.

**But pass@8 is zero.** Across 1,600 samples the model never produced a fully
correct answer. That turns the P2 decision to loosen the gate and score partial
credit from a judgement call into a precondition: under a binary exact-match
reward every group here would score zero, `adv_zero_frac` would be 1.0, and the
gradient would be exactly zero everywhere. **The partial credit is what makes
this trainable at all.** It also means GRPO cannot reinforce a correct answer
directly — only the pieces of one.

**Sampling breaks the format.** Validity falls from 1.000 greedy to 0.665 at
T=1.0, with all 67 failures being `no_json`. That gap is where most of the
current reward variance lives, so the earliest thing GRPO can learn is to stay
parseable while sampling — which is prediction 1, arriving before the first
gradient step.

## The update, written out by hand (P4)

`grpo_scratch.py` is the exit gate: sample a group, score it deterministically,
centre the rewards within the group, take one clipped policy-gradient step. No
TRL. `../toy_mdp/ppo.py` derives the clipped surrogate's gradient by hand on a
tabular policy, and the derivation survives the move to a language model — what
changes is only that autograd carries the chain rule the rest of the way:

    d/d logp  min(rho*A, clip(rho)*A)  =  rho * A   unless the clip binds
                                          0         when A > 0 and rho > 1+eps
                                          0         when A < 0 and rho < 1-eps

`test_grpo_scratch.py` checks autograd against that closed form, including in
the regime where the clip actually binds. At `rho = 1` it collapses to `A`, so
**with one inner epoch this is REINFORCE with a group baseline** and the clip is
inert by construction; it only starts working when a batch is reused.

Three things the tests caught that a smoke run would not have:

**A run that prints a loss can change nothing.** The first end-to-end CPU run
completed two steps and moved no weights — the token budget truncated every
completion before any JSON appeared, so all four rewards were zero, the group
was degenerate, and the gradient was correctly zero throughout. "The loop ran"
and "the policy moved" are separate claims and are now separately tested.

**A degenerate group is not a no-op under AdamW's defaults.** Weight decay is
applied whether or not the gradient said anything, so a batch of degenerate
groups still shrinks the adapter — and 16% of groups are degenerate at the
frozen baseline. `Config.weight_decay` is therefore 0.0: in an experiment whose
question is what the *reward* moved, a force acting independently of the reward
is a confound, not a regulariser. Both behaviours are pinned by tests.

**EOS and padding are not the same event.** Choosing to stop is an action, and
an action that is never scored is one RL cannot learn to take; padding was never
sampled. Qwen2.5 has no distinct pad token so `load_policy` aliases them and the
distinction collapses in practice, but the mask keeps them apart anyway.

The loop, on CPU with a deliberately tiny configuration — one prompt, four
completions, three steps. `runs/smoke-grpo-cpu/`:

```
step  0  reward 0.0375  advzero 0.00  uniq 4.0  tok 114  grad 1.13655  ratio 1.000
step  1  reward 0.0825  advzero 0.00  uniq 4.0  tok 105  grad 1.26630  ratio 1.000
step  2  reward 0.0000  advzero 1.00  uniq 4.0  tok 100  grad 0.00000  ratio 1.000
```

Rewards sit around the frozen baseline's 0.086 and the gradient norm is non-zero,
which is the exit gate: prompt, sampled completions, deterministic reward, policy
update, all of it hand-written.

Two things fell out of it that were not planned.

**Step 2 is a degenerate group in the wild.** All four completions scored
exactly 0.0, the within-group variance vanished, and the gradient norm is
`0.00000` — the case the unit tests pin, happening on its own within three steps
at `adv_zero_frac = 1.0`. The frozen baseline measured 16% of groups this way;
here it was one in three.

**`ratio_mean` is 1.000 on every step**, confirming empirically what the
derivation says: with one inner epoch the policy has not moved between sampling
and scoring, so the clip cannot engage.

One caveat on this artifact. Its `config.json` has no `weight_decay` field
because the run predates that fix, so it used AdamW's default 0.01 — meaning on
step 2 the gradient was exactly zero and the adapter was decayed anyway. The
confound is visible in the very record that motivated removing it. The claim
this run supports is that the loop closes and the gradient flows, and that
claim is unaffected.

## Measured limits (P0', anton)

`NVIDIA RTX 5070 Ti, 16.3 GB` (Blackwell, sm_120) under WSL2, torch 2.13+cu130.
Raw numbers in `runs/probe/throughput_anton.json`.

| | |
| --- | --- |
| Prompt length (v2) | 974 tokens |
| Generation, batch 128 | **5874 tok/s**, 0.11 s/sequence |
| Update (fwd+bwd, LoRA r=16), batch 2 | **0.11 s/sequence**, 8.3 GiB peak |
| Update ceiling | **batch 2** — batch 4 spills, 8x slower |
| Projected step, 4 prompts x 8 completions | ~17.5 s, or **205 steps/hour** |

Measured at a 640-token completion budget, `runs/probe/throughput_anton_640.json`.
An earlier pass at 192 tokens reported 7 s/step and 503 steps/hour; v2's working
made that budget 3x too small, and the update's batch ceiling fell from 4 to 2
as the sequence grew from 1,096 to 1,614 tokens. The estimator is still
conservative — it sizes generation by the best measured batch (128) rather than
the 32 sequences a step actually needs, so the real figure is nearer 11 s.

Against the Mac's 62 s/step and 58 steps/hour, roughly 8.6x. Three things worth
recording:

**The MPS batch-16 cliff does not exist here.** Generation scales smoothly from
105 tok/s at batch 1 to 4807 at batch 128, confirming that discontinuity was a
kernel-selection artifact of the Metal backend and nothing about the workload.

**Full fine-tuning is a real option**, not a theoretical one: 10.5 GiB against
LoRA's 8.4 at the same batch, and only 23% slower per sequence. One caveat — the
figure is that low because torch's AdamW keeps `exp_avg`/`exp_avg_sq` in the
parameter dtype, so those states are bf16 rather than the fp32 a normal
mixed-precision setup would carry. LoRA sidesteps the question.

**Under WSL2 an oversized batch does not fail, it silently crawls.** Batch 8
peaks at 15.8 GiB and batch 16 reports 30.5 GiB — both above the card — because
WDDM lets GPU memory spill into host RAM over PCIe. Batch 16 still "works", at
11.5 s against batch 4's 0.26. That is more dangerous than an OOM, because the
configuration looks valid. **The practical ceiling is batch 4.**

## Superseded: measured limits on the Mac (P0)

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

## Model selection and the three weight sets (P8, P9)

The 0.5B result was confounded: half its reward gain was schema validity going
0.005 → 1.000, which is formatting rather than diagnosis, and its `cause_acc`
sat *exactly* at the 1/7 chance floor, leaving RL nothing partial to sharpen.
Three candidates, measured on the same 200 dev cases (`runs/sel-*`) rather than
argued about:

| frozen, dev, greedy | reward | `cause` | `numeric` | `schema` | tokens |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B-Instruct | 0.000 | 0.000 | 0.000 | 0.000 | 638 |
| Qwen2.5-1.5B-Instruct | 0.209 | 0.215 | 0.000 | 0.000 | 108 |
| **Qwen3-1.7B** | **0.315** | **0.255** | **0.038** | **0.970** | 98 |

Three unrelated failure modes. Math-1.5B writes 638 tokens of arithmetic and
never emits a JSON object — `no_json` 200/200 — trading instruction-following
for exactly the capability wanted. Qwen2.5-1.5B has one mechanical defect: the
numeric fields come out as strings, so `not_a_number` fires 600 times over 200
cases. Qwen3-1.7B was chosen less for its score than because its schema validity
**starts** at 0.970, which leaves at most 0.003 of the headroom that confounded
the 0.5B — so a reward rise on this model cannot be format learning.

The pre-registered go/no-go for this choice was `pass@8 > 0`, and it rejected all
three. It was overridden, and the scoring below explains why it was the wrong
criterion.

### Three weight sets, one variable (P8)

200 steps each, seed 0, identical but for the reward weight vector. Held-out
`root_cause`, greedy, McNemar against the frozen policy on the same 200 cases:

| | `numeric` wt | `cause` wt | `cause_acc` | vs frozen |
| --- | --- | --- | --- | --- |
| frozen | | | 0.255 | |
| `MAIN` | 0.15 | 0.45 | 0.295 | p = 0.057, **not significant** |
| `ABLATE` | **0.35** | 0.25 | **0.430** | p < 1e-4 |
| `PROBE` | **0.35** | **0.10** | **0.450** | p < 1e-4 |

`PROBE` is the control, built to show that reward can rise while capability does
not — weight moved *away* from `root_cause`, 0.45 → 0.10. It produced the best
diagnosis of the three. `ABLATE` was added to isolate which of the four moved
components did it, and it tracked `PROBE`.

### The seed replication (P9)

`MAIN` and `ABLATE` re-run at seed 42, everything else identical:

| held-out `cause` | seed 0 | seed 42 |
| --- | --- | --- |
| `MAIN` | 0.295 | 0.315 |
| `ABLATE` | **0.430** | **0.340** |
| `ABLATE` − `MAIN` | **+0.135**, p < 1e-4 | +0.025, p = 0.30 |

It does not replicate, and the seed moves `ABLATE` further than the weighting
does: `ABLATE` seed 0 against seed 42 is −0.090, 21 discordant pairs to 3,
**p = 0.0003**. `MAIN` is stable (+0.020, 8 discordant, p = 0.29).

The error was reading McNemar's p < 1e-4 at one seed as "the effect is real".
The test asks whether *these two policies* differ on *these 200 cases*; it says
nothing about whether the weighting reliably produces such policies. Only
repetition supports the second claim, and a small p-value on a single run is not
a substitute for it.

`PROBE` was re-run at seed 42 as well and reached 0.425, but **that run's
artifacts were lost** — it executed in `/tmp` on the training box, which was
cleared by a reboot before the results were copied back. The number is therefore
not reported as a result anywhere in this repository, and the run needs redoing.
Every other figure here comes from a committed artifact in `runs/`.

## The oracle decomposition (P10)

Three evaluations of the **frozen** Qwen3-1.7B, no training, 83 seconds each,
differing only in how much of the stated procedure the prompt hands over:

| version | Step 1 | Step 2 |
| --- | --- | --- |
| `v2` | model computes the three percent changes | model derives the flags |
| `v2-oracle-num` | **given** | model derives the flags |
| `v2-oracle-flags` | **given** | **given** |

The injected values come from `task.generate.truth_from_record`, the project's
only grader, so they are *derived from the record* by the same function that
writes the answer key rather than copied out of it — a mislabelled case would be
injected wrong and caught. `test_the_oracle_changes_only_the_step_it_names`
pins that everything before Step 1 and everything from Step 3 to the schema is
byte-identical to `v2`; `PROMPT_VERSION` stays `"v2"`, so every committed run
remains comparable.

```sh
python3 eval.py --backend hf --model Qwen/Qwen3-1.7B --split dev --mode greedy \
  --max-tokens 640 --batch-size 16 --prompt-version v2-oracle-num --run-name oracle-num
```

### What it measured

| frozen, dev, greedy | `oracle-base` | `oracle-num` | `oracle-flags` |
| --- | --- | --- | --- |
| `numeric_acc` | 0.038 | 1.000 | 1.000 |
| `flags_acc`, per field | 0.330 | 0.580 | 1.000 |
| all three flags at once | 4/200 | 26/200 | 200/200 |
| **`cause_acc`** | **0.255** | **0.290** | **0.415** |
| `action_acc` | 0.195 | 0.205 | 0.225 |
| `exact_match` | 0.000 | 0.090 | 0.210 |
| `schema_ok` | 0.970 | 1.000 | 1.000 |

McNemar's exact test, same 200 paired cases:

| | difference | gained | lost | discordant | p |
| --- | --- | --- | --- | --- | --- |
| base → num | +0.035 | 9 | 2 | 11 | 0.0654, **not significant** |
| num → flags | +0.125 | 25 | 0 | 25 | **< 1e-4** |
| base → flags | +0.160 | 34 | 2 | 36 | **< 1e-4** |
| base → num, `flags_ok` | +0.110 | 26 | 4 | 30 | **0.0001** |

Perfect arithmetic — the ceiling of any calculator, tool or solver — is worth
+0.035 and does not clear significance. `numeric → flags` is real and
`flags → cause` is real, but the composite is not: `cause` needs all three flags
at once, and perfect arithmetic lifts the per-field rate 0.330 → 0.580 while
lifting the three-way conjunction only 0.02 → 0.13.

### Where it actually fails: four labels that are never emitted

With numbers and flags both given, the 17-row lookup is the whole remaining
task. `predicted_cause` says what it answered rather than only whether it was
right:

| true cause | n | emitted | correct | what it said instead |
| --- | --- | --- | --- | --- |
| `scaling` | 29 | 79 | **29/29** | — |
| `biofouling` | 29 | 90 | **29/29** | — |
| `colloidal_fouling` | 29 | 31 | 25/29 | `biofouling` ×4 |
| `organic_fouling` | 29 | **0** | 0/29 | `scaling` ×16, `biofouling` ×13 |
| `compaction` | 28 | **0** | 0/28 | `scaling` ×15, `biofouling` ×7, `colloidal` ×6 |
| `oxidation_damage` | 28 | **0** | 0/28 | `biofouling` ×22, `scaling` ×6 |
| `mechanical_leak` | 28 | **0** | 0/28 | `biofouling` ×15, `scaling` ×13 |

Four of seven labels are not in the model's output vocabulary at all, and the
three that are happen to be exactly the three rows requiring `dp = up`. Of the
0.745 between the frozen policy and a perfect one, **0.585 — 78% — survives
handing over everything upstream.** No tool addresses that.

`action` has the same shape plus one failure of its own:

* **The severity override is never applied.** *If the flow loss is past −30 the
  action is `isolate_and_evaluate_replacement` whatever the cause* covers 37 of
  the 200 cases. The model applies it **0 times in all three conditions**,
  including the one that hands it the flow percentage.
  `isolate_and_evaluate_replacement` is emitted 0 times in 200 — and it is
  reachable from no cause, so it appears only behind that one conditional.
* **The cause and action heads collapse onto different subsets.** Four action
  labels of eight, and not the four its own causes imply: it names
  `colloidal_fouling` 31 times but that cause's action 5, reaching for
  `compaction`'s action 26 times instead. The pair is not coming off the table
  together.
* **The lookup itself is fine.** Where the cause is right and the override does
  not apply, `action` follows it 39/39 and 41/41 in the first two conditions.

A guess that did not survive: the obvious reading of `colloidal_fouling`'s 1/20
was a label collision with the string-similar `alkaline_clean_and_sanitize`.
`predicted_action` was recorded rather than asserted, and it says the model
reaches for `no_clean_evaluate_replacement` — a different row entirely.

### Two controls, and the determinism check

`numeric_acc` is exactly 1.000 over 200 cases and three fields, so none of this
is a failure to transcribe what was given. And the gain is not a schema
artifact: none of the nine cases that flipped to a correct cause were among the
baseline's six `no_json`, and restricting to the 194 the baseline parsed leaves
the same picture, 0.263 → 0.299 → 0.428.

All three evaluations were run twice, a commit apart, to add `action`,
`predicted_cause`, `predicted_action` and `severe` to `per_case`. Every headline
number reproduced to three decimals. A frozen policy under greedy decoding at a
fixed batch size is deterministic on this box — which is why P10 is the only
result in this project with no seed for it to fail at.

`cause_given_flags` needed a denominator before any of this was readable. It had
been reporting 0.000 on the frozen baseline, which turns out to be 0 of **4** —
the model assembles a correct flag triple 4 times in 200. `cause_given_flags_n`
is now recorded alongside it.

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

### Scored

Owed since the first run, and worth more than it is comfortable to write down.
Two held, one inverted on the larger model, and two turned out to be untestable
for the same reason.

| | outcome |
| --- | --- |
| 1. format saturation dominates | **held**, and harder than predicted |
| 2. the probe control separates reward from capability | **held on the 0.5B, inverted on the 1.7B** |
| 3. the `hard` tier does not move | **untestable as written** |
| 4. pass@1 rises while pass@8 is flat or falls | **untestable as written** |
| 5. a visible fraction of groups has zero reward variance | **held** |

**1 held, and more sharply than it was written.** On the 0.5B, schema validity
went 0.005 → 1.000, reward 0.086 → 0.279, and `root_cause` did not move "much
less" — it read 0.145 at all nine evaluation points, identical to three decimals,
against a 1/7 = 0.143 chance floor.

**2 held where it was aimed and inverted one model up.** On the 0.5B the same
policy was reportable as a 19x gain under `PROBE`'s weights against 3.2x under
`MAIN`'s, which is the separation the control was built to produce. On
Qwen3-1.7B it reversed: `PROBE` de-emphasises `root_cause` 0.45 → 0.10 and
produced the *best* held-out diagnosis of the three, 0.450 against `MAIN`'s
0.295. The control was written to show that reward can rise without capability;
it showed that a reward weighted away from the target can produce more of it.
P10 explains why the whole family of weighting arguments was mis-aimed.

**3 and 4 were both written about `exact_match`, and that is the finding.** Both
predictions presuppose an exact-match gain to attribute — to a tier, or to
pass@1 against pass@8. There was none to attribute: EM is 0.000 for every 0.5B
and 1.7B policy in this project, trained or not, and pass@8 is 0.000 across
1,600 samples before training and 1,600 after. Neither prediction is right or
wrong; both point at an instrument with no dynamic range on this task.

What did move on the 0.5B was reward, and it moved on both tiers about equally —
`easy` 0.093 → 0.284, `hard` 0.072 → 0.270 — which is itself evidence for 1,
since an arithmetic gain would have shown up on `easy` first. `cause_acc` was
unchanged on both tiers to the digit (`easy` 0.155, `hard` 0.127, before and
after).

Three of five predictions were aimed at exact match. The go/no-go criterion was
too (`pass@8 > 0`, which rejected all three 1.7B candidates and had to be
overridden). The lesson is not that the predictions were sloppy — they were
specific and falsifiable — but that a conjunction over seven fields is the wrong
instrument to hang a research question on when no policy in reach ever satisfies
all seven. What actually distinguished the 0.5B's failure was `cause_acc` sitting
*exactly* at chance.

**5 held**: `adv_zero_frac` = 0.16, so 16% of groups drew identical rewards
across all eight samples and contributed an exactly zero gradient.

## Layout

| Path | |
| --- | --- |
| `task/decision_table.py` | the answer key: 17 symptom combinations, 7 causes |
| `task/generate.py` | backward case construction; the only grader |
| `task/prompt.py` | the frozen template, `PROMPT_VERSION` |
| `task/schema.py` | lenient parser (the reward gate) and strict validator |
| `reward.py` | the deterministic reward; six weighted components |
| `baselines.py` | degenerate strategies, and what the reward pays them |
| `eval.py` | frozen-split evaluation, HTTP or local backend |
| `grpo_scratch.py` | the GRPO update, written out by hand |
| `plots.py` | curves, drawn against the priced baselines |
| [`MEMO.md`](MEMO.md) | what the run actually improved |
| `prompt_ab.py` | prompt-version A/B on identical cases |
| `probe_throughput.py` | hardware measurements |
| `data/` | frozen splits, `SHA256SUMS`, `DATA_CARD.md` |
| `runs/` | measurements and evaluation results |
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
