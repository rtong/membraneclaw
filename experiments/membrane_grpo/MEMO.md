# What My First LLM RL Run Actually Improved

**Scope.** GRPO on a structured membrane-troubleshooting task with a
deterministic reward, on one RTX 5070 Ti. Seven 200-step runs: two on
Qwen2.5-0.5B-Instruct, five on Qwen3-1.7B. Each took about an hour. Two of the
1.7B runs are a seed replication, and they are the reason Part 2 retracts its
own conclusion. Part 3 is three evaluations of the *frozen* policy at 83 seconds
each, and they are worth more than any of the runs.

---

## The short answer, in three parts

**On the 0.5B: formatting, and nothing else.** Held-out reward rose 3.2x while
the model's ability to diagnose the fault did not move at all.

**On the 1.7B: the diagnosis moved — and my explanation of why did not
survive a second seed.** Under one seed, a reward weighted toward the upstream
bottleneck beat one weighted toward the outcome by 3.4x, at p < 1e-4. Under a
second seed the same comparison gives +0.025 at p = 0.30, and the seed-to-seed
spread *within* the winning configuration is larger than the gap between
configurations.

**On the task: the wall I spent five runs pushing against was the wrong one.**
Handing the frozen model perfect arithmetic -- the ceiling of any calculator,
tool or solver -- moves held-out diagnosis by +0.035 and does not reach
significance. Hand it perfect flags too, leaving nothing but a 17-row lookup,
and it answers three of the seven causes and never once emits the other four.

The first conclusion was not wrong; it was a statement about a model that could
not do the task at all, and I had mistaken it for a statement about RL. The
second was wrong, and Part 2 now says so. The third is the only claim here that
does not rest on a seed.

---

## Part 1 — Qwen2.5-0.5B: reward without capability

| held-out, greedy | frozen | after 200 steps |
| --- | --- | --- |
| reward | 0.086 | **0.279** |
| schema valid | 0.005 | **1.000** |
| `flags` correct | 0.005 | **0.562** |
| **`root_cause` correct** | **0.145** | **0.145** |
| `numeric` correct | 0.000 | 0.000 |
| exact match | 0.000 | 0.000 |

`root_cause` read **0.145 at all nine evaluation points**, identical to three
decimals across 200 steps, against a 1/7 = 0.143 chance floor. Decomposed
against the weights, the entire +0.193 is `format` (52%), `flags` vocabulary
(35%) and `stage` — a field the prompt states outright (16%). `numeric` and
`action` went slightly *down*.

Two reference lines make this readable. `baselines.constant` — valid JSON, the
same guess every time, never reading the record — scores **0.245**. The trained
policy reached 0.279. It spent an hour of a 5070 Ti to get 0.034 past a strategy
that ignores its input.

A second run, identical but for the reward weights, reported the same policy as
a **19x** improvement instead of 3.2x. Behaviourally the two were
indistinguishable on every axis measured.

The mechanism was clear: `root_cause` carries the largest weight (0.45) and
requires computing three percent changes, thresholding them, and reading a
table. `numeric` accuracy was 0.000, so that 0.45 was unreachable, and gradient
ascent took the reachable 0.25 instead. Not adversarial reward hacking — the
reward is a faithful description of a good answer. The policy maximised the part
within reach, and the part within reach did not matter.

pass@8 was 0.000 across 1,600 samples before training and 1,600 after. **RL
sharpens what a policy can already sometimes do, and this policy could never
once do it.**

---

## Part 2 — Qwen3-1.7B: the diagnosis moves, but not for the reason I gave

The obvious next question is what happens with a model that *can* partly do the
task. Three candidates, measured on the same 200 dev cases rather than argued
about:

| | reward | cause | numeric | schema |
| --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen2.5-1.5B-Instruct | 0.209 | 0.215 | 0.000 | 0.000 |
| **Qwen3-1.7B** | **0.315** | **0.255** | **0.038** | **0.970** |

Three unrelated failure modes: Math-1.5B writes 638 tokens of arithmetic and
never emits a JSON object at all, trading instruction-following for exactly the
capability I wanted; Qwen2.5-1.5B returns the numeric fields as strings, 600
times over 200 cases. Qwen3-1.7B was chosen less for its score than because
**its schema validity starts at 0.970**, leaving at most 0.003 of the headroom
that confounded Part 1 — so a reward rise here cannot be format learning.

### Three runs, one variable

Held-out `root_cause`, greedy, tested with McNemar's exact test on the same 200
paired cases:

| | `numeric` wt | `cause` wt | cause | vs base |
| --- | --- | --- | --- | --- |
| base | | | 0.255 | |
| MAIN | 0.15 | 0.45 | 0.295 | p = 0.057, **not significant** |
| **ABLATE** | **0.35** | **0.25** | **0.430** | **p < 1e-4** |
| PROBE | 0.35 | 0.10 | 0.450 | p < 1e-4 |

**The control inverted its own purpose.** PROBE exists to demonstrate that a
rising reward need not mean a better policy: I moved weight *away* from
`root_cause`, 0.45 → 0.10, expecting the reward to climb while the diagnosis
stayed put. It produced the best diagnosis of the three — +0.195 against MAIN's
+0.040, which is not even significant.

**ABLATE was built to say why.** PROBE differs from MAIN on four components at
once, so "raising `numeric`" and "lowering `root_cause`" were both live. ABLATE
isolates the first — `numeric` at 0.35, `root_cause` still substantial at 0.25 —
and it tracked PROBE, not MAIN: +0.135 against MAIN, 30 discordant pairs to 3,
p < 1e-4.

The reading at the time: the task is a chain, `numeric → flags → root_cause`,
and weighting its head beats weighting its end, because the end is only
reachable through numbers the model mostly gets wrong. I called it the one
finding I expected to transfer, and noted it rested on a single seed. The seed
took it apart below; Part 3 takes apart the chain it was built on.

### The second seed says no

Re-running MAIN and ABLATE at seed 42, everything else identical:

| held-out `cause` | seed 0 | seed 42 |
| --- | --- | --- |
| MAIN | 0.295 | 0.315 |
| ABLATE | **0.430** | **0.340** |
| ABLATE − MAIN | **+0.135**, p < 1e-4 | **+0.025**, p = 0.30 |

The effect does not replicate. Worse, the seed moves ABLATE more than the
weighting does: ABLATE at seed 0 against ABLATE at seed 42 is −0.090 with 21
discordant pairs against 3, **p = 0.0003** — a larger and better-supported
difference than the one I had attributed to the reward design.

MAIN is stable across seeds (+0.020, 8 discordant pairs, p = 0.29). The
instability is specific to the high-`numeric` configuration, which is
consistent with it being the configuration whose gradient depends on a
component the model is barely able to move.

**Where the mistake was.** McNemar gave p < 1e-4 at seed 0 and I read that as
"the effect is real". But the test asks whether *these two policies* differ on
*these 200 cases* — it says nothing about whether the weighting reliably
produces such policies. The first is a claim about two artifacts; the second is
the claim I actually made. Only repetition supports the second, and a small
p-value on a single run cannot substitute for it.

---

## Part 3 — Qwen3-1.7B: the wall is not where I said it was

Both parts above rest on a premise neither of them examined: that `root_cause`
is hard because it sits at the end of a chain — `numeric → flags → cause` —
whose head this model cannot compute. `numeric_acc` starts at 0.038 and peaks at
0.137 across every run in this memo, while an untrained 9B reaches 0.797, so the
premise looked safe. It had never been tested, because every test of it was
another training run, and a training run changes the policy rather than the
chain.

Three greedy evaluations of the **frozen** policy test it directly, by handing
over one link at a time. No training, 83 seconds each, same 200 dev cases, same
`eval.py` path, same answer key — the injected values come from
`truth_from_record`, so they are derived from the record rather than copied out
of the label.

| frozen, dev, greedy | baseline | + `numeric` given | + `flags` too |
| --- | --- | --- | --- |
| `numeric_acc` | 0.038 | 1.000 | 1.000 |
| `flags_acc`, per field | 0.330 | 0.580 | 1.000 |
| all three flags at once | 4/200 | 26/200 | 200/200 |
| **`cause_acc`** | **0.255** | **0.290** | **0.415** |
| `action_acc` | 0.195 | 0.205 | 0.225 |
| exact match | 0.000 | 0.090 | 0.210 |

McNemar's exact test on the same 200 paired cases:

| | difference | gained | lost | p |
| --- | --- | --- | --- | --- |
| baseline → `numeric` | +0.035 | 9 | 2 | 0.0654, **not significant** |
| `numeric` → `flags` | +0.125 | 25 | 0 | **< 1e-4** |
| baseline → `flags` | +0.160 | 34 | 2 | **< 1e-4** |

**Perfect arithmetic is worth +0.035 and does not clear significance.** That is
the ceiling of any calculator, tool or solver wired into this task — measured
rather than argued — and it closes a question I had been treating as open.

**The chain is real link by link and does not transmit.** `numeric → flags`
holds: all-three-flags goes 4 → 26, p = 0.0001. `flags → cause` holds: +0.125.
The composite does not, because `cause` needs all three flags *at once*. Perfect
arithmetic lifts per-field flag accuracy 0.330 → 0.580 and lifts the three-way
conjunction only 0.02 → 0.13.

**And the bottleneck was never arithmetic.** With numbers and flags both handed
over, the 17-row lookup is the entire remaining task, and:

| | true | emitted | correct |
| --- | --- | --- | --- |
| `scaling` | 29 | 79 | **29/29** |
| `biofouling` | 29 | 90 | **29/29** |
| `colloidal_fouling` | 29 | 31 | 25/29 |
| `organic_fouling` | 29 | **0** | 0/29 |
| `compaction` | 28 | **0** | 0/28 |
| `oxidation_damage` | 28 | **0** | 0/28 |
| `mechanical_leak` | 28 | **0** | 0/28 |

This is not a lookup performed badly. Four of the seven labels are not in the
model's output vocabulary at all, and the three that are happen to be exactly
the three rows requiring `dp = up`. Of the 0.745 between the frozen policy and a
perfect one, **0.585 — 78% — survives handing over everything upstream.**

`action` has the same shape and one failure of its own. It emits four of eight
labels and not the four its own causes imply — it names `colloidal_fouling` 31
times but that cause's action 5, reaching for `compaction`'s 26 times instead —
so the pair is not coming off the table together. And the severity override, one
stated conditional covering 37 of the 200 cases, is applied **zero** times in
all three conditions, including the one that hands it the flow percentage;
`isolate_and_evaluate_replacement` is emitted 0 times in 200. Where the cause is
right and the override does not apply, the lookup is perfect: 39/39 and 41/41.

Two controls. `numeric_acc` is exactly 1.000 over 200 cases and three fields, so
none of this is a failure to transcribe. And the gain is not the schema artifact
it could have been: none of the nine cases that flipped to a correct cause were
among the baseline's six `no_json`, and restricting to the 194 it parsed leaves
the same picture, 0.263 → 0.299 → 0.428.

What this costs Part 2 is its interpretation, not its numbers. `root_cause` was
never a measurement of whether the model could read the table. On four rows of
seven it was measuring whether a label the model never produces happened to come
out right, and the answer was always no.

---

## Two things I got wrong along the way

**I used a monotonicity argument as a significance test.** MAIN's `cause` rose
at nine consecutive evaluation points, and I called the gain real on that basis.
Monotonicity across nine points of an noisy series is not a test; McNemar says
p = 0.057. The commit that introduced the paired test also retracts the claim.

**I assumed pairing always helps.** It does not. On `schema` the discordant
pairs were 6:2 and the paired test came out *more* conservative than the
unpaired reading. Pairing helps when two policies agree on most cases; where the
disagreements are themselves balanced, it is the stricter test. `paired_test.py`
documents that rather than quietly reporting whichever number is smaller.

Also worth recording: my pre-registered go/no-go criterion was `pass@8 > 0`, and
it rejected all three 1.7B candidates. It was the wrong criterion — exact match
is a conjunction over seven fields, and even the 9B only reaches 0.245. What
actually distinguished the 0.5B's failure was `cause` sitting *exactly* at
chance, leaving nothing partial to sharpen.

---

## What I would do differently

**Log the held-out curve from step one.** The first training loop recorded
reward only. Reward alone cannot distinguish learning from collapse — in a
sibling actor-critic experiment on this task, a run held a *training* reward of
1.000 for 115 steps while its completion length fell 106 → 35 tokens and its
held-out reward was 0.000. The length and entropy panels showed it 85 steps
before the reward panel did.

**Price the cheats before training, not after.** Knowing that a constant guesser
scores 0.245 is what turned "reward tripled" into "reward is 0.034 past a
strategy that ignores the input". Without that line, +224% reads as a triumph.

**Do not read a component's weight as its influence — and do not read the
cascade as capability either.** One wrong number costs 0.69 of a possible 1.0
through the `numeric → flags → root_cause → action` cascade, so `numeric`'s
nominal 0.15 understates its weight in the *reward* roughly fourfold. That much
was measured and pinned by a test before any training ran, and it is still true.
What I then did with it was assume the cascade ran the other way too — that
fixing the head would move the tail. Part 3 handed the model a perfect head and
the tail moved +0.035, not significantly. An accounting identity about how a
reward decomposes is not a causal claim about what a policy can learn, and I
spent five runs treating it as one.

**Check the task before blaming the model.** Pointing a 9B at the task first
found a real ambiguity in the `dp` field; fixing it moved that field from 0.025
to 1.000. Had I frozen a baseline first, that defect would have been measured
into every number afterwards.

---

## Limits

**Two seeds, and they disagree.** This design cannot separate a reward-weighting
effect from seed noise at n=2, and would need perhaps five seeds per
configuration to try. PROBE has been run once, so everything said about it
carries the caveat that just cost ABLATE its conclusion. Part 3 is the
exception, and that is why it is the strongest section here: a frozen policy
under greedy decoding is deterministic — repeating the three evaluations
reproduces them to three decimals — so there is no seed for it to fail.

**200 steps is not convergence.** All three 1.7B curves were still climbing at
step 200; the endpoints are a snapshot, not a ceiling.

**Nothing trained here reaches exact match.** EM is 0.000 for every 0.5B and
1.7B policy measured, trained or not; the only non-zero figures in this memo are
Part 3's, where four of the seven fields are handed over. The 9B reaches EM
0.245 and `numeric` 0.797 unaided, so the task is solvable — but Part 3 is the
reason I no longer describe what stops the 1.7B as an arithmetic wall.

**Everything is synthetic**, every parameter hand-chosen; see `data/DATA_CARD.md`.
Not a diagnostic tool for a real membrane train.

One reproducibility caveat: two greedy evaluations of the same frozen policy
differing only in **batch size** returned `flags` 0.005 against 0.007, because a
batch is left-padded to its longest member. Batch composition held fixed, the
same evaluation is exact. Every figure above comes from committed artifacts in
`runs/`.

---

The three claims I will stand behind:

**On the 0.5B** — a 3.2x rise in reward corresponded to zero improvement in what
the reward was written to measure, and the same policy could be reported as 19x
better by changing the weights.

**On the 1.7B** — the diagnosis did move: every trained policy beats the
frozen 0.255, and MAIN reaches 0.315 at seed 42 against a 0.143 chance floor.
What I cannot claim is *why*. The upstream-weighting explanation held at one
seed and vanished at the next, and within the configuration that produced it
the seed matters more than the weighting does.

**On the task** — the arithmetic I built the whole experiment around is not what
gates the diagnosis. Perfect numbers buy +0.035, not significant; perfect
numbers and perfect flags still leave 78% of the gap, because four of the seven
causes and four of the eight actions are never emitted at all, and one stated
conditional is never applied in 37 chances. Five hours of training were spent
optimising through a bottleneck that three minutes of evaluation would have
found. **Measure what the ceiling is before spending the GPU trying to reach
it.**
