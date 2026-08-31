# What My First LLM RL Run Actually Improved

**Scope.** GRPO on a structured membrane-troubleshooting task with a
deterministic reward, on one RTX 5070 Ti. Seven 200-step runs: two on
Qwen2.5-0.5B-Instruct, five on Qwen3-1.7B. Each took about an hour. Two of the
1.7B runs are a seed replication, and they are the reason Part 2 retracts its
own conclusion.

---

## The short answer, in two parts

**On the 0.5B: formatting, and nothing else.** Held-out reward rose 3.2x while
the model's ability to diagnose the fault did not move at all.

**On the 1.7B: the diagnosis moved — and my explanation of why did not
survive a second seed.** Under one seed, a reward weighted toward the upstream
bottleneck beat one weighted toward the outcome by 3.4x, at p < 1e-4. Under a
second seed the same comparison gives +0.025 at p = 0.30, and the seed-to-seed
spread *within* the winning configuration is larger than the gap between
configurations.

The first conclusion was not wrong; it was a statement about a model that could
not do the task at all, and I had mistaken it for a statement about RL. The
second was wrong, and Part 2 now says so.

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

Three unrelated failure modes. Math-1.5B writes 638 tokens of arithmetic and
never emits a JSON object — 200/200 `no_json` — trading instruction-following
for exactly the capability I wanted. Qwen2.5-1.5B has one mechanical defect: the
numeric fields come out as strings, so `not_a_number` fires exactly 600 times
over 200 cases.

Qwen3-1.7B was chosen less for its score than because **its schema validity
starts at 0.970**. The 0.5B experiment was confounded — half its reward gain was
schema going 0.005 → 1.000. Here that headroom is worth at most 0.003, so a
reward rise cannot be explained away as format learning.

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

**ABLATE says why.** PROBE differs from MAIN on four components at once, so both
"raising `numeric`" and "lowering `root_cause`" were live explanations. ABLATE
raises `numeric` to 0.35 and leaves `root_cause` substantial at 0.25 — and it
tracks PROBE, not MAIN. Directly: MAIN-trained → ABLATE-trained is +0.135, with
30 discordant pairs against 3, p < 1e-4.

The reading at the time: the task is a chain, `numeric → flags → root_cause`,
and weighting the head of the chain beats weighting the end, because the end is
only reachable through numbers the model mostly gets wrong. I called it the one
finding I expected to transfer, and noted it rested on a single seed.

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

**Do not read a component's weight as its influence.** One wrong number costs
0.69 of a possible 1.0 through the `numeric → flags → root_cause → action`
cascade, so `numeric`'s nominal 0.15 understates it roughly fourfold. That
cascade turned out to be the whole story of Part 2, and it was measured and
pinned by a test before any training ran.

**Check the task before blaming the model.** Pointing a 9B at the task first
found a real ambiguity in the `dp` field; fixing it moved that field from 0.025
to 1.000. Had I frozen a baseline first, that defect would have been measured
into every number afterwards.

---

## Limits

**Two seeds, and they disagree.** The one claim I thought would transfer did
not survive its own replication. Two is still far too few — the honest position
is that this design cannot separate a reward-weighting effect from seed noise
at n=2, and would need perhaps five seeds per configuration to try. PROBE has
still only been run once, so everything said about it above carries the same
caveat that just cost ABLATE its conclusion.

**200 steps is not convergence.** All three 1.7B curves were still climbing at
step 200, and the two `numeric`-heavy runs only began their rise at step
150–175. The endpoints are a snapshot, not a ceiling.

**Nothing here reaches exact match.** EM is 0.000 for every 0.5B and 1.7B policy
measured, trained or not. `numeric` peaks at 0.137. The 9B reaches EM 0.245 and
`numeric` 0.797, so the task is solvable — the arithmetic wall simply sits above
1.7B.

**Everything is synthetic**, every parameter hand-chosen; see `data/DATA_CARD.md`.
Not a diagnostic tool for a real membrane train.

One reproducibility caveat: two greedy evaluations of the same frozen policy
differing only in batch size returned `flags` 0.005 against 0.007, because a
batch is left-padded to its longest member and batch composition perturbs the
result. Every figure above comes from committed artifacts in `runs/`.

---

The two claims I will stand behind:

**On the 0.5B** — a 3.2x rise in reward corresponded to zero improvement in what
the reward was written to measure, and the same policy could be reported as 19x
better by changing the weights.

**On the 1.7B** — the diagnosis did move: every trained policy beats the
frozen 0.255, and MAIN reaches 0.315 at seed 42 against a 0.143 chance floor.
What I cannot claim is *why*. The upstream-weighting explanation held at one
seed and vanished at the next, and within the configuration that produced it
the seed matters more than the weighting does.
