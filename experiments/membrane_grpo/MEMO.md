# What My First LLM RL Run Actually Improved

**Scope.** Two 200-step GRPO runs on Qwen2.5-0.5B-Instruct over a structured
membrane-troubleshooting task with a deterministic reward, on one RTX 5070 Ti.
This is a smoke test. Every number below is from a run that took under an hour,
and nothing here should be read as a claim about GRPO in general.

---

## The short answer

It improved JSON formatting and vocabulary compliance, and nothing else.

Held-out reward on the frozen dev split went **0.086 → 0.279**, a 3.2x rise. The
model's ability to diagnose the fault did not change at all:

| held-out, greedy | frozen | after 200 steps | |
| --- | --- | --- | --- |
| reward | 0.086 | **0.279** | +224% |
| schema valid | 0.005 | **1.000** | saturated |
| `flags` correct | 0.005 | **0.562** | +112x |
| **`root_cause` correct** | **0.145** | **0.145** | **unchanged** |
| `numeric` correct | 0.000 | 0.000 | unchanged |
| `action` correct | 0.130 | 0.110 | worse |
| exact match | 0.000 | 0.000 | unchanged |

`root_cause` read **0.145 at all nine evaluation points**, identical to three
decimals across 200 steps. A uniform guess over seven balanced labels scores
1/7 = 0.143. The policy began at chance and ended at chance.

Decomposing the +0.193 against the reward weights:

| component | contribution | share |
| --- | --- | --- |
| `format` — schema compliance | +0.0995 | **52%** |
| `flags` — right vocabulary | +0.0668 | **35%** |
| `stage` — copying a field the prompt states outright | +0.030 | **16%** |
| `action` | −0.003 | |
| `numeric` | 0.000 | |
| **`root_cause`** | **0.000** | **0%** |

What the policy learned, concretely: the base model wrote the *threshold text*
into the flag field — `"flow": "<-10"` — and copied threshold constants into the
numbers. After training it writes `"down"`, and copies the stage from the prompt
instead of inventing one. That is the whole of it.

---

## The same policy, two headline numbers

The second run is identical to the first in code, data, model, seed and
hyperparameters. One thing differs: the reward weights move away from
`root_cause` and toward `format` and `numeric`.

| | main | probe |
| --- | --- | --- |
| held-out reward | 0.086 → 0.279 | 0.024 → **0.463** |
| **reported as** | **3.2x** | **19x** |
| `root_cause` | 0.145 → 0.145 | 0.145 → 0.145 |
| `flags` | 0.005 → 0.562 | 0.005 → 0.523 |
| `numeric` | 0.000 → 0.000 | 0.000 → 0.000 |
| schema valid | 0.005 → 1.000 | 0.005 → 1.000 |
| exact match | 0.000 | 0.000 |
| completion length | 106 → 101 | 106 → 101 |

The two policies are behaviourally indistinguishable on every measured axis. The
headline improvement differs by **6x**, and the difference is entirely a property
of the weight vector I chose.

"Held-out reward improved 19x" would be a true sentence about the probe run. It
describes my spreadsheet, not the model.

---

## Reference lines, and why 0.279 is not good news

Before training, three strategies that do not solve the task were priced
(`baselines.py`):

| | reward |
| --- | --- |
| `constant` — valid JSON, same guess every time | **0.245** |
| `skip_correction` — everything right but the temperature correction | 0.890 |
| `oracle` | 1.000 |

The trained policy scores **0.279**. It has spent 200 steps and roughly an hour
of an RTX 5070 Ti to arrive just past a strategy that ignores the input
entirely. The frozen baseline at 0.086 was *below* that line, so the honest
description of the run is: it closed the gap to a constant guesser and went
0.034 past it.

Read against zero, +224% is a result. Read against 0.245, it is most of the way
to nothing. Building the reference lines before the run, rather than after, is
the single practice from this project I would keep unchanged.

---

## The five pre-registered predictions

Written before the first gradient step, and scored as they fell.

**1. Format saturation dominates the reward curve. — Hit.** Schema validity went
0.005 → 1.000 by step 50 and is the largest single contributor at 52% of the
gain. Reward plateaued at the same time.

**2. The probe control separates reward from capability. — Hit, decisively.**
See above. This was the prediction I was least sure could be demonstrated inside
one experiment rather than argued about; the 6x gap on indistinguishable
policies settles it.

**3. The hard tier will not move. — Not testable as stated.** I predicted that
any exact-match gain would come from `easy` cases. There was no exact-match gain
anywhere: EM was 0.000 at every evaluation of both runs. What can be said is
that the reward gain was tier-*independent*: easy 0.093 → 0.284 and hard
0.072 → 0.270, a rise of +0.191 against +0.197. That is itself confirmation
that nothing arithmetic was learned. A gain that does not
care whether the temperature correction is needed is not a gain in arithmetic.

**4. pass@1 rises while pass@8 is flat or falls, with diversity collapsing. —
Falsified, and instructively.** pass@8 was 0.000 before and after; both sit on
the floor, so the comparison is empty. Diversity did the opposite of collapsing
on one metric and collapsed hard on another:

| sampled, k=8 | frozen | trained |
| --- | --- | --- |
| unique answers per group | 5.54 / 8 | **7.97 / 8** |
| distinct-4 (lexical) | 0.727 | **0.256** |
| validity under sampling | 0.665 | **0.995** |

Both numbers are correct. The model converged onto a single rigid output
template — hence distinct-4 falling by two thirds — while the *values* it fills
in stayed noisy, which is why nearly every sample is now a distinct object. The
form collapsed; the content did not. **Which diversity metric I had chosen would
have determined which conclusion I reached**, and I had pre-registered only the
one that says diversity fell.

**5. A visible fraction of groups will have zero reward variance. — Falsified
during training.** True at the frozen baseline, where 16% of groups were
degenerate. During both runs `adv_zero_frac` was 0.00 at every step after the
first. Format variation under sampling supplied reward variance immediately, so
the degenerate-group risk I had built the whole reward around avoiding never
materialised once training started.

Two hits, two falsified, one unanswerable. The two falsified ones taught me more
than the two hits.

---

## Where reward and capability came apart, mechanically

The reward has six components. Five of them are cheap: schema compliance, flag
vocabulary, stage-copying. One of them — `root_cause`, at 0.45, the largest
single weight — requires actually computing three percent changes, thresholding
them, and reading a table.

A 0.5B model can learn the cheap five from a scalar reward in 50 steps. It
cannot learn the expensive one at all, and the run shows why: `numeric` accuracy
is 0.000. `root_cause` depends on flags, flags depend on the numbers, and the
numbers are never right. The 0.45 weight was unreachable, so gradient ascent did
what gradient ascent does and took the reachable 0.25.

This is not reward hacking in the adversarial sense. The reward function is a
faithful description of what a good answer looks like. The policy simply
maximised the part of it that was within reach, and the part within reach
happened to be the part that does not matter.

For context, Qwen3.5-9B under the same prompt reaches exact match 0.325 and
`numeric` around 0.86. The task is solvable. It is not solvable by this model,
and reinforcement learning on the final answer did not change that — RL
sharpens what a policy can already sometimes do, and this policy could never
once produce a correct answer: pass@8 was 0.000 across 1,600 samples before
training and 1,600 after. The fraction of sampled groups with no reward
variance was 0.160 both before and after — identical.

---

## What I would do differently

**Log the held-out curve from step one.** The first version of the training loop
recorded reward only. Reward alone cannot distinguish learning from collapse —
in a sibling experiment on this same task, a run held a *training* reward of
1.000 for 115 steps while its completion length fell from 106 tokens to 35 and
its held-out reward was 0.000. The length and entropy panels showed it 85 steps
before the reward panel did.

**Do not trust a component's weight as a measure of its influence.** One wrong
number costs 0.69 of a possible 1.0 through the cascade `numeric → flags →
root_cause → action`, so `numeric`'s nominal 0.15 understates it by 4x. This was
measured and pinned by a test before the run, and it is why `cause_given_flags`
exists as a separate diagnostic.

**Give the model somewhere to put its working.** The first prompt forbade any
text besides the JSON object, and under it even the 9B did not compute — it
emitted well-formed JSON carrying numbers it had estimated. Allowing terse
arithmetic before the answer took the 9B's numeric accuracy from 0.08 to 0.58.
The 0.5B ignored the invitation entirely and went straight to JSON at 108
tokens, which is worth knowing: prompt affordances only help a model that can
use them.

**Check the task before blaming the model.** Pointing the 9B at the task first
found a genuine ambiguity — the `dp` field could reasonably be read as the train
total or the anomalous stage's own, and the generator put the whole change on
that stage. Fixing it moved that field from 0.025 to 1.000. Had I frozen a
baseline first, I would have measured that defect into every number afterwards.

---

## Limits

200 steps, one seed, one model, one task, LoRA rank 16, no learning-rate sweep.
`root_cause` staying at exactly 0.145 is strong evidence that nothing was
learned about diagnosis, but it is not evidence that nothing *could* be — a
longer run, a larger model, or a curriculum that rewards the arithmetic directly
might all move it. The data is synthetic and every parameter in it was chosen by
hand; see `data/DATA_CARD.md`. This is not a diagnostic tool for a real membrane
train and was never meant to be.

One reproducibility caveat worth stating: two greedy evaluations of the same
frozen policy, differing only in batch size (64 against 32), returned `flags`
0.005 against 0.007. Greedy decoding is deterministic per sequence but a batch
is left-padded to its longest member, so batch composition perturbs the result
slightly. Every figure above is quoted from the committed artifacts in `runs/`
rather than from the training loop's own step-0 evaluation, which used the
other batch size.

The claim I will stand behind is narrow: **on this task, at this scale, a 3.2x
improvement in reward corresponded to zero improvement in the thing the reward
was written to measure, and I could make the same policy look 19x better by
changing the weights.**
