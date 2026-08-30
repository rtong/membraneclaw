# smoke_test — actor-critic PPO, as the counterpart to GRPO

The same task, the same frozen data, the same deterministic reward and the same model as
[`../../membrane_grpo`](../../membrane_grpo). One thing is different: the baseline is a
**learned value function** rather than the mean of a group, and the advantage is assigned
**per token** rather than per sequence.

| | GRPO (`membrane_grpo/grpo_scratch.py`) | actor-critic PPO (here) |
| --- | --- | --- |
| baseline | mean reward of `G` samples of one prompt | learned `V(s_t)` |
| advantage | one scalar per sequence | one per token, via GAE |
| rollouts per prompt | 8 | 1 |
| degenerate case | all `G` rewards equal → gradient exactly zero, 16% of groups | none |
| extra parameters | 0 | `hidden_size + 1` (2049 on Qwen3-1.7B) |

Nothing about the task is redefined here. `reward.py`, `task/prompt.py`, `data/*.jsonl`,
`build_mask` and — since the model change — `eval.py`'s `generate_hf` / `summarise` /
`supports_thinking_toggle` are **imported** from `membrane_grpo`, not copied. Two methods
measured with one ruler, or the comparison means nothing.

## The model: Qwen3-1.7B

Chosen on the GRPO side by measurement rather than argument (`membrane_grpo/runs/sel-*`),
and the reasons carry over unchanged:

| frozen, dev, greedy | Qwen2.5-0.5B | **Qwen3-1.7B** |
| --- | --- | --- |
| schema valid | 0.005 | **0.970** |
| `cause_acc` (chance = 0.143) | 0.145 | **0.255** |
| `numeric_acc` | 0.000 | 0.038 |

The 0.5B result was confounded — 52% of its reward gain was schema validity going
0.005 → 1.000, which is formatting, not diagnosis. Here that headroom is worth at most
0.003, so a reward rise on this model cannot be explained away as format learning. And its
`cause_acc` sat *exactly* at chance, leaving RL nothing partial to sharpen.

The 0.5B runs stay in `runs/` and stay reproducible: `FROZEN_BASELINE` and
`value_init_bias_for()` keep both models' numbers, and `--model Qwen/Qwen2.5-0.5B-Instruct`
still resolves the right value-head initialisation.

## Layout

| | |
| --- | --- |
| `ppo_ac.py` | the implementation: GAE, both clipped losses, the value head, the loop. The gradient is written out by hand in the module docstring, as `grpo_scratch.py` does. |
| `01_actor_critic_and_gae.ipynb` | the derivation, and 14 checks that need no GPU. Runs on CPU in about two minutes. Committed with outputs. |
| `02_ppo_actor_critic_1.7b.ipynb` | the run on the card, and the curves. Executed, committed with outputs; re-running it against a finished run directory plots instead of retraining. |
| `runs/ppo-qwen3-17b-main-s0/` | the 200-step run: `metrics.jsonl`, `eval.jsonl`, `curves.png`, `critic.png`. |
| `runs/smoke-ppo-ac-cpu/`, `runs/ppo-ac-ie2-s0/`, `runs/gonogo-stage-s0*/` | the 0.5B runs. Kept as evidence, not as a baseline for the new model. |

## Running

```sh
cd experiments/notebooks/smoke_test

# the checks, no GPU needed
../../membrane_grpo/.venv/bin/python -m nbclient 01_actor_critic_and_gae.ipynb

# the run
../../membrane_grpo/.venv/bin/python ppo_ac.py --steps 200 --prompts-per-step 8
```

The venv is `membrane_grpo`'s: torch 2.13.0+cu130, `sm_120`, transformers 5.15.0. There is
no second environment to keep in step.

## What came across from GRPO with the model

Four things, none of them cosmetic.

**`enable_thinking=False`, and it is fatal without it.** A Qwen3 chat template defaults
thinking *on*. Measured on the 9B: the model spends its entire budget inside `<think>` and
returns empty content, so every case fails the reward gate. Training against that optimises
a policy for a format we never sample in. Detected from the template rather than switched
on by model name (`eval.supports_thinking_toggle`), because getting it wrong silently
produces a baseline that looks like a capability floor and is a formatting artefact.

**A held-out curve during training** (`eval_every=25`, the full 200-case dev split, through
`eval.py`'s own `generate_hf`). The reason to have it is on *this* side of the fence:
`runs/gonogo-stage-s0` held a training reward of 1.0000 for thirty steps while the policy
it was checkpointing was worth 0.000 on dev, and `adapter-best` saved the wreck. A reward
curve cannot tell learning from collapse. Using `eval.py`'s function rather than a
reimplementation is what makes step 0 comparable to the frozen baseline and to GRPO.

**The `ABLATE` weight set.** On GRPO it settled which weight was moving the diagnosis:
holding `numeric` at 0.35 while leaving `root_cause` at 0.25 behaved like `PROBE`, not like
`MAIN` — held-out `cause_acc` 0.430 against 0.295, McNemar p < 1e-4. The reading is that
the task is a chain, `numeric → flags → cause`, and weighting its head beats weighting its
tail. Whether that survives a change of algorithm is untested and cheap to test.

**`paired_test.py`.** Aggregates force an independent-sample comparison; on 200 cases the
SE on a proportion near 0.26 is 0.031, wide enough to leave a real effect looking like
noise. The two evaluations are the *same* 200 cases, so McNemar is the test the design
called for. It needs the `per_case` block `eval.py` now writes — run `eval.py` against a
saved adapter, not the mid-run `eval.jsonl`, which deliberately stores scalars only.

## Three things the smoke tests caught

**The critic's learning rate is a derived quantity, and it does not port between models.**
Adam moves each of a linear head's `d` weights by about `lr` per step and those steps sum
through the features, so `|dV| ≈ lr * ||h||_1`. The first CPU run used `value_lr=1e-3` and
`V` went `0.086 → -1.868` in one step, gradient norm 10 → 239. What the model change added:

| | hidden | `E|h|` | `||h||_1` | `|dV|` at `lr=5e-6` |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 896 | 4.824 | 4323 | 0.0216 |
| Qwen3-1.7B | 2048 | 1.127 | **2308** | 0.0115 |

2.3× the dimensions and a 47% *smaller* `||h||_1` — Qwen3 does not carry the handful of
enormous outlier features the 0.5B does. Porting `5e-6` unchanged would have quietly halved
the critic's step. The default is now `8.5e-6`, which holds it at the ~0.021 the old
default was chosen for, and `01` re-measures both models rather than asserting either.

**Left padding is a seam GRPO never touches.** A GRPO group is `G` samples of one prompt,
so every sequence has the same length and there is no padding. Taking one sample each from
eight different prompts left-pads the batch, and a left-padded batch through a plain
`model(input_ids=...)` attends to the pad tokens and starts RoPE at the pad. It raises no
error; it just corrupts the log-probabilities, which are the entire training signal.
Measured in `01`: **9.8 nats per token** on the padded row, and exact on the unpadded one —
invisible to any test whose batch happens to be uniform.

**The critic on the 0.5B was not a critic, and `value_mae` could not say so.** Over
`runs/ppo-ac-ie2-s0`, the learned value function's explained variance was **+0.07** — its
mean absolute error, 0.0886, was *worse* than predicting the constant mean of the returns,
0.0797. Its correlation with the current target was 0.29 but with a 20-step trailing mean
of the target it was 0.69: a low-pass filter of the reward history, not a function of the
state. On the 125 steps of `gonogo-stage-s0` where every sequence in the batch drew the
identical reward, V had mean 0.95 and a within-batch spread of 0.065.

The cause is rate, not wiring: 200 steps × 2 inner epochs at `lr=5e-6` bounds the head's
weight travel at 2.0e-3 per coordinate, and the saved head's `max|w|` is 1.49e-3 — 75% of
the way to a bound it should never have been near. `value_ev` and `value_std` are now in
`metrics.jsonl` so this is a number rather than an argument. **Nothing has been done about
it yet** beyond re-deriving `value_lr` for the new model; the fixes on the table are more
critic steps per rollout batch (the head is 2049 parameters on cached detached hidden
states, so 20–50 Adam steps cost nothing), normalising the head's input, and raising the
target `|dV|` above 0.02 — the regression target's own batch-to-batch swing reaches 1.0.

## The card has to be free first

`anton` has one RTX 5070 Ti at 16 GiB and the vLLM 9B service holds about 14.6 GiB of it.
Stopping it needs `sudo`, and this machine has no `NOPASSWD` rule — so it is done by a
person at a terminal, not by a notebook and not by an agent:

```sh
sudo systemctl stop vllm-qwen membraneclaw-agent    # any branch
deploy/stack.sh down                                 # branches that have it
```

Notebook 02's first cell refuses to run until that has happened, rather than letting a run
OOM twenty minutes in. Its threshold is 11 GiB now rather than 8: Qwen3-1.7B is 3.4 GiB of
bf16 weights before any activation, and the value head forces `output_hidden_states=True`,
which keeps all 29 layers' hidden states alive through the backward pass.

**Do not trust `torch.cuda.mem_get_info` on this machine.** With vLLM loaded it reported
13,454 MiB free while `nvidia-smi` reported 544 — under WSL2 the driver counts memory it
could page to host RAM as available to a new context. The guard uses `nvidia-smi` plus
`systemctl is-active`, and that pairing is the only one that has matched reality here.

## The runs on this model

`runs/ppo-qwen3-17b-main-s0` — 200 steps, MAIN weights, seed 0, 8 sequences per step, about
35 minutes on the card. Held out on the full 200-case dev split, greedy, through `eval.py`:

| held out | step 0 | step 200 | |
| --- | --- | --- | --- |
| reward | 0.3015 | 0.3273 | +0.026 |
| `cause_acc` | 0.2350 | 0.2900 | **+0.055** |
| `flags_acc` | 0.3233 | 0.2700 | **−0.053** |
| `numeric_acc` | 0.0350 | 0.0283 | −0.007 |
| `action_acc` | 0.1750 | 0.2050 | +0.030 |
| schema valid | 0.965 | 0.995 | +0.030 |
| exact match | 0.000 | 0.000 | — |

Step 0 reproduces the frozen baseline exactly, which is the check that the training loop's
evaluation and the frozen measurement are the same measurement.

`cause_acc` +0.055 cannot be read off these aggregates — the independent-sample SE on a
proportion near 0.26 over 200 cases is 0.031 — but the paired test below says the effect is
real: 13 cases gained against 2 lost, p = 0.0074. What the aggregates do show plainly is
that **`flags_acc` fell by about as much as `cause_acc` rose**, which is not what a clean
improvement looks like.

**The critic did not work, and the new panels say so plainly.** Median per-step
`value_ev` was **+0.005** and the run-level explained variance was **−0.000**: the learned
value function did exactly as well as predicting the mean of the returns, for 200 steps.
`value_std` averaged 0.0123 against a `value_mean` near 0.32 — a 3.8% relative spread, so
every token in a batch received effectively the same baseline. At λ=1 that makes
`A_t = R − const`, which is REINFORCE with a slow moving average. Whatever moved the policy
above, the critic was not it.

### The ablation: `ABLATE` does not reproduce here

`runs/ppo-qwen3-17b-ablate-s0` — identical to the run above in model, data, seed and every
hyperparameter. One thing differs: `numeric` is weighted 0.35 instead of 0.15, and
`root_cause` 0.25 instead of 0.45. The hypothesis is that the task is a chain,
`numeric → flags → cause`, so weighting the head beats weighting the tail.

Only the component accuracies are comparable between the two — `reward` is a weighted sum
and the two runs use different weights, so the frozen policy scores 0.3015 under `MAIN` and
0.2615 under `ABLATE` without changing at all. Both runs report identical step-0
accuracies, which is the check that this is the same frozen policy on the same cases.

| Δ over 200 steps | `cause_acc` | `numeric_acc` | `flags_acc` | `schema_ok` |
| --- | --- | --- | --- | --- |
| MAIN (`numeric` 0.15) | +0.0550 → 0.2900 | −0.0067 | **−0.0533** → 0.2700 | 0.9950 |
| ABLATE (`numeric` 0.35) | +0.0500 → 0.2850 | −0.0017 | **−0.0050** → 0.3183 | 0.9850 |

**The `cause_acc` curves lie on top of each other** and both plateau by step 100. Raising
the weight on `numeric` by 2.3× did not move the diagnosis. The one real difference is that
`ABLATE` did not degrade `flags` the way `MAIN` did — it ended roughly where it started
while `MAIN` gave up 0.053.

`numeric_acc` itself is noise at this scale and should not be read as a trend either way:
200 cases × 3 fields is 600 binary events, so an accuracy near 0.03 is about 18 hits, and
the 0.021–0.045 band both curves wander through is what 18 hits looks like. Neither policy
learned any arithmetic, and exact match was 0.000 at every one of the eighteen evaluation
points across both runs.

### The paired test, which overturns the aggregate reading

The aggregates say neither `cause_acc` delta is significant: +0.055 and +0.050 against an
independent-sample SE of 0.031 on 200 cases, about 1.8 SE and less. **That is the wrong
test.** The evaluations are the same 200 cases in the same order, decoded greedily, so
almost every case is answered identically by both policies and carries no information about
the difference. Only the disagreements do. McNemar's exact test uses exactly them.

Three greedy evaluations through `eval.generate_hf`, written to `runs/paired/`, then
`paired_test.exact_binomial_two_sided` on the per-case `cause` outcomes:

| comparison | before | after | gained | lost | discordant | p |
| --- | --- | --- | --- | --- | --- | --- |
| frozen → MAIN | 0.235 | 0.290 | 13 | 2 | 15 | **0.0074** |
| frozen → ABLATE | 0.235 | 0.285 | 13 | 3 | 16 | **0.0213** |
| MAIN → ABLATE | 0.290 | 0.285 | 0 | 1 | **1** | 1.0000 |

**Both runs improved the diagnosis significantly.** The aggregate SE was hiding a real
effect, which is the entire reason the test exists.

And the ablation is settled harder than the curves alone could settle it: **MAIN and ABLATE
disagree on the root cause for exactly one case out of 200.** They are, on this metric, the
same policy. Raising `numeric` from 0.15 to 0.35 did not change what the model learned to
diagnose — it only spared `flags` the degradation `MAIN` suffered.

`runs/paired/*.json` carry the full `per_case` blocks, so this is re-checkable without
re-running anything.

### The critic, in both runs

`|ΔV|` came in at a median of 0.0033 against the 0.0196 the learning rate was derived for.
The `|ΔV| ≈ lr·‖h‖₁` formula assumes all `d` coordinates move coherently through the
features, but the gradient `2(V−G)·h` takes each coordinate's sign from `h`'s own signs, so
Adam's ±lr steps partly cancel in the dot product. **It is an upper bound, not an
estimate** — it overshoots by ~5× on both models, which is why `CHECK 12` is written as a
safety assertion (it did catch `lr=1e-3`) and should not be read as a prediction.

## Status

* `01` — executed on Qwen3-1.7B's constants, 14/14 checks pass, committed with outputs.
* `ppo_ac.py` — two 200-step runs end to end on the card, MAIN and ABLATE, exit 0 both.
* `02` — executed against both, committed with outputs. Its figures are drawn inline; the
  PNGs beside the runs are written by the same cells.

What is shown: the loop is correct on real hardware at this scale; both policies improved
held-out `root_cause` accuracy by a margin that survives a paired test (p = 0.007 and
p = 0.021); raising the weight on `numeric` changed nothing about that, the two trained
policies disagreeing on exactly one case in 200; and the critic contributed none of it,
with explained variance at zero for all 400 steps.

What is not shown: that this is arithmetic rather than a better prior over seven labels —
`numeric_acc` is ~18 hits out of 600 and exact match is 0.000 everywhere. Nor that any of it
survives a second seed, a second split, or the sealed `test.jsonl`. One seed is one seed.
