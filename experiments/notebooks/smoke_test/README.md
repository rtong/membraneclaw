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
| `03_critic_and_trust_region.ipynb` | the two items this README's *Status* left open — the critic's representation (`--no-value-detach`) and the trust region (`inner_epochs` vs. an entropy bonus). Three runs, each notebook 02's MAIN config with one knob changed. Same plot-or-retrain guard as 02. |
| `runs/ppo-qwen3-17b-{main,ablate}-s0/` | the two 200-step runs. Every run directory carries the same four artefacts: `metrics.jsonl`, `eval.jsonl`, `curves.png`, `critic.png` — notebook 02 iterates over `RUNS`, so adding a run to that dict is all it takes to get the same figures. |
| `runs/paired/` | the three greedy evaluations the paired test reads — frozen, MAIN, ABLATE — each with the full `per_case` block. |
| `runs/main_vs_ablate.png` | the cross-run comparison; it belongs to neither directory, so it sits one level up. |
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

Three 200-step runs, Qwen3-1.7B, seed 0, identical in everything but the reward
weight vector. Held out on the full 200-case dev split every 25 steps, greedy,
through `eval.py`'s own code path; all three reproduce the frozen baseline at
step 0 to four decimals.

### The headline: the weight on `root_cause` predicts the outcome, inversely

| | `root_cause` weight | `cause_acc` 0 → 200 | `flags_acc` | `action_acc` | McNemar vs frozen |
| --- | --- | --- | --- | --- | --- |
| `MAIN` | **0.45** | 0.235 → **0.170** (−0.065) | −0.012 | −0.015 | p = 0.171 (32 gained, 45 lost) |
| `ABLATE` | 0.25 | 0.235 → 0.245 (+0.010) | +0.075 | +0.040 | p = 0.918 (48 gained, 46 lost) |
| `PROBE` | **0.10** | 0.235 → **0.335** (+0.100) | **+0.353** | +0.095 | **p = 0.0055** (34 gained, 14 lost) |

Monotone, and the direction is the uncomfortable one: **the run that weighted the
diagnosis most heavily is the only one that got worse at diagnosing**, and it is
the only one that destabilised — `MAIN`'s held-out `cause_acc` fell to 0.135 at
step 125, below the 1/7 = 0.143 chance rate, alongside a 31% monotone drop in the
entropy proxy. `PROBE`, weighting it at 0.10, sailed through the same window and
peaked at 0.380.

The paired tests on the same 200 cases:

| | before | after | gained | lost | discordant | p |
| --- | --- | --- | --- | --- | --- | --- |
| frozen → MAIN | 0.235 | 0.170 | 32 | 45 | 77 | 0.171 |
| frozen → ABLATE | 0.235 | 0.245 | 48 | 46 | 94 | 0.918 |
| frozen → PROBE | 0.235 | 0.335 | 34 | 14 | 48 | **0.0055** |
| MAIN → ABLATE | 0.170 | 0.245 | 20 | 5 | 25 | **0.0041** |
| MAIN → PROBE | 0.170 | 0.335 | 60 | 27 | 87 | **0.0005** |
| ABLATE → PROBE | 0.245 | 0.335 | 61 | 43 | 104 | 0.095 |

Only `PROBE` beats the frozen policy significantly. `MAIN` is not significantly
*worse* either — 45 lost against 32 gained does not reach 0.05 — but it is
significantly worse than both of the others.

**`PROBE`'s reward number is not evidence of anything.** It reaches 0.518 against
`MAIN`'s 0.271, but the two are different weighted sums: `PROBE` puts 0.35 on
`format`, and this model's schema validity starts at 0.965, so a third of that
reward is collected before the first gradient step. Only the accuracies carry
across weight sets, and those are the table above.

Nothing learned arithmetic. `numeric_acc` moved −0.015 / +0.017 / −0.007 and
exact match was 0.000 at all 27 evaluation points across the three runs.

### What the four algorithm fixes actually did

`MAIN` is the clean before/after: same model, same weights, same seed, only the
algorithm differs (`runs/v1-before-fixes/` holds the earlier pair).

| | before | after |
| --- | --- | --- |
| `clip_frac` | **0.000000**, all 400 steps | 0.008 |
| `ratio_mean` | exactly 1.000000000, min and max | varies |
| per-token share of the advantage | **6.4%** | **46–66%** |
| `value_std` | 0.012 | 0.022–0.042 |
| held-out `cause_acc` at step 200 | 0.290 | 0.170 |

The first four rows are the fixes working: the trust region exists, and credit
assignment is genuinely per token for the first time. The last row is the result
being *worse*, and the explanation is in the third-from-last: `clip_frac` is
0.008, so the trust region is engaging on under 1% of tokens. Raising
`inner_epochs` from 1 to 4 therefore did not buy four restrained updates per
batch — it bought four unrestrained ones, and that is what drove the entropy
collapse. A clip that binds on 0.8% of tokens is a clip in name.

### The critic still does not work, and it is not the optimiser

| | `value_ev` (out of sample) | `value_ev_fit` (in sample) | undefined steps |
| --- | --- | --- | --- |
| `MAIN` | +0.009 | +0.014 | 28 / 200 |
| `ABLATE` | −0.003 | +0.015 | 12 / 200 |
| `PROBE` | −0.026 | −0.001 | 3 / 200 |

Eight dedicated gradient steps per batch on cached features, and the head still
cannot fit the batch it was just trained on — `value_ev_fit` is the in-sample
number and it is ~0.01. That rules out the learning rate and the update count as
the cause. What is left is the representation: under `value_detach=True` the
critic is a linear probe on the last layer's hidden states, and those apparently
do not linearly encode the return for this task. Testing that means
`--no-value-detach`, which is a separate run because it lets the value loss
reshape what the policy reads.

**`undefined steps` is a result, not a footnote.** Those are batches where every
sequence drew nearly the same reward, so the returns have no variance and
explained variance is undefined rather than bad. 14% of `MAIN`'s batches. That is
the same degeneracy GRPO's group-mean baseline hits, arriving here as a collapsed
signal-to-noise ratio instead of an exactly-zero gradient.

### Both of the above, tested in `03_critic_and_trust_region.ipynb`

Three runs, each notebook 02's `MAIN` configuration with exactly one knob moved,
seed 0, 200 steps. All three beat the frozen policy on the paired McNemar test
(p = 0.0013 / 0.0074 / 0.0165) **and** beat `MAIN`, which is confirmed the worst
of the four — the only run whose held-out `cause_acc` fell and the only one to
dip below the 1/7 chance rate.

| run | change vs. MAIN | result |
| --- | --- | --- |
| `runs/ppo-qwen3-17b-nodetach-s0` | `value_detach=False` | **weak yes for Q1.** Out-of-sample `value_ev` +0.009 → +0.016, degenerate batches 28/200 → 6/200 — the predicted direction, but still a critic explaining ~2% of the return. Representation is part of the constraint, not all of it. Held-out `cause_acc` 0.235 → 0.305, but confounded by the shared trunk. |
| `runs/ppo-qwen3-17b-ie2-s0` | `inner_epochs` 4 → 2 | entropy proxy 0.169 → 0.141 (vs `MAIN`'s 0.073), `cause_acc` 0.235 → 0.290 — but by engaging the clip *less* (`clip_frac` 0.008 → 0.003). The gain is from fewer unrestrained updates; `inner_epochs=4` was too many. |
| `runs/ppo-qwen3-17b-ent-s0` | `entropy_coef` 0 → 0.005 | **the winner.** The only run where the trust region bound at all — `ratio_mean` max **1.405**. Best held-out result of the series: `cause_acc` 0.235 → 0.350, `flags_acc` 0.323 → 0.610, McNemar `MAIN → ENT` p < 0.0001. `entropy_coef` is now a real dial; 0.005 is a floor found after 0.05 swamped the reward gradient and drove the policy into 300-token garbage that overflowed the scorer. |

`ppo_ac.py` picked up four corrections while these ran: the entropy feature
(`Config.entropy_coef`, `token_entropy`, the `-entropy_coef · H` term —
`entropy_coef=0.0` default, every run in `runs/` bit-identical), plus three latent
bugs from commit `58833d3` that these runs were the first to hit: `best`
referenced before assignment in the step-0 eval, `value_ev_gain` / the progress
line not handling the `None` `critic_fit` returns on a degenerate batch, and
`rollout` not surviving a completion the scorer cannot parse.

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

**The critic barely worked, and the new panels are what say so.** Two numbers, measuring
different things, and both are needed:

| | MAIN | ABLATE |
| --- | --- | --- |
| median **within-batch** `value_ev` (what feeds the advantage) | +0.0052 | +0.0047 |
| **across-step** explained variance of `value_mean` against `return_mean` | +0.0315 | +0.0166 |
| the same, for a zero-parameter trailing mean of the last 20 rewards | +0.0115 | +0.0073 |
| `std(V) / std(G)` | 0.083 | 0.124 |

So the 2049-parameter learned value function is worth about **two percentage points of
explained variance over `sum(last 20 rewards) / 20`**, and it explains essentially none of
the *within-batch* variation, which is the part the advantage is built from. `V` lived in
[0.265, 0.381] while its regression target ranged over [0.000, 0.860] — it moved 8% as much
as the thing it was predicting.

`value_std` averaged 0.0123 against a `value_mean` near 0.32 — a 3.8% relative spread, so
every token in a batch received effectively the same baseline. At λ=1 that makes
`A_t = R − const`, which is REINFORCE with a slow moving average. Whatever moved the policy
above, the critic contributed almost none of it.

*Correction:* an earlier version of this file quoted the across-step figure as −0.000. That
was read off a mid-run snapshot at step 93 and is wrong for the completed run; the numbers
above are from all 200 steps of each.

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
* `03` — executed, 5/5 CPU checks plus the three 200-step runs (`nodetach`, `ie2`, `ent`),
  committed with outputs. All three beat frozen and beat `MAIN` on the paired test; the
  entropy bonus (`ent`) is the only configuration in which the clip ever binds. Re-running
  it against the finished run directories plots instead of retraining.

What is shown: the loop is correct on real hardware at this scale; both policies improved
held-out `root_cause` accuracy by a margin that survives a paired test (p = 0.007 and
p = 0.021); raising the weight on `numeric` changed nothing about that, the two trained
policies disagreeing on exactly one case in 200; and the critic contributed none of it,
with explained variance at zero for all 400 steps.

What is not shown: that this is arithmetic rather than a better prior over seven labels —
`numeric_acc` is ~18 hits out of 600 and exact match is 0.000 everywhere. Nor that any of it
survives a second seed, a second split, or the sealed `test.jsonl`. One seed is one seed.
