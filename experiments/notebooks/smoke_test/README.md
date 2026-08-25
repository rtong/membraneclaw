# smoke_test — actor-critic PPO, as the counterpart to GRPO

The same task, the same frozen data, the same deterministic reward and the same 0.5B model
as [`../../membrane_grpo`](../../membrane_grpo). One thing is different: the baseline is a
**learned value function** rather than the mean of a group, and the advantage is assigned
**per token** rather than per sequence.

| | GRPO (`membrane_grpo/grpo_scratch.py`) | actor-critic PPO (here) |
| --- | --- | --- |
| baseline | mean reward of `G` samples of one prompt | learned `V(s_t)` |
| advantage | one scalar per sequence | one per token, via GAE |
| rollouts per prompt | 8 | 1 |
| degenerate case | all `G` rewards equal → gradient exactly zero, 16% of groups | none |
| extra parameters | 0 | 897 |

Nothing about the task is redefined here. `reward.py`, `task/prompt.py`, `data/*.jsonl` and
`build_mask` are **imported** from `membrane_grpo`, not copied — two methods measured with
one ruler, or the comparison means nothing.

## Layout

| | |
| --- | --- |
| `ppo_ac.py` | the implementation: GAE, both clipped losses, the value head, the loop. The gradient is written out by hand in the module docstring, as `grpo_scratch.py` does. |
| `01_actor_critic_and_gae.ipynb` | the derivation, and 14 checks that need no GPU. Runs on CPU in about a minute. Committed with outputs. |
| `02_ppo_actor_critic_0.5b.ipynb` | the run on the card, and the curves. **Not yet executed** — see below. |
| `runs/smoke-ppo-ac-cpu/` | 4 steps on CPU, evidence that the loop is whole. |

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

## The card has to be free first

`anton` has one RTX 5070 Ti at 16 GiB and the vLLM 9B service holds about 14.6 GiB of it.
Stopping it needs `sudo`, and this machine has no `NOPASSWD` rule — so it is done by a
person at a terminal, not by a notebook and not by an agent:

```sh
sudo systemctl stop vllm-qwen membraneclaw-agent    # any branch
deploy/stack.sh down                                 # branches that have it
```

Notebook 02's first cell refuses to run until that has happened, rather than letting a run
OOM twenty minutes in.

**Do not trust `torch.cuda.mem_get_info` on this machine.** With vLLM loaded it reported
13,454 MiB free while `nvidia-smi` reported 544 — under WSL2 the driver counts memory it
could page to host RAM as available to a new context. The guard uses `nvidia-smi` plus
`systemctl is-active`, and that pairing is the only one that has matched reality here.

## Two things the smoke tests caught

**The critic's learning rate is a derived quantity, not a guess.** Adam moves each of a
linear head's `d` weights by about `lr` per step and those steps sum through the features,
so `|dV| ≈ lr * ||h||_1`. On this model `||h||_1 ≈ 4.3e3`, so `value_lr=1e-3` moves the
value by ~4.3 against a reward that lives in `[0, 1]`. The first CPU run did exactly that:
`V` went `0.086 → -1.868` in one step and the gradient norm went from 10 to 239. The
default is now `5e-6`, which targets a step of ~0.02, and `value_step` is in the metrics so
the next person does not have to rediscover this from a bad reward curve.

**Left padding is a seam GRPO never touches.** A GRPO group is `G` samples of one prompt,
so every sequence has the same length and there is no padding. Taking one sample each from
eight different prompts left-pads the batch, and a left-padded batch through a plain
`model(input_ids=...)` attends to the pad tokens and starts RoPE at the pad. It raises no
error; it just corrupts the log-probabilities, which are the entire training signal.
Measured in `01`: **9.8 nats per token** on the padded row, and exact on the unpadded one —
invisible to any test whose batch happens to be uniform.

## Status

* `01` — executed, 14/14 checks pass, committed with outputs.
* `ppo_ac.py` — runs end to end. The CPU run in `runs/smoke-ppo-ac-cpu/` hit a batch where
  every reward was exactly `0.0` on step 2, the same case GRPO's own CPU smoke run hit on
  step 2. GRPO recorded `|grad| = 0.00000` there; this recorded `7.806`. That is the
  structural claim, arriving on its own within four steps rather than being staged.
* `02` — written and guarded, **not executed**: the card was held by vLLM throughout, and
  freeing it is a `sudo` action that belongs to a person at a terminal.

Nothing here has been shown to *learn*. Fourteen checks say the mathematics is implemented
correctly and the plumbing is not corrupting the signal; the reward curve is `02`'s job.
