"""Actor-critic PPO written out by hand, as the counterpart to GRPO.

`../../membrane_grpo/grpo_scratch.py` drops the value network: the other G-1
samples of the same prompt *are* the baseline. This file keeps the value
network and drops the group. Same data, same reward, same model, same clipped
surrogate -- the only thing that moves is where the baseline comes from and how
finely credit is assigned.

## The two methods side by side

| | GRPO | actor-critic PPO |
| --- | --- | --- |
| baseline | mean reward of G samples of one prompt | learned `V_phi(s_t)` |
| advantage | one scalar per sequence, broadcast to every token | one per token, via GAE |
| rollouts per prompt | G (8 in the reference run) | 1 is enough |
| degenerate case | all G rewards equal -> advantages exactly 0, no gradient | none: `R - V` is nonzero unless the critic is exactly right |
| extra parameters | none | `hidden_size + 1` |

The last two rows are the reasons to bother. 16% of GRPO's groups at the frozen
baseline are degenerate and contribute exactly nothing; a critic has no such
hole. What it has instead is a second thing that can be wrong, and a wrong
critic corrupts the *sign* of the advantage rather than merely zeroing it.

## The token-level MDP

The task is a contextual bandit -- one prompt, one completion, one terminal
scalar -- but PPO is written for a sequence of states, so the completion is read
as a trajectory:

    s_t = (prompt, a_0 .. a_{t-1})      the state before emitting token t
    a_t                                 the token emitted
    r_t = 0 for t < L-1,  r_{L-1} = R   reward arrives only at the last token
    gamma = 1                           completions are ~110 tokens; nothing
                                        here justifies preferring early tokens

`L` is the length of the *active* span: up to and including the first EOS, per
`build_mask` in `grpo_scratch.py`, which this file imports rather than
reimplements. Deciding to stop is an action and is scored; padding is not.

## GAE

With `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` and `V(s_L) = 0` (terminal):

    A_t = delta_t + (gamma * lam) * A_{t+1},      A_t = 0 for t >= L

Two limits are worth holding on to, and `01_actor_critic_and_gae.ipynb` checks
both numerically rather than taking them on faith:

* `lam = 1` collapses to `A_t = R - V(s_t)`. Unbiased, and the whole trajectory's
  noise lands on every token.
* `lam = 0` collapses to `A_t = V(s_{t+1}) - V(s_t)` for t < L-1. Low variance,
  and completely dependent on the critic being right.

The critic's regression target is the GAE return `G_t = A_t + V(s_t)`, which at
`lam = 1` is just `R` at every position.

## The actor gradient, by hand

Identical in form to the GRPO surrogate -- only `A` is now indexed by t:

    rho_t = exp(logp_t - logp_old_t)
    obj_t = min(rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t)

    d obj_t / d logp_t  =  rho_t * A_t   inside the trust region
                           0             when A_t > 0 and rho_t > 1+eps
                           0             when A_t < 0 and rho_t < 1-eps

At `rho = 1` this reduces to `A_t`, so with one inner epoch the clip is inert by
construction and this is vanilla actor-critic policy gradient. That is not a
defect: it is the same property `grpo_scratch` documents, and it makes the two
runs comparable at `--inner-epochs 1`.

## Two choices worth arguing about

**The value head does not backpropagate into the trunk** (`value_detach=True`).
A shared trunk lets the value loss reshape the representations the policy reads,
which is standard in RLHF and is also a confound in an experiment whose question
is what the *policy* loss did. Detached, the critic is a linear probe on the
policy's own hidden states: weaker, and clean. `--no-value-detach` restores the
shared-trunk version.

**Advantages are not whitened** (`whiten=False`). Whitening across a batch of
mixed prompts would re-introduce a batch-level baseline and paper over whether
the critic works at all -- which is the one thing being measured here. It also
happens to hide the pathology below, and that pathology is worth seeing.

## The pathology to watch for

The reward is non-negative. A value head initialised to zero therefore produces
`A_t = R >= 0` for every token of every completion, and the first updates
reinforce *everything*, garbage included. GRPO cannot do this: group centring
makes advantages zero-mean by construction. The default here initialises the
head's bias to the frozen policy's own held-out reward -- looked up in
`FROZEN_BASELINE` per `(model, weight set)`, because the reward is a weighted sum
and the same frozen Qwen3-1.7B scores 0.3015 under MAIN, 0.4500 under PROBE and
0.2615 under ABLATE. So the critic starts approximately calibrated instead of
catastrophically low. `--value-init-bias 0` reproduces the failure on purpose,
and the failure is now larger than it was on the 0.5B: the gap between a cold
head and a calibrated one is 0.30 here against 0.086 before.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SMOKE_DIR = Path(__file__).resolve().parent
GRPO_DIR = SMOKE_DIR.parents[1] / "membrane_grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

# Imported, never reimplemented: one tokenizer convention, one reward, one
# prompt, shared with the GRPO run so the comparison is not between two
# subtly different tasks.
from eval import supports_thinking_toggle  # noqa: E402
from grpo_scratch import build_mask, selective_logprobs  # noqa: E402
from reward import ABLATE, MAIN, PROBE, Weights, score  # noqa: E402
from task.prompt import PROMPT_VERSION, build_messages  # noqa: E402

DATA = GRPO_DIR / "data"

#: Held-out reward of the *frozen* policy on dev, greedy, prompt v2 -- one
#: number per weight set, because the reward is a weighted sum and the same
#: policy scores differently under each. This is what the value head's bias is
#: initialised to, so it must match the weights the run is actually using;
#: initialising to another weight set's baseline is the pathology in the module
#: docstring, only quieter.
#:
#: Qwen3-1.7B, from `membrane_grpo/runs/q3-{main,probe,ablate}-s0/eval.jsonl`
#: step 0, which is the same `eval.py` code path that produced the frozen
#: baselines. `STAGE_ONLY` is derived rather than measured: the MAIN evaluation
#: reports `components.stage = 0.0291` against a weight of 0.03, so the frozen
#: policy already copies the stage on 97% of cases.
FROZEN_BASELINE: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-1.7B": {
        "MAIN": 0.3015,
        "PROBE": 0.4500,
        "ABLATE": 0.2615,
        "STAGE_ONLY": 0.970,
    },
    # Kept so the 0.5B runs already in `runs/` stay reproducible from this file.
    # MAIN from runs/baseline-0.5b-v2/eval_dev_greedy.json -> overall.reward;
    # PROBE from MEMO.md's probe run at step 0. ABLATE is absent rather than
    # guessed -- it was never run on this model, and an unmeasured pair should
    # fall through to the loud 0.0 rather than borrow a neighbour's number.
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "MAIN": 0.086,
        "PROBE": 0.024,
        "STAGE_ONLY": 0.0,
    },
}


def value_init_bias_for(model: str, weights: str) -> float:
    """The warm start for the value head, or 0.0 if this pair was never measured.

    0.0 is the *documented* failure -- the reward is non-negative, so a critic at
    zero makes every advantage non-negative and the first updates reinforce
    everything, garbage included. Falling back to it is loud rather than
    plausible, which is what a fallback for a missing measurement should be.
    """
    return FROZEN_BASELINE.get(model, {}).get(weights, 0.0)


#: The default run's warm start: Qwen3-1.7B under MAIN. Named separately because
#: `01_actor_critic_and_gae.ipynb` contrasts it against a cold head at 0.0.
VALUE_INIT_BIAS = FROZEN_BASELINE["Qwen/Qwen3-1.7B"]["MAIN"]

#: The go/no-go weighting. Every point is on `stage`, which the prompt states
#: outright -- `task/generate.py` puts `anomaly_stage` in the record and the
#: answer copies it, so the correct output is a literal already on screen. No
#: arithmetic, no table lookup, no vocabulary the model has to infer.
#:
#: The question this asks is deliberately the easiest one available: can RL move
#: this model on a pure copy task when copying is the *only* thing that pays?
#: Under MAIN weights `stage` is worth 0.03 and 200 steps left it at 0.000. If
#: it does not move even here, the ceiling is not a tuning problem.
#:
#: That question was asked of the 0.5B and answered (`runs/gonogo-stage-s0`).
#: **It is not a live probe on Qwen3-1.7B**, which already copies the stage on
#: 97% of dev cases before any training -- there is 0.03 of headroom, so a
#: STAGE_ONLY run here measures a ceiling effect and nothing else. Kept for
#: reproducing the 0.5B result, not for rerunning on the new model.
STAGE_ONLY = Weights(format=0.0, numeric=0.0, flags=0.0, stage=1.0, root_cause=0.0, action=0.0)

#: The weight sets a run may be trained under, resolved by name in one place so
#: `train`, `evaluate` and the CLI cannot drift apart. `ABLATE` comes from the
#: GRPO side: it holds `numeric` at PROBE's 0.35 while leaving `root_cause`
#: substantial at 0.25, and on Qwen3-1.7B it settled which of the two weights
#: was doing the work (`membrane_grpo/runs/q3-ablate-s0`).
WEIGHT_SETS: dict[str, Weights] = {
    "MAIN": MAIN,
    "PROBE": PROBE,
    "ABLATE": ABLATE,
    "STAGE_ONLY": STAGE_ONLY,
}

#: Steps averaged when deciding which policy was "best". A single step is 8
#: samples and far too noisy to checkpoint on; a trailing mean is not.
#:
#: This is a *training* reward, and a training reward is not evidence: the
#: 0.5B stage run held 1.0000 here for 30 steps while it was worth 0.000 on
#: dev. With `eval_every` on, prefer `eval.jsonl` for that judgement; this
#: checkpoints often enough not to lose the policy in between.
BEST_WINDOW = 10

__all__ = [
    "ABLATE",
    "DATA",
    "FROZEN_BASELINE",
    "PROMPT_VERSION",
    "VALUE_INIT_BIAS",
    "WEIGHT_SETS",
    "value_init_bias_for",
    "MAIN",
    "PROBE",
    "STAGE_ONLY",
    "Weights",
    "ValueHead",
    "build_mask",
    "build_messages",
    "critic_fit",
    "evaluate",
    "gae",
    "load_cases",
    "ppo_actor_loss",
    "score",
    "selective_logprobs",
    "terminal_rewards",
    "value_loss",
    "whiten",
]


def load_cases(split: str = "train") -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


# --- the value head -----------------------------------------------------------


class ValueHead(nn.Module):
    """A scalar read off the policy's own last hidden state.

    Kept in float32 regardless of the trunk's dtype. The head is the only part
    of this that is a regression rather than a classification, and bf16's
    ~3 decimal digits of mantissa are not enough to keep a squared error stable
    when the target sits in [0, 1] and the differences being learned are ~0.01.
    """

    def __init__(self, hidden_size: int, init_bias: float = VALUE_INIT_BIAS):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1, dtype=torch.float32)
        nn.init.zeros_(self.v.weight)
        nn.init.constant_(self.v.bias, init_bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.v(hidden.float()).squeeze(-1)


# --- advantages ---------------------------------------------------------------


def terminal_rewards(rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Place each sequence's scalar reward on its own last active token.

    `rewards` is (batch,), `mask` is (batch, tokens). Returns (batch, tokens),
    zero everywhere except one position per row. A row with an empty mask -- a
    completion that was pure padding -- gets no reward anywhere rather than a
    reward at position -1, which is why the index is clamped and then masked
    again.
    """
    lengths = mask.sum(dim=-1).long()
    out = torch.zeros_like(mask)
    idx = (lengths - 1).clamp(min=0)
    out.scatter_(1, idx.unsqueeze(-1), rewards.unsqueeze(-1).to(out.dtype))
    return out * (lengths > 0).float().unsqueeze(-1)


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over the completion span.

    Shapes are all (batch, tokens). `rewards` is the per-token reward from
    `terminal_rewards`; `values` is `V(s_t)`, the value of the state *before*
    token t, so it lines up index-for-index with the log-probs.

    Returns `(advantages, returns)`, both zero outside the mask. `returns` is
    the GAE return `A_t + V(s_t)`, which is what the critic regresses against.

    The bootstrap past the last active token is zero, not `V(s_L)`: the episode
    genuinely ends there. Bootstrapping off a padded position would let the
    critic's own error leak in as if it were future reward.
    """
    batch, tokens = mask.shape
    advantages = torch.zeros_like(values)
    running = torch.zeros(batch, dtype=values.dtype, device=values.device)

    for t in range(tokens - 1, -1, -1):
        m = mask[:, t]
        # Zero past the end of the span, so V(s_{t+1}) is 0 on the last token.
        next_value = values[:, t + 1] * mask[:, t + 1] if t + 1 < tokens else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = delta + gamma * lam * running
        # An inactive position contributes nothing and must not carry the
        # accumulator backwards across the padding boundary either.
        running = running * m
        advantages[:, t] = running

    return advantages * mask, (advantages + values) * mask


def critic_fit(
    values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """How much of the return the critic actually explains, over one batch.

    `value_mean` and `value_mae` cannot tell a working critic from one that has
    collapsed to a constant, and the 0.5B runs in `runs/` are the demonstration:
    on `ppo-ac-ie2-s0` the learned value function posted a respectable MAE of
    0.089 and an **explained variance of +0.07** -- worse than predicting the
    mean of the returns, which is what a critic exists to beat. `value_std` says
    the same thing from the other side: on 125 steps of `gonogo-stage-s0` where
    every sequence in the batch drew the identical reward, V had mean 0.95 and a
    within-batch spread of 0.065, so V(s) was barely a function of s at all.

    Computed over the whole batch from the values the advantages were actually
    built from, not per micro-batch -- the per-micro-batch stats in `value_loss`
    are overwritten by whichever slice happens to run last.
    """
    sel = mask > 0
    v, g = values[sel], returns[sel]
    if v.numel() < 2:
        return {"value_ev": 0.0, "value_std": 0.0}
    g_var = float(g.var(unbiased=False))
    return {
        "value_ev": round(1.0 - float((g - v).var(unbiased=False)) / g_var, 6) if g_var > 0 else 0.0,
        "value_std": round(float(v.std(unbiased=False)), 6),
    }


def whiten(advantages: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance over active tokens only. Off by default."""
    active = mask.sum().clamp(min=1.0)
    mean = (advantages * mask).sum() / active
    var = (((advantages - mean) * mask) ** 2).sum() / active
    return ((advantages - mean) / (var.sqrt() + eps)) * mask


# --- the losses ---------------------------------------------------------------


def ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    normalize: str = "token",
) -> tuple[torch.Tensor, dict[str, float]]:
    """The clipped surrogate, with a per-token advantage.

    Differs from `grpo_surrogate` in exactly one respect: `advantages` is
    (batch, tokens) here and (batch,) there. The clipping, the normalisation and
    the diagnostics are deliberately identical so the two runs' numbers can be
    read against each other.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    objective = torch.min(unclipped, clipped)

    per_token = -objective * mask
    if normalize == "sequence":
        lengths = mask.sum(dim=-1).clamp(min=1.0)
        loss = (per_token.sum(dim=-1) / lengths).mean()
    elif normalize == "token":
        loss = per_token.sum() / mask.sum().clamp(min=1.0)
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    with torch.no_grad():
        active = mask.sum().clamp(min=1.0)
        binding = ((unclipped > clipped) & (mask > 0)).float().sum() / active
        stats = {
            "actor_loss": float(loss.detach()),
            "ratio_mean": float((ratio * mask).sum() / active),
            "clip_frac": float(binding),
            "adv_mean": float((advantages * mask).sum() / active),
            "adv_std": float(
                (((advantages - (advantages * mask).sum() / active) * mask) ** 2).sum().div(active).sqrt()
            ),
            "entropy_proxy": float(-(logprobs * mask).sum() / active),
        }
    return loss, stats


def value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float | None = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped squared error, the standard PPO critic loss.

    The clip is a trust region for the *critic*: the value is not allowed to
    move more than `clip_eps` from the estimate the batch was collected under,
    which stops one unlucky batch from dragging the baseline somewhere the
    advantages of the next batch cannot recover from. Taking the `max` of the
    two errors -- rather than the `min`, as the actor does -- is what makes it a
    penalty instead of a licence.

    `clip_eps=None` gives plain MSE, which is the honest thing to compare
    against and is what `01_actor_critic_and_gae.ipynb` checks the clip against.
    """
    active = mask.sum().clamp(min=1.0)
    plain = (values - returns) ** 2
    if clip_eps is None:
        loss = (plain * mask).sum() / active
        clipped_frac = 0.0
    else:
        moved = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        clipped_err = (moved - returns) ** 2
        both = torch.max(plain, clipped_err)
        loss = (both * mask).sum() / active
        with torch.no_grad():
            clipped_frac = float(((clipped_err > plain) & (mask > 0)).float().sum() / active)

    with torch.no_grad():
        err = (values - returns) * mask
        stats = {
            "value_loss": float(loss.detach()),
            "value_mean": float((values * mask).sum() / active),
            "return_mean": float((returns * mask).sum() / active),
            "value_mae": float(err.abs().sum() / active),
            "value_clip_frac": clipped_frac,
        }
    return loss, stats


@dataclass
class Config:
    """Defaults chosen to sit alongside `grpo_scratch.Config`, not to beat it."""

    # Qwen3-1.7B, chosen on the GRPO side by measurement rather than argument
    # (`membrane_grpo/runs/sel-*`). The reason that matters here: its schema
    # validity is already 0.970, so a reward rise cannot be explained away as
    # format learning -- which is exactly what 52% of the 0.5B's gain turned out
    # to be. It also starts at cause_acc 0.255 against a 1/7 = 0.143 chance
    # floor, so there is something partial for RL to sharpen; the 0.5B sat
    # *exactly* at chance and had nothing.
    model: str = "Qwen/Qwen3-1.7B"
    steps: int = 200
    prompts_per_step: int = 8
    # 1 is the point: unlike GRPO, the algorithm needs no second sample of the
    # same prompt. Raise it only to fill the batch, never for the baseline.
    samples_per_prompt: int = 1
    max_new_tokens: int = 640
    temperature: float = 1.0
    lr: float = 1e-5
    # Derived, not guessed, and re-derived for this model rather than carried
    # over. Adam moves every one of the head's `hidden_size` weights by about
    # `lr` per step, and those steps are summed through the features, so the
    # value itself moves by roughly `lr * ||h||_1`. On the 0.5B that meant
    # lr=1e-3 moved V by ~4.2 against a reward living in [0, 1], and the first
    # CPU smoke run did exactly that: V went 0.086 -> -1.868 in one step.
    #
    # The constant does not survive the model change, and not in the direction
    # the parameter count suggests. Measured in `01_actor_critic_and_gae.ipynb`:
    #
    #     Qwen2.5-0.5B   hidden  896   E|h| 4.824   ||h||_1 4323
    #     Qwen3-1.7B     hidden 2048   E|h| 1.127   ||h||_1 2308
    #
    # 2.3x the dimensions but 4.3x smaller per-dimension activations -- Qwen3
    # simply does not carry the huge outlier features the 0.5B does -- so
    # `||h||_1` *falls* by 47%. Keeping 5e-6 would have quietly cut the critic's
    # step from 0.022 to 0.012 on a model where the critic was already the
    # bottleneck. 8.5e-6 puts it at 0.020, which is the target the old default
    # was chosen for; `value_step` in the metrics confirms it per run.
    #
    # That target is itself the open question, and it is deliberately *not*
    # touched here so that the model is the only thing that changed. The 0.5B
    # runs moved V by a median of 0.004-0.009 per step against a regression
    # target whose own batch-to-batch swing reaches 1.0 -- a ~100-step time
    # constant chasing a signal that changes every step, which is why
    # `value_ev` came out at +0.07. `value_ev` is now in the metrics so the
    # next round argues from a number instead of from a reward curve.
    value_lr: float = 8.5e-6
    weight_decay: float = 0.0  # same reasoning as grpo_scratch: no reward-free force
    gamma: float = 1.0
    lam: float = 1.0
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    value_coef: float = 0.5
    value_detach: bool = True
    whiten_advantages: bool = False
    # -1.0 means "look it up in FROZEN_BASELINE for this model and weight set"
    # and is resolved in __post_init__, so config.json records the number that
    # was actually used. 0.0 still reproduces the documented failure on purpose.
    value_init_bias: float = -1.0
    inner_epochs: int = 1
    # 1, not 2: 1.7B in bf16 is 3.4 GiB of weights before activations, and the
    # value head needs `output_hidden_states=True`, which keeps all 29 layers'
    # hidden states alive through the backward pass. Same choice GRPO made when
    # it moved to this model.
    micro_batch: int = 1
    normalize: str = "token"
    lora_r: int = 16
    weights: str = "MAIN"
    seed: int = 0
    dtype: str = "bfloat16"
    split: str = "train"
    prompt_version: str = PROMPT_VERSION
    # Held-out evaluation during training, taken from `grpo_scratch.Config`.
    # The reason to have it is on this side of the fence, not GRPO's: this
    # method has already produced a run (`runs/gonogo-stage-s0`) whose *training*
    # reward sat at 1.0000 for 30 steps while the policy it was checkpointing
    # was worth 0.000 on dev. A reward curve alone cannot tell learning from
    # collapse, and `adapter-best` picked the wreck.
    eval_every: int = 25
    # The full dev split, so step 0 is directly comparable to the frozen
    # baseline rather than to a subset of it with a different denominator.
    eval_cases: int = 200
    eval_batch: int = 32
    eval_split: str = "dev"
    eval_max_tokens: int = 640
    metrics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.weights not in WEIGHT_SETS:
            raise ValueError(f"unknown weights {self.weights!r}; have {sorted(WEIGHT_SETS)}")
        if self.value_init_bias < 0:
            self.value_init_bias = value_init_bias_for(self.model, self.weights)


# --- the model side -----------------------------------------------------------
#
# One seam here that GRPO never meets. A GRPO group is G samples of a *single*
# prompt, so every sequence in the batch has the same prompt length and there is
# no padding to get wrong. This method takes one sample each from many different
# prompts, so the batch is left-padded -- and a left-padded batch fed through a
# plain `model(input_ids=...)` is silently wrong twice over: the model attends to
# the pad tokens, and RoPE reads position 0 at the pad rather than at the first
# real token. Both are fixed below, and `01_actor_critic_and_gae.ipynb` checks
# the fix by scoring the same sequence padded and unpadded and comparing.


def padded_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that start at 0 on the first *real* token of each row."""
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def forward_policy_value(
    model,
    value_head,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_len: int,
    *,
    value_detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs and values over the completion span, in one forward.

    `logits_to_keep` restricts the (batch, positions, 151936) output head to the
    span that is actually scored -- the single largest tensor in the step, and
    what exhausted the card on the GRPO run's first attempt. The hidden states
    come back full length regardless, so `V(s_t)` is sliced from them at the
    same positions whose logits produced token t.
    """
    out = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        position_ids=padded_position_ids(attention_mask),
        logits_to_keep=completion_len + 1,
        output_hidden_states=True,
    )
    logprobs = selective_logprobs(out.logits[:, :-1], sequences[:, -completion_len:])

    hidden = out.hidden_states[-1]
    if hidden.shape[1] == sequences.shape[1]:
        hidden = hidden[:, -(completion_len + 1) : -1]
    else:  # a transformers version that also truncates the hidden states
        hidden = hidden[:, :-1]
    values = value_head(hidden.detach() if value_detach else hidden)
    return logprobs, values


def load_policy(cfg: "Config", device: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=getattr(torch, cfg.dtype))
    policy = get_peft_model(
        base,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=2 * cfg.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    value_head = ValueHead(base.config.hidden_size, cfg.value_init_bias).to(device)
    return policy, value_head, tokenizer


@dataclass
class Rollout:
    cases: list[dict[str, Any]]
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    mask: torch.Tensor
    completions: list[str]
    rewards: list[float]


@torch.no_grad()
def rollout(
    policy,
    tokenizer,
    cases: list[dict[str, Any]],
    *,
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    weights: Weights,
    device: str,
    template_kwargs: dict[str, Any] | None = None,
) -> Rollout:
    """Sample one batch: `samples_per_prompt` completions for each case."""
    expanded = [c for c in cases for _ in range(samples_per_prompt)]
    texts = [
        tokenizer.apply_chat_template(
            build_messages(c["record"]),
            tokenize=False,
            add_generation_prompt=True,
            **(template_kwargs or {}),
        )
        for c in expanded
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    prompt_len = encoded["input_ids"].shape[1]

    out = policy.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if out.shape[1] < prompt_len + max_new_tokens:
        pad = out.new_full(
            (out.shape[0], prompt_len + max_new_tokens - out.shape[1]), tokenizer.pad_token_id
        )
        out = torch.cat([out, pad], dim=1)

    completion_ids = out[:, prompt_len:]
    mask = build_mask(completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    rewards = [score(c, case["answer"], weights).total for c, case in zip(completions, expanded)]

    return Rollout(
        cases=expanded,
        sequences=out,
        # The prompt's own padding, then the completion's active span. A token
        # after the first EOS was never chosen and must not be attended to.
        attention_mask=torch.cat([encoded["attention_mask"], mask.long()], dim=1),
        mask=mask,
        completions=completions,
        rewards=rewards,
    )


# --- held-out evaluation ------------------------------------------------------


def evaluate(policy, tokenizer, cfg: "Config", cases: list[dict], step: int) -> dict[str, Any]:
    """Greedy pass over a fixed held-out slice, through `eval.py`'s own code path.

    Lifted from `grpo_scratch.evaluate` unchanged in substance, and that is the
    point: the PPO curve, the GRPO curve and the frozen baseline are then all
    produced by literally the same function, so the three are comparable without
    an argument about whose harness was fairer.

    The value head is not involved. This measures the policy, and the critic is
    scaffolding for training it.
    """
    from eval import generate_hf, summarise

    # generate_hf still moves the encoded batch onto a device, so it needs the
    # one the live policy is already on rather than a re-derived guess.
    device = str(next(policy.parameters()).device)
    results = generate_hf(
        cases,
        model=cfg.model,
        device=device,
        dtype=cfg.dtype,
        n=1,
        temperature=0.0,
        max_tokens=cfg.eval_max_tokens,
        seed=cfg.seed,
        batch_size=cfg.eval_batch,
        adapter=None,
        prompt_version=cfg.prompt_version,
        loaded=(policy, tokenizer),
    )
    metrics = summarise(results, WEIGHT_SETS[cfg.weights])
    return {
        "step": step,
        "reward": metrics["reward"],
        "exact_match": metrics["exact_match"],
        "validity_gate": metrics["validity_gate"],
        "schema_ok": metrics["schema_ok"],
        "cause_acc": metrics["cause_acc"],
        "flags_acc": metrics["flags_acc"],
        "numeric_acc": metrics["numeric_acc"],
        "action_acc": metrics["action_acc"],
        "completion_tokens": metrics["completion_tokens_mean"],
    }


# --- training -----------------------------------------------------------------


def train(cfg: Config, out_dir: Path, device: str, *, progress=print) -> list[dict[str, Any]]:
    """One PPO actor-critic run. Returns the per-step metrics it also writes.

    Deliberately the same shape as `grpo_scratch.train`: same rollout-then-update
    order, same micro-batching, same `metrics.jsonl` next to a `config.json`, so
    `02_ppo_actor_critic_1.7b.ipynb` can plot the two runs on one axis without
    special-casing either.
    """
    import json
    import random
    import time
    from dataclasses import asdict

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    cases = load_cases(cfg.split)
    weights = WEIGHT_SETS[cfg.weights]

    policy, value_head, tokenizer = load_policy(cfg, device)

    # A Qwen3 chat template defaults thinking *on*, and on this task that is
    # fatal rather than merely verbose: the policy spends its whole budget
    # inside <think> and returns nothing to score, so every rollout fails the
    # gate at 0.0 and the run trains on noise. Qwen2.5 templates have no such
    # variable, so this is detected rather than switched on by model name --
    # the same fix `eval.py` and `grpo_scratch.py` already carry.
    template_kwargs: dict[str, Any] = {}
    if supports_thinking_toggle(tokenizer):
        template_kwargs["enable_thinking"] = False
        progress("  chat template honours enable_thinking; set to False")

    actor_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": actor_params, "lr": cfg.lr},
            {"params": list(value_head.parameters()), "lr": cfg.value_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    eval_path = out_dir / "eval.jsonl"
    eval_path.write_text("")

    eval_cases = [
        json.loads(line)
        for line in (DATA / f"{cfg.eval_split}.jsonl").read_text().splitlines()
    ][: cfg.eval_cases]

    def run_eval(step: int) -> dict[str, Any]:
        record = evaluate(policy, tokenizer, cfg, eval_cases, step)
        with eval_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        progress(
            f"    eval @{step:<4} reward {record['reward']:.4f} "
            f"cause {record['cause_acc']:.3f} flags {record['flags_acc']:.3f} "
            f"EM {record['exact_match']:.3f} valid {record['validity_gate']:.3f}"
        )
        return record

    per_step = cfg.prompts_per_step * cfg.samples_per_prompt
    progress(
        f"{cfg.model} | {cfg.steps} steps | {cfg.prompts_per_step}x{cfg.samples_per_prompt}"
        f"={per_step} seq/step | lam={cfg.lam} | {device}"
    )

    if cfg.eval_every:
        run_eval(0)  # step 0 must reproduce the frozen baseline

    history: list[dict[str, Any]] = []
    # An RL run that has solved its task can still destroy itself: this one's
    # reward sat at 1.0000 from step 170 and collapsed to 0.0000 at 196 with the
    # gradient norm going from 2.7 to 109. Saving only the final policy saved the
    # wreck, and every measurement afterwards was of the wreck. Checkpoint the
    # best trailing mean as well, and say so if the two differ.
    best: dict[str, Any] = {"reward": float("-inf"), "step": None}
    for step in range(cfg.steps):
        started = time.perf_counter()
        batch = [rng.choice(cases) for _ in range(cfg.prompts_per_step)]

        policy.eval()
        roll = rollout(
            policy,
            tokenizer,
            batch,
            samples_per_prompt=cfg.samples_per_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            weights=weights,
            device=device,
            template_kwargs=template_kwargs,
        )
        gen_seconds = time.perf_counter() - started

        sequences, mask = roll.sequences, roll.mask
        attn = roll.attention_mask
        completion_len = mask.shape[1]
        rewards = torch.tensor(roll.rewards, device=device, dtype=torch.float32)
        per_token_rewards = terminal_rewards(rewards, mask)

        policy.train()
        with torch.no_grad():
            old_lp, old_v = [], []
            for i in range(0, len(sequences), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                lp, v = forward_policy_value(
                    policy, value_head, sequences[sl], attn[sl], completion_len,
                    value_detach=cfg.value_detach,
                )
                old_lp.append(lp)
                old_v.append(v)
            old_logprobs = torch.cat(old_lp)
            old_values = torch.cat(old_v)

            advantages, returns = gae(
                per_token_rewards, old_values, mask, gamma=cfg.gamma, lam=cfg.lam
            )
            if cfg.whiten_advantages:
                advantages = whiten(advantages, mask)

        actor_stats: dict[str, float] = {}
        critic_stats: dict[str, float] = {}
        for _ in range(cfg.inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for i in range(0, len(sequences), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                share = min(cfg.micro_batch, len(sequences) - i) / len(sequences)
                logprobs, values = forward_policy_value(
                    policy, value_head, sequences[sl], attn[sl], completion_len,
                    value_detach=cfg.value_detach,
                )
                a_loss, actor_stats = ppo_actor_loss(
                    logprobs, old_logprobs[sl], advantages[sl], mask[sl],
                    clip_eps=cfg.clip_eps, normalize=cfg.normalize,
                )
                v_loss, critic_stats = value_loss(
                    values, old_values[sl], returns[sl], mask[sl], clip_eps=cfg.value_clip_eps,
                )
                loss = a_loss + cfg.value_coef * v_loss
                (loss * share).backward()
                accumulated += float(loss.detach()) * share
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_params + list(value_head.parameters()), 1.0
            )
            optimizer.step()

        record = {
            "step": step,
            "reward_mean": sum(roll.rewards) / len(roll.rewards),
            "reward_max": max(roll.rewards),
            # GRPO's failure mode, measured on this method for comparison: the
            # fraction of tokens whose advantage is exactly zero. A critic makes
            # this ~0 by construction, which is the claim being checked.
            "adv_zero_frac": float(
                (((advantages == 0) & (mask > 0)).float().sum() / mask.sum().clamp(min=1)).item()
            ),
            # Whether the critic is a critic. See `critic_fit`: on the 0.5B runs
            # `value_mean` and `value_mae` looked reasonable while `value_ev`
            # was +0.07, i.e. the learned value function lost to a constant.
            **critic_fit(old_values, returns, mask),
            "completion_tokens": float(mask.sum() / mask.shape[0]),
            "unique_completions": float(len(set(roll.completions))),
            "loss": accumulated,
            "grad_norm": float(grad_norm),
            "gen_seconds": round(gen_seconds, 2),
            "step_seconds": round(time.perf_counter() - started, 2),
            **{k: round(v, 6) for k, v in actor_stats.items()},
            **{k: round(v, 6) for k, v in critic_stats.items()},
        }
        # How far the critic actually moved this step. The knob that sets it is
        # `value_lr`, and the first CPU smoke run had it two orders of magnitude
        # too high; this is the number that makes that visible without waiting
        # for the reward curve to look wrong.
        record["value_step"] = round(
            abs(record["value_mean"] - history[-1]["value_mean"]) if history else 0.0, 6
        )
        history.append(record)
        window = [r["reward_mean"] for r in history[-BEST_WINDOW:]]
        trailing = sum(window) / len(window)
        record["reward_trailing"] = round(trailing, 6)
        if len(history) >= BEST_WINDOW and trailing > best["reward"]:
            best = {"reward": trailing, "step": step}
            policy.save_pretrained(out_dir / "adapter-best")
            torch.save(value_head.state_dict(), out_dir / "value_head-best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n")
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        progress(
            f"  step {step:>4} reward {record['reward_mean']:.4f} "
            f"V {record['value_mean']:.3f} (mae {record['value_mae']:.3f}) "
            f"adv {record['adv_mean']:+.3f}+-{record['adv_std']:.3f} "
            f"tok {record['completion_tokens']:.0f} "
            f"|g| {record['grad_norm']:.3f} {record['step_seconds']:.1f}s"
        )

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            run_eval(step + 1)

    policy.save_pretrained(out_dir / "adapter")
    torch.save(value_head.state_dict(), out_dir / "value_head.pt")
    progress(f"\nwrote {out_dir}")
    if best["step"] is not None and best["step"] < cfg.steps - 1:
        progress(
            f"NOTE: best trailing-{BEST_WINDOW} reward {best['reward']:.4f} was at step "
            f"{best['step']}, not at the end ({record['reward_mean']:.4f}). "
            f"adapter-best/ holds that policy; adapter/ holds the final one."
        )
    return history


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, value in asdict(Config()).items():
        if name == "metrics":
            continue
        if isinstance(value, bool):
            parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=int(value))
        else:
            parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    # The sentinel, not the resolved value. `asdict(Config())` has already run
    # __post_init__, so without this every run would inherit MAIN's baseline as
    # a hard-coded default -- including `--weights PROBE`, whose frozen policy
    # scores 0.45 rather than 0.30. A wrong bias on a critic this slow to move
    # is not a detail: it is most of the baseline for the whole run.
    parser.set_defaults(value_init_bias=-1.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.weights not in WEIGHT_SETS:
        parser.error(f"--weights must be one of {sorted(WEIGHT_SETS)}")

    fields = {k: v for k, v in vars(args).items() if k in asdict(Config()) and k != "metrics"}
    fields["value_detach"] = bool(fields["value_detach"])
    fields["whiten_advantages"] = bool(fields["whiten_advantages"])
    cfg = Config(**fields)
    tag = cfg.model.rsplit("/", 1)[-1].replace(".", "").lower()
    out = args.out or SMOKE_DIR / "runs" / f"ppo-{tag}-{cfg.weights.lower()}-s{cfg.seed}"
    train(cfg, out, resolve_device(args.device))


if __name__ == "__main__":
    main()
