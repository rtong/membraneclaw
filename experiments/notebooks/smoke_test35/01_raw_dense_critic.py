"""Actor-critic PPO on the raw Qwen3.5-0.8B, dense per-field credit, v2 prompt.

This file belongs to `01_raw_dense_critic.ipynb` and to nothing else. The
series in this directory does not share code between notebooks: each notebook
that needs more than fits in its cells gets its own file with its own prefix,
and a later notebook copies what it needs instead of importing this.

## What is being reproduced

The baseline is the configuration `../smoke_test/` settled on for the v2 prompt
(`13b`): PPO with a learned value head, and the reward paid **per field, at the
token that writes the field**, instead of once at the end. Two things are taken
away and nothing is added:

* no calculator tool -- the model does its own arithmetic, as v2 asks it to;
* no SFT seed -- PPO starts from the raw model.

The prompt is `task.prompt.build_messages(record, "v2")`, byte for byte.

## The model

`Qwen/Qwen3.5-0.8B`: 24 layers, three of every four a GatedDeltaNet linear-
attention layer and every fourth full attention, hidden 1024, vocabulary
248,320. Its chat template honours `enable_thinking`, and it is set to False,
for the reason `membrane_grpo/eval.py` records. LoRA covers the mixing
projections of both layer types (`LORA_TARGETS`); covering only
`q/k/v/o_proj` would leave 18 of the 24 layers frozen.

## The loop, in one paragraph

Sample one completion for each of 8 training prompts at temperature 1.0. Score
it with `reward.score` under ABLATE and split the score into per-field credits
placed at the tokens that complete each field (`dense_rewards`; rows sum to the
score exactly). Read `V(s_t)` off the policy's hidden state with a linear head
that does not backpropagate into the trunk. Build per-token advantages with GAE
(`gamma = 1`, `lam = 0.95`). Fit the head for 8 Adam steps over the last 25
batches, then take 4 clipped-surrogate epochs on the actor with an entropy bonus
of 0.005. Every 25 steps, one greedy pass over the 200 dev cases through
`eval.generate_hf` and `eval.summarise`, the same functions that measure the
frozen model.

`reward.py`, `task/`, `data/`, `build_mask`, `selective_logprobs`,
`generate_hf` and `summarise` are imported from `membrane_grpo` and never
copied, so the task and the ruler are the ones every other run in this
repository was measured with.
"""
from __future__ import annotations

import sys

# Importing from membrane_grpo must leave no trace in that tree.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import asdict, dataclass, fields  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
PREFIX = "01_raw_dense_critic"
RUNS = HERE / f"{PREFIX}_runs"
GRPO_DIR = HERE.parents[1] / "membrane_grpo"
if str(GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_DIR))

from eval import generate_hf, summarise, supports_thinking_toggle  # noqa: E402
from grpo_scratch import build_mask, selective_logprobs  # noqa: E402
from reward import ABLATE, Weights, _flag_hits, _numeric_hits, score  # noqa: E402
from task.prompt import build_messages  # noqa: E402
from task.schema import FLAG_KEYS, NUMERIC_KEYS, parse_answer, validate  # noqa: E402

DATA = GRPO_DIR / "data"
MODEL = "Qwen/Qwen3.5-0.8B"
PROMPT_VERSION = "v2"

#: The attention projections of both layer types. Full attention: q/k/v/o.
#: GatedDeltaNet: the fused qkv input projection, the output gate `z`, the two
#: per-head scalars `a` (decay) and `b` (write strength), and the output.
LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
)

WEIGHT_SETS: dict[str, Weights] = {"ABLATE": ABLATE}


def load_cases(split: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


def render_prompt(tokenizer, record: dict[str, Any]) -> str:
    """The v2 messages through the model's own chat template, thinking off.

    `eval.generate_hf` renders its prompts with exactly these arguments, which is
    what makes a training rollout and a held-out evaluation the same task.
    """
    kwargs = {"enable_thinking": False} if supports_thinking_toggle(tokenizer) else {}
    return tokenizer.apply_chat_template(
        build_messages(record, PROMPT_VERSION), tokenize=False, add_generation_prompt=True, **kwargs
    )


def safe_score(completion: str, answer: dict[str, Any], weights: Weights):
    """`reward.score`, with a completion the scorer cannot process worth 0.

    An exploring policy can write a number outside float range, which
    `reward._numeric_hits` raises on. That earns nothing; it must not end the run.
    """
    try:
        return score(completion, answer, weights)
    except (OverflowError, ValueError, TypeError, KeyError):
        return None


# --- dense per-field credit ---------------------------------------------------

#: Each output field, with the regex that finds the *end* of its value. The three
#: flags are searched inside the flags object, because "flow" also occurs inside
#: "normalized_flow_change_pct".
FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("normalized_flow_change_pct", r'"normalized_flow_change_pct"\s*:\s*(-?\d+(?:\.\d+)?)'),
    ("salt_passage_change_pct", r'"salt_passage_change_pct"\s*:\s*(-?\d+(?:\.\d+)?)'),
    ("dp_change_pct", r'"dp_change_pct"\s*:\s*(-?\d+(?:\.\d+)?)'),
    ("flags.flow", r'"flow"\s*:\s*"[a-z_]*"'),
    ("flags.salt_passage", r'"salt_passage"\s*:\s*"[a-z_]*"'),
    ("flags.dp", r'"dp"\s*:\s*"[a-z_]*"'),
    ("stage", r'"stage"\s*:\s*"[a-z_]*"'),
    ("root_cause", r'"root_cause"\s*:\s*"[a-z_]*"'),
    ("action", r'"action"\s*:\s*"[a-z_]*"'),
)


def field_credits(
    completion: str, answer: dict[str, Any], weights: Weights, flat_scale: float = 1.0
) -> tuple[dict[str, tuple[int, float]], float]:
    """One completion's reward split into `{field: (char offset, credit)}` and a rest.

    With `flat_scale = 1` the credits and the rest sum to
    `score(completion, answer, weights).total` exactly. `numeric` and `flags` are
    scored by `reward.py` as the mean of three hits, so each hit carries a third
    of the component's weight.

    `flat_scale` multiplies the credit of a flag whose true value is `flat` and
    that the completion got right -- `smoke_test/09`'s lever, part of the
    baseline configuration. A wrong `flat` still earns nothing. It changes what
    the actor is paid, not what any evaluation reports.

    `format` (schema validity) is not decided until the object closes, and a
    credit whose field cannot be located goes to the same place: the last active
    token, where a terminal reward would have put everything.
    """
    parsed = parse_answer(completion)
    obj = parsed.obj if isinstance(parsed.obj, dict) else None
    if obj is None:
        return {}, 0.0

    numeric, flags = _numeric_hits(obj, answer), _flag_hits(obj, answer)

    def flag_credit(key: str, hit: bool) -> float:
        scale = flat_scale if answer["flags"][key] == "flat" else 1.0
        return weights.flags / 3 * scale * hit

    earned = {
        **{k: weights.numeric / 3 * hit for k, hit in zip(NUMERIC_KEYS, numeric)},
        **{f"flags.{k}": flag_credit(k, hit) for k, hit in zip(FLAG_KEYS, flags)},
        "stage": weights.stage * (obj.get("stage") == answer["stage"]),
        "root_cause": weights.root_cause * (obj.get("root_cause") == answer["root_cause"]),
        "action": weights.action * (obj.get("action") == answer["action"]),
    }
    rest = weights.format * float(validate(obj).ok)

    flags_block = re.search(r'"flags"\s*:\s*\{[^}]*\}', completion)
    placed: dict[str, tuple[int, float]] = {}
    for name, pattern in FIELD_PATTERNS:
        credit = float(earned.get(name, 0.0))
        if not credit:
            continue
        if name.startswith("flags."):
            if flags_block is None:
                rest += credit
                continue
            where, base = flags_block.group(), flags_block.start()
        else:
            where, base = completion, 0
        hit = None
        for hit in re.finditer(pattern, where):
            pass
        if hit is None:
            rest += credit
            continue
        placed[name] = (base + hit.end(), credit)
    return placed, rest


def dense_rewards(
    completions: list[str],
    completion_ids: torch.Tensor,
    mask: torch.Tensor,
    cases: list[dict[str, Any]],
    weights: Weights,
    tokenizer,
    flat_scale: float = 1.0,
) -> torch.Tensor:
    """Per-token rewards, each field's credit on the token that completes it.

    A character offset is turned into a token index through prefix decodes --
    the decoded length after each token -- rather than per-token decodes, which
    do not add up when byte-level BPE splits a character across tokens.
    """
    out = torch.zeros_like(mask, dtype=torch.float32)
    lengths = mask.sum(dim=-1).long()
    for b, (text, case) in enumerate(zip(completions, cases)):
        length = int(lengths[b])
        if length == 0:
            continue
        ids = completion_ids[b, :length].tolist()
        bounds = [len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)) for i in range(length)]
        try:
            placed, rest = field_credits(text, case["answer"], weights, flat_scale)
        except (OverflowError, ValueError, TypeError, KeyError):
            placed, rest = {}, 0.0
        for offset, credit in placed.values():
            index = next((i for i, end in enumerate(bounds) if end >= offset), length - 1)
            out[b, index] += credit
        out[b, length - 1] += rest
    return out


# --- GAE and the critic's scorecard -------------------------------------------


def gae(
    rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, *, gamma: float = 1.0, lam: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over the completion span.

    All (batch, tokens). `values[:, t]` is `V(s_t)`, the state before token t.
    The bootstrap past the last active token is 0: the episode ends there, and
    bootstrapping off a padded position would feed the critic's own error back
    in as reward. Returns `(advantages, returns)`, zero outside the mask;
    `returns = advantages + values` is the critic's regression target.
    """
    batch, tokens = mask.shape
    advantages = torch.zeros_like(values)
    running = torch.zeros(batch, dtype=values.dtype, device=values.device)
    for t in range(tokens - 1, -1, -1):
        next_value = values[:, t + 1] * mask[:, t + 1] if t + 1 < tokens else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = (delta + gamma * lam * running) * mask[:, t]
        advantages[:, t] = running
    return advantages * mask, (advantages + values) * mask


def position_clock(returns: torch.Tensor, mask: torch.Tensor, bins: int = 20) -> torch.Tensor:
    """The mean return at each relative position, in `bins` bins.

    With dense credit the return falls as the answer is written, so a head that
    knows nothing but "how far along am I" already explains part of it. This is
    that head, fitted for free on the same batch. A critic is worth having only
    by what it explains above this.
    """
    sel = mask > 0
    out = torch.zeros_like(returns)
    index = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    binned = (index / lengths * bins).floor().clamp(max=bins - 1).long()
    for b in range(bins):
        hit = (binned == b) & sel
        if hit.any():
            out[hit] = returns[hit].mean()
    return out


def critic_fit(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> dict[str, float | None]:
    """Explained variance of the critic, and of the position clock, on one batch.

    None, not 0.0, when the returns have no variance: the metric is undefined on
    that batch, which is different from the critic failing on it.
    """
    sel = mask > 0
    v, g = values[sel], returns[sel]
    g_var = float(g.var(unbiased=False)) if g.numel() > 1 else 0.0
    std = round(float(v.std(unbiased=False)), 6) if v.numel() > 1 else 0.0
    if g_var < 1e-6:
        return {"value_ev": None, "value_ev_position": None, "value_std": std}
    clock = position_clock(returns, mask)[sel]
    return {
        "value_ev": round(1.0 - float((g - v).var(unbiased=False)) / g_var, 6),
        "value_ev_position": round(1.0 - float((g - clock).var(unbiased=False)) / g_var, 6),
        "value_std": std,
    }


# --- the losses ---------------------------------------------------------------


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the next-token distribution, per position, in float32."""
    logits = logits.float()
    return torch.logsumexp(logits, dim=-1) - (torch.softmax(logits, dim=-1) * logits).sum(dim=-1)


def ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float,
    entropy: torch.Tensor | None,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The clipped surrogate with a per-token advantage, averaged over active tokens,
    minus `entropy_coef` times the mean entropy over the same tokens."""
    active = mask.sum().clamp(min=1.0)
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -(torch.min(unclipped, clipped) * mask).sum() / active
    stats: dict[str, float] = {"actor_loss": float(loss.detach())}
    if entropy is not None:
        mean_entropy = (entropy * mask).sum() / active
        loss = loss - entropy_coef * mean_entropy
        stats["policy_entropy"] = float(mean_entropy.detach())
    with torch.no_grad():
        mean_adv = (advantages * mask).sum() / active
        stats.update({
            "ratio_mean": float((ratio * mask).sum() / active),
            "clip_frac": float((((unclipped > clipped) & (mask > 0)).float().sum()) / active),
            "adv_mean": float(mean_adv),
            "adv_std": float(((((advantages - mean_adv) * mask) ** 2).sum() / active).sqrt()),
            "entropy_proxy": float(-(logprobs * mask).sum() / active),
        })
    return loss, stats


def value_loss(
    values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, *, clip_eps: float | None
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped squared error on flat (n_active,) tensors: the value may not move
    more than `clip_eps` from the estimate the batch was collected under."""
    plain = (values - returns) ** 2
    if clip_eps is None:
        loss = plain.mean()
    else:
        moved = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        loss = torch.max(plain, (moved - returns) ** 2).mean()
    with torch.no_grad():
        stats = {
            "value_loss": float(loss),
            "value_mean": float(values.mean()),
            "return_mean": float(returns.mean()),
            "value_mae": float((values - returns).abs().mean()),
        }
    return loss, stats


# --- the model side -----------------------------------------------------------


class ValueHead(nn.Module):
    """`V(s_t) = w . h_t + b` on one detached hidden layer, in float32.

    Zero weight at init, so `V` starts at exactly `init_bias` everywhere.
    """

    def __init__(self, hidden_size: int, init_bias: float):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1, dtype=torch.float32)
        nn.init.zeros_(self.v.weight)
        nn.init.constant_(self.v.bias, init_bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.v(hidden.float()).squeeze(-1)


def padded_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that start at 0 on each row's first real token (left padding)."""
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def forward_logprobs(
    model,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_len: int,
    *,
    hidden_layer: int | None = None,
    want_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Per-token log-probs over the completion span, plus optionally the hidden
    layer the critic reads (the state *before* each token) or the entropy.

    `logits_to_keep` restricts the 248,320-wide output head to the scored span.
    """
    out = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        position_ids=padded_position_ids(attention_mask),
        logits_to_keep=completion_len + 1,
        output_hidden_states=hidden_layer is not None,
    )
    logits = out.logits[:, :-1]
    logprobs = selective_logprobs(logits, sequences[:, -completion_len:])
    entropy = token_entropy(logits) if want_entropy else None
    hidden = None
    if hidden_layer is not None:
        hidden = out.hidden_states[hidden_layer][:, -(completion_len + 1) : -1].detach()
    return logprobs, hidden, entropy


def load_policy(model_id: str, lora_r: int, device: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    policy = get_peft_model(
        base,
        LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM"),
    ).to(device)
    return policy, tokenizer


# --- rollouts -----------------------------------------------------------------


@dataclass
class Rollout:
    cases: list[dict[str, Any]]
    sequences: torch.Tensor  # (batch, prompt + completion), prompt left-padded
    attention_mask: torch.Tensor  # the prompt's padding, then the active completion
    mask: torch.Tensor  # (batch, completion): 1 up to and including the first <|im_end|>
    completions: list[str]
    rewards: list[float]
    parsed: list[bool]
    truncated: list[bool]


@torch.no_grad()
def rollout(policy, tokenizer, cases, *, max_new_tokens: int, temperature: float, weights: Weights) -> Rollout:
    """One sample per case. The completion span is cut to the longest active row:
    positions past every row's end carry no log-prob, reward or value, and at a
    248k vocabulary scoring them is the most expensive thing in the step."""
    device = next(policy.parameters()).device
    texts = [render_prompt(tokenizer, c["record"]) for c in cases]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    prompt_len = encoded["input_ids"].shape[1]
    out = policy.generate(
        **encoded, do_sample=True, temperature=temperature, top_p=1.0,
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
    )
    completion_ids = out[:, prompt_len:]
    mask = build_mask(completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    span = max(1, int(mask.sum(dim=-1).max()))
    completion_ids, mask = completion_ids[:, :span], mask[:, :span]
    completions = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    scored = [safe_score(c, case["answer"], weights) for c, case in zip(completions, cases)]
    return Rollout(
        cases=list(cases),
        sequences=out[:, : prompt_len + span],
        attention_mask=torch.cat([encoded["attention_mask"], mask.long()], dim=1),
        mask=mask,
        completions=completions,
        rewards=[s.total if s is not None else 0.0 for s in scored],
        parsed=[s is not None and s.gate_passed for s in scored],
        truncated=[bool((ids != tokenizer.eos_token_id).all()) for ids in out[:, prompt_len:]],
    )


def sampled_numeric(roll: Rollout) -> float:
    hits = []
    for text, case in zip(roll.completions, roll.cases):
        s = safe_score(text, case["answer"], ABLATE)
        hits.append(((s.diagnostics.get("numeric_correct", 0) or 0) / 3) if s is not None else 0.0)
    return sum(hits) / max(1, len(hits))


# --- held-out evaluation ------------------------------------------------------


def evaluate(policy, tokenizer, cases, *, weights: Weights, batch: int, max_tokens: int, seed: int):
    """Greedy over `cases` through `eval.generate_hf` / `eval.summarise`."""
    device = str(next(policy.parameters()).device)
    results = generate_hf(
        cases, model=MODEL, device=device, dtype="bfloat16", n=1, temperature=0.0,
        max_tokens=max_tokens, seed=seed, batch_size=batch, adapter=None,
        prompt_version=PROMPT_VERSION, loaded=(policy, tokenizer),
    )
    return results, summarise(results, weights)


EVAL_KEYS = ("reward", "exact_match", "validity_gate", "schema_ok", "cause_acc", "flags_acc",
             "numeric_acc", "action_acc", "completion_tokens_mean", "predicted_cause_hist")


# --- the run ------------------------------------------------------------------


@dataclass
class Config:
    model: str = MODEL
    #: The step budget. A run normally ends before it on `patience`.
    steps: int = 2000
    prompts_per_step: int = 8
    max_new_tokens: int = 640
    temperature: float = 1.0
    lr: float = 1e-5
    weights: str = "ABLATE"
    gamma: float = 1.0
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.005
    inner_epochs: int = 4
    micro_batch: int = 1
    lora_r: int = 16
    flat_credit_scale: float = 3.0
    #: The critic: which hidden layer it reads (an index into `hidden_states`),
    #: its learning rate, its starting value, and how it is fitted -- 8 Adam steps
    #: per batch over the active positions of the last 25 batches. The first three
    #: are measured for this model in the notebook, not carried over.
    value_layer: int = -1
    value_lr: float = 0.0
    value_init_bias: float = -1.0
    critic_epochs: int = 8
    critic_window: int = 25
    seed: int = 0
    split: str = "train"
    eval_every: int = 25
    eval_split: str = "dev"
    #: The first N cases of `eval_split`; 200 is all of dev.
    eval_cases: int = 200
    eval_batch: int = 32
    eval_max_tokens: int = 640
    #: Convergence: stop once this many evaluations in a row fail to beat the best
    #: held-out reward so far.
    patience: int = 8
    #: No-regression gate: stop if held-out reward is below step 0's by more than
    #: this at two evaluations in a row.
    regress_margin: float = 0.05
    #: Critic gate, from step `critic_gate_from` on: at every evaluation, the
    #: median of `value_ev - value_ev_position` over the last 25 steps must be
    #: above 0. Two failing evaluations in a row stop the run.
    critic_gate_from: int = 40

    def __post_init__(self) -> None:
        if self.weights not in WEIGHT_SETS:
            raise ValueError(f"unknown weights {self.weights!r}")
        if self.value_lr <= 0 or self.value_init_bias < 0:
            raise ValueError("value_lr and value_init_bias are measured in the notebook; pass them")


def _critic_window_stats(history: list[dict[str, Any]], n: int = 25) -> dict[str, float | None]:
    rows = [r for r in history[-n:] if r.get("value_ev") is not None and r.get("value_ev_position") is not None]
    if not rows:
        return {"critic_residual_median": None, "critic_ahead_frac": None}
    res = sorted(r["value_ev"] - r["value_ev_position"] for r in rows)
    mid = len(res) // 2
    median = res[mid] if len(res) % 2 else (res[mid - 1] + res[mid]) / 2
    return {"critic_residual_median": round(median, 6),
            "critic_ahead_frac": round(sum(x > 0 for x in res) / len(res), 4)}


def _save_state(path: Path, policy, value_head, optimizer, critic_optimizer, critic_history,
                rng: random.Random, gate: dict[str, Any], step: int) -> None:
    """Everything a continuation needs, so a resumed run is the same run: the
    adapter, the head, both optimisers' moments, the critic's window and every
    random state. Restoring weights alone is a restart, not a continuation."""
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path / "adapter")
    torch.save({
        "value_head": value_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "critic_history": list(critic_history),
        "python_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "gate": gate,
        "step": step,
    }, path / "state.pt")


def train(cfg: Config, run_dir: Path, *, resume: bool = False, progress=print) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = WEIGHT_SETS[cfg.weights]
    cases = load_cases(cfg.split)
    eval_cases = load_cases(cfg.eval_split)[: cfg.eval_cases]

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    policy, tokenizer = load_policy(cfg.model, cfg.lora_r, device)
    hidden_size = policy.get_base_model().config.hidden_size
    value_head = ValueHead(hidden_size, cfg.value_init_bias).to(device)

    actor_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(actor_params, lr=cfg.lr, weight_decay=0.0)
    critic_optimizer = torch.optim.AdamW(value_head.parameters(), lr=cfg.value_lr, weight_decay=0.0)
    critic_history: deque = deque(maxlen=cfg.critic_window)

    metrics_path, eval_path = run_dir / "metrics.jsonl", run_dir / "eval.jsonl"
    history: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    #: best: held-out reward; since_best: evaluations without a new best;
    #: regress / critic_fail: consecutive failing evaluations of each gate.
    gate: dict[str, Any] = {"best": None, "best_step": None, "since_best": 0, "regress": 0,
                            "critic_fail": 0, "step0_reward": None}
    start = 0

    if resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        # On the CPU: the RNG states must stay CPU ByteTensors; the rest is moved on load.
        state = torch.load(run_dir / "last" / "state.pt", map_location="cpu", weights_only=False)
        # In place, so the optimiser's references to the LoRA parameters stay valid.
        set_peft_model_state_dict(
            policy, load_file(str(run_dir / "last" / "adapter" / "adapter_model.safetensors"), device=device)
        )
        value_head.load_state_dict(state["value_head"])
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        critic_history.extend(state["critic_history"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        gate, start = state["gate"], state["step"]
        # Drop whatever the crashed process wrote after its last checkpoint.
        history = [r for r in map(json.loads, metrics_path.read_text().splitlines()) if r["step"] < start]
        evals = [r for r in map(json.loads, eval_path.read_text().splitlines()) if r["step"] <= start]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in history))
        eval_path.write_text("".join(json.dumps(r) + "\n" for r in evals))
        progress(f"resumed {run_dir.name} at step {start}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        metrics_path.write_text("")
        eval_path.write_text("")

    def run_eval(step: int) -> str | None:
        """Evaluate, checkpoint, and apply the three stopping rules. Returns the
        reason to stop, or None."""
        # generate_hf reseeds torch; keep the training stream where it was.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _, m = evaluate(policy, tokenizer, eval_cases, weights=weights, batch=cfg.eval_batch,
                        max_tokens=cfg.eval_max_tokens, seed=cfg.seed)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

        record = {"step": step, **{k: m[k] for k in EVAL_KEYS}, **_critic_window_stats(history)}
        reason = None
        if step == 0:
            gate["step0_reward"] = record["reward"]
        if gate["best"] is None or record["reward"] > gate["best"]:
            gate.update(best=record["reward"], best_step=step, since_best=0)
            policy.save_pretrained(run_dir / "adapter-best")
            torch.save(value_head.state_dict(), run_dir / "value_head-best.pt")
            (run_dir / "best.json").write_text(json.dumps(record, indent=2) + "\n")
        else:
            gate["since_best"] += 1
            if gate["since_best"] >= cfg.patience:
                reason = (f"converged: {cfg.patience} evaluations without beating held-out reward "
                          f"{gate['best']:.4f} (step {gate['best_step']})")
        gate["regress"] = gate["regress"] + 1 if record["reward"] < gate["step0_reward"] - cfg.regress_margin else 0
        if gate["regress"] >= 2:
            reason = f"no-regression gate: held-out reward {record['reward']:.4f} below step 0 twice"
        if step >= cfg.critic_gate_from and record["critic_residual_median"] is not None:
            gate["critic_fail"] = gate["critic_fail"] + 1 if record["critic_residual_median"] <= 0 else 0
            if gate["critic_fail"] >= 2:
                reason = "critic gate: value_ev not above the position clock at two evaluations"
        record["gate"] = dict(gate)
        evals.append(record)
        with eval_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        if step:
            _save_state(run_dir / "last", policy, value_head, optimizer, critic_optimizer,
                        critic_history, rng, gate, step)
        crit = record["critic_residual_median"]
        progress(f"    eval @{step:<5} reward {record['reward']:.4f} EM {record['exact_match']:.3f} "
                 f"cause {record['cause_acc']:.3f} flags {record['flags_acc']:.3f} "
                 f"numeric {record['numeric_acc']:.3f} schema {record['schema_ok']:.3f} "
                 f"critic {'-' if crit is None else f'{crit:+.4f}'} | best {gate['best']:.4f}@{gate['best_step']}")
        return reason

    progress(f"{cfg.model} | ABLATE | gamma {cfg.gamma} lam {cfg.lam} | value layer {cfg.value_layer} "
             f"lr {cfg.value_lr:.3g} bias {cfg.value_init_bias:.4f} | {device}")
    stop = run_eval(0) if (start == 0 and cfg.eval_every) else None

    step = start
    while stop is None and step < cfg.steps:
        t0 = time.perf_counter()
        batch = [rng.choice(cases) for _ in range(cfg.prompts_per_step)]
        policy.eval()
        roll = rollout(policy, tokenizer, batch, max_new_tokens=cfg.max_new_tokens,
                       temperature=cfg.temperature, weights=weights)
        gen_seconds = time.perf_counter() - t0
        seqs, attn, mask = roll.sequences, roll.attention_mask, roll.mask
        L = mask.shape[1]
        rewards = dense_rewards(roll.completions, seqs[:, -L:], mask, roll.cases, weights, tokenizer,
                                cfg.flat_credit_scale)

        # Old log-probs and the critic's features, one no-grad pass.
        policy.train()
        with torch.no_grad():
            old_lp, old_h = [], []
            for i in range(0, len(seqs), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                lp, h, _ = forward_logprobs(policy, seqs[sl], attn[sl], L, hidden_layer=cfg.value_layer)
                old_lp.append(lp)
                old_h.append(h)
            old_logprobs, hidden = torch.cat(old_lp), torch.cat(old_h)
            old_values = value_head(hidden) * mask
            advantages, returns = gae(rewards, old_values, mask, gamma=cfg.gamma, lam=cfg.lam)

        # The critic, on cached features: no forward through the trunk.
        sel = mask > 0
        critic_history.append((hidden[sel].cpu(), old_values[sel].cpu(), returns[sel].cpu()))
        fit_h = torch.cat([h for h, _, _ in critic_history]).to(device)
        fit_v = torch.cat([v for _, v, _ in critic_history]).to(device)
        fit_r = torch.cat([r for _, _, r in critic_history]).to(device)
        critic_stats: dict[str, float] = {}
        for _ in range(cfg.critic_epochs):
            critic_optimizer.zero_grad(set_to_none=True)
            v_loss, critic_stats = value_loss(value_head(fit_h), fit_v, fit_r, clip_eps=cfg.value_clip_eps)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
            critic_optimizer.step()
        with torch.no_grad():
            fitted_values = value_head(hidden) * mask
        del fit_h, fit_v, fit_r

        # The actor.
        actor_stats: dict[str, float] = {}
        grad_norm = torch.zeros(())
        for _ in range(cfg.inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            epoch_stats: dict[str, float] = {}
            for i in range(0, len(seqs), cfg.micro_batch):
                sl = slice(i, i + cfg.micro_batch)
                share = min(cfg.micro_batch, len(seqs) - i) / len(seqs)
                lp, _, ent = forward_logprobs(policy, seqs[sl], attn[sl], L, want_entropy=cfg.entropy_coef > 0)
                loss, stats = ppo_actor_loss(lp, old_logprobs[sl], advantages[sl], mask[sl],
                                             clip_eps=cfg.clip_eps, entropy=ent, entropy_coef=cfg.entropy_coef)
                (loss * share).backward()
                for k, v in stats.items():
                    epoch_stats[k] = epoch_stats.get(k, 0.0) + v * share
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            optimizer.step()
            actor_stats = epoch_stats

        record = {
            "step": step,
            "reward_mean": sum(roll.rewards) / len(roll.rewards),
            "parsed_frac": sum(roll.parsed) / len(roll.parsed),
            "truncated_frac": sum(roll.truncated) / len(roll.truncated),
            "train_numeric_acc": round(sampled_numeric(roll), 6),
            **critic_fit(old_values, returns, mask),
            **{f"{k}_fit": v for k, v in critic_fit(fitted_values, returns, mask).items()},
            **{k: round(v, 6) for k, v in actor_stats.items()},
            **{k: round(v, 6) for k, v in critic_stats.items()},
            "completion_tokens": float(mask.sum() / mask.shape[0]),
            "grad_norm": float(grad_norm),
            "gen_seconds": round(gen_seconds, 2),
            "step_seconds": round(time.perf_counter() - t0, 2),
        }
        record["value_step"] = round(abs(record["value_mean"] - history[-1]["value_mean"]), 6) if history else 0.0
        history.append(record)
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        ev = record["value_ev"]
        progress(f"  step {step:>5} reward {record['reward_mean']:.4f} V {record['value_mean']:.3f} "
                 f"ev {'undef' if ev is None else f'{ev:+.3f}'} clock "
                 f"{'undef' if record['value_ev_position'] is None else format(record['value_ev_position'], '+.3f')} "
                 f"adv {record['adv_mean']:+.4f} clip {record['clip_frac']:.3f} H {record.get('policy_entropy', 0):.3f} "
                 f"tok {record['completion_tokens']:.0f} parsed {record['parsed_frac']:.2f} "
                 f"|g| {record['grad_norm']:.2f} {record['step_seconds']:.1f}s")
        step += 1
        if cfg.eval_every and step % cfg.eval_every == 0:
            stop = run_eval(step)

    policy.save_pretrained(run_dir / "adapter")
    torch.save(value_head.state_dict(), run_dir / "value_head.pt")
    ending = {"stopped_at": step, "reason": stop or f"step budget {cfg.steps} reached",
              "best_step": gate["best_step"], "best_reward": gate["best"]}
    (run_dir / "ending.json").write_text(json.dumps(ending, indent=2) + "\n")
    progress(f"END {ending}")
    return ending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help=f"run name; written to {RUNS.name}/<run>")
    parser.add_argument("--resume", action="store_true", help="continue <run> from its last/ checkpoint")
    parser.add_argument("--max-vram-mib", type=int, default=0, help="cap this process's CUDA allocator")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        parser.add_argument(flag, type=type(f.default), default=f.default)
    args = parser.parse_args()
    if args.max_vram_mib and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_mib * 2**20 / total))
    run_dir = RUNS / args.run
    if args.resume:
        cfg = Config(**json.loads((run_dir / "config.json").read_text()))
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"{run_dir} exists; pick another --run or pass --resume")
        cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    train(cfg, run_dir, resume=args.resume, progress=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
