"""GRPO written out by hand: prompt, sampled group, deterministic reward, update.

`../toy_mdp/ppo.py` derives the clipped surrogate's gradient by hand on a tabular
policy, because at that size the derivation *is* the idea. The same derivation
survives the move to a language model — what changes is only that the policy is
now a network and autograd carries the chain rule the rest of the way. So the
gradient with respect to the log-probabilities is still written out below, and
`test_grpo_scratch.py` checks autograd against it.

## The algorithm, in the order the code runs

1. **Sample a group.** G completions for one prompt, from the current policy.
2. **Score them.** `reward.score` and nothing else. Deterministic, no judge.
3. **Centre within the group.** `A_i = (r_i - mean r) / std r`. This is the whole
   trick that lets GRPO drop the value network: the other G-1 samples *are* the
   baseline. When a group's rewards are identical the advantages are exactly
   zero and the group contributes no gradient, which is a real event on this
   task and is counted rather than smoothed away.
4. **Update.** A PPO-style clipped surrogate on the per-token ratio.

## The gradient

With `rho = exp(logp - logp_old)` and a per-sequence advantage `A`:

    per_token_objective = min(rho * A, clip(rho, 1-eps, 1+eps) * A)

Differentiating with respect to `logp`, and using `d rho / d logp = rho`:

    d/d logp  =  rho * A     unless the clip is active *and* binding
                 0           when A > 0 and rho > 1+eps
                 0           when A < 0 and rho < 1-eps

That is the same three-case result as `toy_mdp/ppo.py`: full gradient inside the
trust region, zero once the ratio has been pushed past the boundary in the
direction the advantage favours. Note that at `rho = 1` it reduces to `A`, the
vanilla policy gradient — so with one inner epoch the clip is inert by
construction and GRPO here *is* REINFORCE with a group baseline. The clip only
starts doing work at `--inner-epochs 2` and above, when the batch is being
reused and the policy has drifted from the one that generated it.

## Two choices worth arguing about

**Length normalisation.** Dividing each sequence's token sum by its own length
(`--normalize sequence`, the original formulation) weights short completions
more heavily per token and is a known source of length drift. Summing over all
tokens and dividing once (`--normalize token`) does not. Completion length is a
tracked metric in this experiment rather than a nuisance, so the default is
`token` — the reward is not allowed a length term either, for the same reason.

**KL estimator.** `exp(d) - d - 1` where `d = logp_ref - logp`, which is
non-negative by construction and lower variance than `-d`. With `--beta 0` the
reference model is never run at all.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from reward import MAIN, PROBE, Weights, group_advantages, score
from task.prompt import PROMPT_VERSION, build_messages

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def selective_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log P(target) per position, without materialising a full log-softmax.

    The obvious `log_softmax(...).gather(...)` allocates two more tensors the
    size of the logits -- the softmax output, which autograd saves for the
    backward pass, and its gradient. Against Qwen2.5-0.5B's 151,936-token
    vocabulary those are the largest tensors in the step by a wide margin, and
    they are what exhausted a 16 GB card on the first attempt. Writing it as
    `chosen - logsumexp` keeps the reduction at (batch, positions).
    """
    logits = logits.float()
    chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(logits, dim=-1)


def completion_logprobs(
    model, sequences: torch.Tensor, completion_len: int
) -> torch.Tensor:
    """Per-token log-probs over the completion span of each sequence.

    `logits_to_keep` restricts the output head to the positions that are
    actually scored. Prompt tokens are given rather than sampled, contribute no
    policy-gradient term, and their logits are never read -- so computing them
    would be both wasteful and, at this vocabulary size, fatal.
    """
    logits = model(input_ids=sequences, logits_to_keep=completion_len + 1).logits[:, :-1]
    return selective_logprobs(logits, sequences[:, -completion_len:])


def grpo_surrogate(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    ref_logprobs: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    beta: float = 0.0,
    normalize: str = "token",
) -> tuple[torch.Tensor, dict[str, float]]:
    """The clipped surrogate, plus the diagnostics the run is judged on.

    Shapes: `logprobs`, `old_logprobs`, `mask` are (batch, tokens);
    `advantages` is (batch,), one per sampled sequence.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    advantage = advantages.unsqueeze(-1)

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    objective = torch.min(unclipped, clipped)

    if beta > 0.0 and ref_logprobs is not None:
        delta = ref_logprobs - logprobs
        kl = torch.exp(delta) - delta - 1.0  # k3: non-negative by construction
        objective = objective - beta * kl
    else:
        kl = torch.zeros_like(objective)

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
        # A token is "clipped" only when the clip is also *binding* -- when it
        # changed which branch the min() selected. A ratio outside the band that
        # does not bind still passes its full gradient through.
        binding = ((unclipped > clipped) & (mask > 0)).float().sum() / active
        stats = {
            "loss": float(loss.detach()),
            "ratio_mean": float((ratio * mask).sum() / active),
            "clip_frac": float(binding),
            "kl": float((kl * mask).sum() / active) if beta > 0 else 0.0,
            "entropy_proxy": float(-(logprobs * mask).sum() / active),
        }
    return loss, stats


# --- rollout ------------------------------------------------------------------


@dataclass
class Group:
    case: dict[str, Any]
    sequences: torch.Tensor  # (G, prompt_len + completion_len)
    mask: torch.Tensor  # (G, completion_len)
    completions: list[str]
    rewards: list[float]
    advantages: list[float]
    degenerate: bool


def build_mask(completion_ids: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    """Score up to and including the first EOS; never score padding.

    The two are not the same event even though they often share an id. EOS is a
    token the policy *chose* — deciding to stop is an action, and an action that
    is never scored is an action RL cannot learn to take. Padding is an artefact
    of making the batch rectangular and was never sampled at all.

    Qwen2.5 has no distinct pad token, so `load_policy` aliases pad to EOS and
    the distinction collapses to "score the first one". The code keeps them
    apart anyway, because a tokenizer where they differ would otherwise have its
    first pad token silently trained on.
    """
    is_eos = completion_ids == eos_id
    is_pad = completion_ids == pad_id
    terminal = is_eos | is_pad
    # positions with no terminator strictly before them
    active = (terminal.cumsum(dim=-1) - terminal.long()) == 0
    return (active & ~(is_pad & ~is_eos)).float()


@torch.no_grad()
def rollout(
    policy,
    tokenizer,
    case: dict[str, Any],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    weights: Weights,
    device: str,
) -> Group:
    """Sample one group, score it, and centre the rewards within the group."""
    text = tokenizer.apply_chat_template(
        build_messages(case["record"]), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer([text] * group_size, return_tensors="pt", padding=True).to(device)
    prompt_len = encoded["input_ids"].shape[1]

    out = policy.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Pad every group to the same width so the update sees a rectangular batch.
    if out.shape[1] < prompt_len + max_new_tokens:
        pad = out.new_full(
            (out.shape[0], prompt_len + max_new_tokens - out.shape[1]), tokenizer.pad_token_id
        )
        out = torch.cat([out, pad], dim=1)

    completion_ids = out[:, prompt_len:]
    mask = build_mask(completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    rewards = [score(c, case["answer"], weights).total for c in completions]
    advantages, degenerate = group_advantages(rewards)

    return Group(
        case=case,
        sequences=out,
        mask=mask,
        completions=completions,
        rewards=rewards,
        advantages=advantages,
        degenerate=degenerate,
    )


# --- training -----------------------------------------------------------------


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    steps: int = 100
    prompts_per_step: int = 4
    group_size: int = 8
    max_new_tokens: int = 640
    temperature: float = 1.0
    lr: float = 1e-5
    # Zero, deliberately. AdamW's default 0.01 decays the adapter even on steps
    # where the gradient is exactly zero -- which happens whenever a whole batch
    # of groups is degenerate, and 16% of groups are at the frozen baseline. In
    # an experiment whose entire question is what the reward moved, a force that
    # shrinks the policy independently of the reward is a confound, not a
    # regulariser. Found by a test asserting a degenerate group is a no-op.
    weight_decay: float = 0.0
    clip_eps: float = 0.2
    beta: float = 0.0
    inner_epochs: int = 1
    micro_batch: int = 2
    normalize: str = "token"
    lora_r: int = 16
    weights: str = "MAIN"
    seed: int = 0
    dtype: str = "bfloat16"
    split: str = "train"
    prompt_version: str = PROMPT_VERSION
    # Held-out evaluation during training. Without it the run produces a reward
    # curve and nothing to read it against, and "reward rose" cannot be
    # separated from "the policy got better" -- which is the question.
    eval_every: int = 25
    # The full dev split, so step 0 is directly comparable to the frozen
    # baseline rather than to a subset of it with a different denominator.
    eval_cases: int = 200
    eval_batch: int = 32
    eval_split: str = "dev"
    eval_max_tokens: int = 640


def load_policy(cfg: Config, device: str):
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
    return policy, tokenizer


def evaluate(policy, tokenizer, cfg: Config, cases: list[dict], step: int) -> dict[str, Any]:
    """Greedy pass over a fixed held-out slice, through eval.py's own code path.

    Fixed slice and fixed decoding, so successive evaluations differ only by the
    policy. Comparable to the frozen baseline because it is literally the same
    function that produced it.
    """
    from eval import generate_hf, summarise

    weights = {"MAIN": MAIN, "PROBE": PROBE}[cfg.weights]
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
    metrics = summarise(results, weights)
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


def train(cfg: Config, out_dir: Path, device: str) -> None:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    cases = [json.loads(line) for line in (DATA / f"{cfg.split}.jsonl").read_text().splitlines()]
    weights = {"MAIN": MAIN, "PROBE": PROBE}[cfg.weights]

    policy, tokenizer = load_policy(cfg, device)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr,
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

    def run_eval(step: int) -> None:
        record = evaluate(policy, tokenizer, cfg, eval_cases, step)
        with eval_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"    eval @{step:<4} reward {record['reward']:.4f} "
            f"cause {record['cause_acc']:.3f} flags {record['flags_acc']:.3f} "
            f"EM {record['exact_match']:.3f} valid {record['validity_gate']:.3f}"
        )

    print(f"{cfg.model} | {cfg.steps} steps | {cfg.prompts_per_step}x{cfg.group_size} | {device}")
    if cfg.eval_every:
        run_eval(0)  # step 0 must reproduce the frozen baseline

    for step in range(cfg.steps):
        started = time.perf_counter()
        batch = [rng.choice(cases) for _ in range(cfg.prompts_per_step)]

        policy.eval()
        groups = [
            rollout(
                policy,
                tokenizer,
                case,
                group_size=cfg.group_size,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                weights=weights,
                device=device,
            )
            for case in batch
        ]
        gen_seconds = time.perf_counter() - started

        # Groups are kept apart rather than concatenated. Different prompts
        # tokenize to different lengths, so one batch would need padding -- and
        # a padded batch through `model(input_ids=...)` with no attention mask
        # attends to the pad tokens and starts RoPE at the pad, silently
        # corrupting the very log-probs that are the training signal. Within a
        # group there is no padding at all, because a group is G samples of one
        # prompt. That is a structural convenience of GRPO worth not throwing
        # away for the sake of one `torch.cat`.
        total_sequences = sum(len(g.rewards) for g in groups)

        def micro_batches(group: Group):
            for i in range(0, group.sequences.shape[0], cfg.micro_batch):
                yield slice(i, i + cfg.micro_batch)

        policy.train()
        with torch.no_grad():
            old_by_group = [
                torch.cat(
                    [
                        completion_logprobs(policy, g.sequences[sl], g.mask.shape[1])
                        for sl in micro_batches(g)
                    ]
                )
                for g in groups
            ]
            ref_by_group: list[torch.Tensor] | None = None
            if cfg.beta > 0:
                # The reference policy is the base model: disabling the adapters
                # gives it for free, with no second copy of the weights.
                with policy.disable_adapter():
                    ref_by_group = [
                        torch.cat(
                            [
                                completion_logprobs(policy, g.sequences[sl], g.mask.shape[1])
                                for sl in micro_batches(g)
                            ]
                        )
                        for g in groups
                    ]

        step_stats: dict[str, float] = {}
        for _ in range(cfg.inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for gi, g in enumerate(groups):
                completion_len = g.mask.shape[1]
                advantages = torch.tensor(g.advantages, device=device, dtype=torch.float32)
                for sl in micro_batches(g):
                    logprobs = completion_logprobs(policy, g.sequences[sl], completion_len)
                    loss, stats = grpo_surrogate(
                        logprobs,
                        old_by_group[gi][sl],
                        advantages[sl],
                        g.mask[sl],
                        ref_logprobs=None if ref_by_group is None else ref_by_group[gi][sl],
                        clip_eps=cfg.clip_eps,
                        beta=cfg.beta,
                        normalize=cfg.normalize,
                    )
                    # Each micro-batch carries an equal share of the step.
                    share = logprobs.shape[0] / total_sequences
                    (loss * share).backward()
                    accumulated += stats["loss"] * share
                    step_stats = stats
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        rewards = [r for g in groups for r in g.rewards]
        record = {
            "step": step,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "adv_zero_frac": sum(g.degenerate for g in groups) / len(groups),
            "completion_tokens": float(
                sum(float(g.mask.sum()) for g in groups) / total_sequences
            ),
            "unique_completions": sum(len(set(g.completions)) for g in groups) / len(groups),
            "loss": accumulated,
            "grad_norm": float(grad_norm),
            "clip_frac": step_stats.get("clip_frac", 0.0),
            "ratio_mean": step_stats.get("ratio_mean", 1.0),
            "kl": step_stats.get("kl", 0.0),
            "gen_seconds": round(gen_seconds, 2),
            "step_seconds": round(time.perf_counter() - started, 2),
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"  step {step:>4} reward {record['reward_mean']:.4f} "
            f"advzero {record['adv_zero_frac']:.2f} uniq {record['unique_completions']:.1f} "
            f"tok {record['completion_tokens']:.0f} clip {record['clip_frac']:.3f} "
            f"{record['step_seconds']:.1f}s"
        )

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            run_eval(step + 1)

    policy.save_pretrained(out_dir / "adapter")
    print(f"\nwrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in asdict(Config()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = Config(**{k: v for k, v in vars(args).items() if k in asdict(Config())})
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    out = args.out or ROOT / "runs" / f"grpo-{cfg.weights.lower()}-s{cfg.seed}"
    train(cfg, out, device)


if __name__ == "__main__":
    main()
