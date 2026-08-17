"""Measure what one GRPO step will cost on this machine, before committing to a run.

The step budget for this experiment is not a round number picked in advance --
it is whatever the measured throughput allows overnight. That means measuring
three things on the actual prompts, not on synthetic filler:

1. **Prompt length.** Sets `max_prompt_length` and decides how much of each step
   goes into prefill that no one reads.
2. **Batched decode throughput.** GRPO samples a group of G completions per
   prompt, so the useful question is tokens/second at the batch sizes a group
   actually produces, not at batch 1.
3. **Forward + backward on a LoRA policy.** The half of the step that generation
   timings alone would leave out.

Everything is written to `runs/probe/throughput.json` so the README's numbers
have a file behind them.

    .venv/bin/python probe_throughput.py
    .venv/bin/python probe_throughput.py --batch-sizes 1,4,8,16 --max-new-tokens 256
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task.prompt import PROMPT_VERSION, build_messages

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device: str) -> None:
    """Generation is queued asynchronously; timing it without this measures nothing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def release(device: str) -> None:
    """Drop the cached allocations, so one measurement does not OOM the next."""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def pool_memory_gib(device: str) -> float:
    """Allocator pool size, *not* a true peak.

    On MPS `driver_allocated_memory` reports what the allocator holds, which
    includes cached blocks from earlier measurements and never shrinks mid-run.
    It is useful for spotting the batch size where things fall over and useless
    as an estimate of what a single step actually needs.
    """
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return float("nan")


def load_cases(limit: int) -> list[dict]:
    path = ROOT / "data" / "dev.jsonl"
    if not path.exists():
        raise SystemExit("no dev split; run `python3 -m task.generate` first")
    return [json.loads(line) for line in path.read_text().splitlines()[:limit]]


def measure_prompt_lengths(tokenizer, cases: list[dict]) -> dict[str, Any]:
    lengths = []
    for case in cases:
        text = tokenizer.apply_chat_template(
            build_messages(case["record"]), tokenize=False, add_generation_prompt=True
        )
        lengths.append(len(tokenizer(text)["input_ids"]))
    lengths.sort()
    return {
        "n": len(lengths),
        "min": lengths[0],
        "p50": lengths[len(lengths) // 2],
        "p95": lengths[int(len(lengths) * 0.95)],
        "max": lengths[-1],
    }


def time_generation(
    model, tokenizer, text: str, batch: int, max_new_tokens: int, device: str
) -> dict[str, Any]:
    """Wall time for one group of `batch` completions from a single prompt.

    `min_new_tokens` is pinned to `max_new_tokens` so every sequence runs the
    full length. Without it an early EOS makes a slow machine look fast.
    """
    encoded = tokenizer([text] * batch, return_tensors="pt", padding=True).to(device)

    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **encoded,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            min_new_tokens=max_new_tokens,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    sync(device)
    elapsed = time.perf_counter() - started

    generated = int((out.shape[1] - encoded["input_ids"].shape[1]) * batch)
    return {
        "batch": batch,
        "seconds": round(elapsed, 3),
        "generated_tokens": generated,
        "tokens_per_second": round(generated / elapsed, 1),
        "seconds_per_sequence": round(elapsed / batch, 3),
    }


def selective_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log P(target) per position, without materialising a full log-softmax.

    The obvious `log_softmax(...).gather(...)` allocates two more tensors the
    size of the logits -- the softmax output, which autograd saves for the
    backward pass, and its gradient. Against a 151,936-token vocabulary those
    are the largest tensors in the step by a wide margin. Writing it as
    `chosen - logsumexp` keeps the reduction at (batch, positions) and leaves
    only the logits themselves full-size.
    """
    logits = logits.float()
    chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(logits, dim=-1)


def build_policy(model):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )


def time_train_step(
    policy, tokenizer, batch: int, prompt_len: int, completion_len: int, device: str
) -> dict:
    """Forward + backward for one GRPO-shaped update, on a LoRA policy.

    `logits_to_keep` is the whole trick. Qwen2.5-0.5B carries a 151,936-token
    vocabulary on an 896-dimensional hidden state, so the logits tensor is far
    larger than the model: at batch 4 over a full 1,096-token sequence it is
    1.2 GiB in bf16, roughly 2.5 GiB once cross-entropy upcasts it, and the
    backward pass needs a gradient buffer the same size again. That is what
    exhausted a 16 GB machine on the first attempt.

    Restricting the head to completion positions is not a memory trick applied
    after the fact -- it is what the algorithm wants. Prompt tokens are given,
    not sampled, so they contribute no policy-gradient term and their logits are
    never read.
    """
    total_len = prompt_len + completion_len
    ids = torch.randint(0, tokenizer.vocab_size, (batch, total_len), device=device)
    advantages = torch.randn(batch, device=device)

    sync(device)
    started = time.perf_counter()

    logits = policy(input_ids=ids, logits_to_keep=completion_len + 1).logits[:, :-1]
    chosen = selective_logprobs(logits, ids[:, -completion_len:])
    # Stand-in for the GRPO surrogate: same shapes, same graph, no ratio or clip.
    loss = -(chosen.mean(dim=-1) * advantages).mean()
    loss.backward()

    sync(device)
    elapsed = time.perf_counter() - started

    policy.zero_grad(set_to_none=True)
    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "completion_len": completion_len,
        "seconds": round(elapsed, 3),
        "pool_gib": round(pool_memory_gib(device), 2),
    }


def estimate_step(
    generation: list[dict], train: list[dict], prompts_per_step: int, group_size: int
) -> dict:
    """Project a GRPO step from the measured pieces.

    Generation and the update are sized independently, because they are: the
    update carries a backward pass and runs out of memory several batch sizes
    before sampling does. Both phases get the largest batch that was measured to
    work, which is what the trainer will actually do.
    """
    sequences = prompts_per_step * group_size
    feasible_train = [row for row in train if "seconds" in row]
    if not feasible_train:
        raise SystemExit("no feasible update batch size; lower --max-new-tokens")

    gen = min(generation, key=lambda row: row["seconds"] / row["batch"])
    update = min(feasible_train, key=lambda row: row["seconds"] / row["batch"])

    gen_seconds = -(-sequences // gen["batch"]) * gen["seconds"]
    update_seconds = -(-sequences // update["batch"]) * update["seconds"]
    total = gen_seconds + update_seconds

    return {
        "prompts_per_step": prompts_per_step,
        "group_size": group_size,
        "sequences_per_step": sequences,
        "generate_batch": gen["batch"],
        "update_batch": update["batch"],
        "generate_seconds": round(gen_seconds, 1),
        "update_seconds": round(update_seconds, 1),
        "step_seconds": round(total, 1),
        "steps_per_hour": round(3600 / total, 1),
        "steps_in_8_hours": int(8 * 3600 / total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--train-batch-sizes", default="1,2,4")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sample-prompts", type=int, default=64)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "probe" / "throughput.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    train_batch_sizes = [int(b) for b in args.train_batch_sizes.split(",")]

    print(f"device={device} dtype={args.dtype} model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    cases = load_cases(args.sample_prompts)
    lengths = measure_prompt_lengths(tokenizer, cases)
    print(
        f"\nprompt tokens over {lengths['n']} dev cases: "
        f"min={lengths['min']} p50={lengths['p50']} p95={lengths['p95']} max={lengths['max']}"
    )

    text = tokenizer.apply_chat_template(
        build_messages(cases[0]["record"]), tokenize=False, add_generation_prompt=True
    )

    print(f"\nwarming up ({args.max_new_tokens} new tokens)...")
    time_generation(model, tokenizer, text, 1, 8, device)

    generation = []
    print(f"\ngeneration\n{'batch':>6} {'seconds':>9} {'tok/s':>9} {'s/seq':>8}")
    for batch in batch_sizes:
        release(device)
        row = time_generation(model, tokenizer, text, batch, args.max_new_tokens, device)
        generation.append(row)
        print(
            f"{row['batch']:>6} {row['seconds']:>9.2f} "
            f"{row['tokens_per_second']:>9.1f} {row['seconds_per_sequence']:>8.2f}"
        )

    commit = getattr(model.config, "_commit_hash", None)
    del model
    release(device)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.grad_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    policy = build_policy(base)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    train = []
    print(
        f"\nforward+backward, LoRA r=16 ({trainable:,} trainable params), "
        f"grad_checkpointing={args.grad_checkpointing}"
    )
    print(f"{'batch':>6} {'seconds':>9} {'pool GiB':>10}")
    for batch in train_batch_sizes:
        release(device)
        try:
            row = time_train_step(
                policy, tokenizer, batch, lengths["p95"], args.max_new_tokens, device
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc):
                raise
            print(f"{batch:>6} {'OOM':>9}")
            train.append({"batch": batch, "oom": True})
            release(device)
            continue
        row["trainable_params"] = trainable
        train.append(row)
        print(f"{row['batch']:>6} {row['seconds']:>9.2f} {row['pool_gib']:>10.2f}")

    est = estimate_step(generation, train, args.prompts_per_step, args.group_size)
    print(
        f"\nprojected GRPO step: {est['prompts_per_step']} prompts x {est['group_size']} "
        f"completions = {est['sequences_per_step']} sequences"
    )
    print(f"  generate (batch {est['generate_batch']}): {est['generate_seconds']:>7.1f} s")
    print(f"  update   (batch {est['update_batch']}): {est['update_seconds']:>7.1f} s")
    print(f"  step                : {est['step_seconds']:>7.1f} s")
    print(f"  -> {est['steps_per_hour']:.1f} steps/hour, {est['steps_in_8_hours']} in 8 hours")

    payload = {
        "model": args.model,
        "model_commit": commit,
        "prompt_version": PROMPT_VERSION,
        "device": device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "prompt_tokens": lengths,
        "generation": generation,
        "train_step": train,
        "grad_checkpointing": args.grad_checkpointing,
        "step_estimate": est,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
