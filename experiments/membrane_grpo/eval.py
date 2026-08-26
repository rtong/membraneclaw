"""Run a model against the frozen splits and report what it can actually do.

This is the first place in the project where a model is asked to answer the
question rather than to act as a stopwatch load. Everything before it -- the
answer key, the generator, the reward, the throughput probes -- was built so
that this file could produce numbers worth trusting.

## Two backends, on purpose

`--backend openai` talks to an OpenAI-compatible endpoint over HTTP and needs
nothing installed beyond the standard library. It is how the 9B production
server on `anton` gets measured without loading anything onto the GPU.

`--backend hf` loads weights locally through transformers. It is how the 0.5B
policy gets measured, before and after training.

## Greedy and sampled are different questions

`--mode greedy` decodes at temperature 0 and reports pass@1: what the model does
when asked once. `--mode sample` draws `k` completions at temperature 1 and
reports pass@k along with the diversity of what came back.

Both are needed, and the gap between them is the point. GRPO tends to raise
pass@1 by concentrating probability on answers the base model could already
produce, which shows up as pass@k staying flat or falling while pass@1 climbs.
Measuring only pass@1 would record that as pure improvement.

## Reproducibility

Every run records the model, the prompt version, the sampling parameters, the
seeds, and the split checksum. A result whose prompt version does not match the
one it is being compared against is not a comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import reward as reward_module
from reward import score
from task.prompt import PROMPT_VERSION, VERSIONS, build_messages
from task.schema import canonical, parse_answer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


@dataclass
class Sample:
    case_id: str
    completion: str
    completion_tokens: int | None = None


@dataclass
class CaseResult:
    case: dict[str, Any]
    samples: list[Sample] = field(default_factory=list)


# --- backends -----------------------------------------------------------------


def _post_chat(base_url: str, api_key: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def generate_openai(
    cases: list[dict],
    *,
    model: str,
    base_url: str,
    api_key: str,
    n: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    concurrency: int,
    timeout: float,
    thinking: bool,
    prompt_version: str = PROMPT_VERSION,
) -> list[CaseResult]:
    """Fan out one request per case, each asking for `n` completions.

    vLLM batches server-side far better than this client could, so the only
    thing concurrency buys is keeping its queue fed.
    """

    def run(case: dict) -> CaseResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": build_messages(case["record"], prompt_version),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "seed": seed,
        }
        if temperature > 0:
            payload["top_p"] = 1.0
        # Qwen3-family servers started with a reasoning parser will otherwise
        # spend the whole budget inside <think> and return empty content --
        # see agent/config.py in the parent repo, which hit exactly this.
        if not thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            body = _post_chat(base_url, api_key, payload, timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            return CaseResult(case, [Sample(case["id"], "", None) for _ in range(n)]), exc

        usage = body.get("usage") or {}
        per_completion = (usage.get("completion_tokens") or 0) // max(len(body["choices"]), 1)
        samples = [
            Sample(
                case["id"],
                (choice["message"].get("content") or ""),
                per_completion or None,
            )
            for choice in body["choices"]
        ]
        return CaseResult(case, samples), None

    results: list[CaseResult] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for result, exc in pool.map(run, cases):
            results.append(result)
            if exc is not None:
                errors.append(f"{result.case['id']}: {exc}")

    if errors:
        print(f"  !! {len(errors)} request(s) failed, e.g. {errors[0]}", file=sys.stderr)
    return results


def supports_thinking_toggle(tokenizer) -> bool:
    """Whether this tokenizer's chat template honours `enable_thinking`.

    Qwen3 templates default it to *on*, and on this task that is fatal rather
    than merely verbose: measured on the 9B, the model spends its entire budget
    inside <think> and returns empty content, so every case fails the reward
    gate with `empty`. Qwen2.5 templates have no such variable.

    Detected rather than hard-coded by model name, and the result is recorded in
    the run's output — silently getting this wrong produces a baseline that
    looks like a capability floor and is actually a formatting artefact.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in template


def generate_hf(
    cases: list[dict],
    *,
    model: str,
    device: str,
    dtype: str,
    n: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    batch_size: int,
    adapter: str | None,
    prompt_version: str = PROMPT_VERSION,
    loaded: tuple | None = None,
    thinking: bool = False,
) -> list[CaseResult]:
    """Local decoding. Imported lazily so the HTTP path stays dependency-free.

    `loaded` takes an already-constructed `(policy, tokenizer)` so a training
    loop can evaluate its live policy mid-run. That path exists specifically so
    the mid-run numbers come from *this* function rather than a reimplementation
    of it -- a held-out curve measured by different code than the frozen
    baseline is not comparable to it, which would cost the comparison the whole
    point.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if loaded is not None:
        policy, tokenizer = loaded
        was_training = policy.training
    else:
        was_training = False
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(device)
        if adapter:
            from peft import PeftModel

            policy = PeftModel.from_pretrained(policy, adapter)
    policy.eval()

    torch.manual_seed(seed)

    template_kwargs: dict[str, Any] = {}
    if supports_thinking_toggle(tokenizer):
        template_kwargs["enable_thinking"] = thinking
        print(
            f"  chat template honours enable_thinking; set to {thinking}",
            file=sys.stderr,
        )

    # One case contributes n sequences; pack whole cases into a batch so a
    # group is never split across two generate calls.
    per_batch = max(1, batch_size // n)
    results: list[CaseResult] = []

    for start in range(0, len(cases), per_batch):
        chunk = cases[start : start + per_batch]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(case["record"], prompt_version),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            for case in chunk
            for _ in range(n)
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = policy.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0 if temperature > 0 else None,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        lengths = [int((row != tokenizer.pad_token_id).sum()) for row in generated]

        for i, case in enumerate(chunk):
            samples = [
                Sample(case["id"], texts[i * n + j], lengths[i * n + j]) for j in range(n)
            ]
            results.append(CaseResult(case, samples))

        done = min(start + per_batch, len(cases))
        print(f"  {done}/{len(cases)} cases", end="\r", file=sys.stderr)

    print(file=sys.stderr)
    if loaded is not None and was_training:
        policy.train()
    return results


# --- metrics -------------------------------------------------------------------


def _distinct_4(texts: list[str]) -> float:
    """Fraction of distinct 4-grams over whitespace tokens. Tokenizer-free."""
    grams: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - 3):
            grams.add(tuple(words[i : i + 4]))
            total += 1
    return len(grams) / total if total else 0.0


def summarise(results: list[CaseResult], weights) -> dict[str, Any]:
    rows = []
    for result in results:
        scored = [score(s.completion, result.case["answer"], weights) for s in result.samples]
        parsed = [parse_answer(s.completion).obj for s in result.samples]
        rows.append(
            {
                "case": result.case,
                "samples": result.samples,
                "scored": scored,
                "unique_answers": len(
                    {canonical(obj) for obj in parsed if isinstance(obj, dict)}
                ),
                "unique_completions": len({s.completion.strip() for s in result.samples}),
            }
        )

    n = len(rows)
    k = len(rows[0]["samples"]) if rows else 0

    def mean(values) -> float:
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else float("nan")

    first = [row["scored"][0] for row in rows]
    lengths = [s.completion_tokens for row in rows for s in row["samples"]]

    metrics = {
        "n_cases": n,
        "k": k,
        # First sample: what the model does when asked once.
        "reward": mean(r.total for r in first),
        "exact_match": mean(float(r.diagnostics["exact_match"]) for r in first),
        "validity_gate": mean(float(r.gate_passed) for r in first),
        "schema_ok": mean(float(r.diagnostics.get("schema_ok", False)) for r in first),
        "cause_acc": mean(float(r.diagnostics.get("root_cause_correct", False)) for r in first),
        "action_acc": mean(float(r.diagnostics.get("action_correct", False)) for r in first),
        "numeric_acc": mean((r.diagnostics.get("numeric_correct", 0) or 0) / 3 for r in first),
        "flags_acc": mean((r.diagnostics.get("flags_correct", 0) or 0) / 3 for r in first),
        "cause_given_flags": mean(
            r.diagnostics.get("cause_given_flags") for r in first
            if r.diagnostics.get("cause_given_flags") is not None
        ),
        "components": {
            name: mean(r.components.get(name, 0.0) for r in first)
            for name in ("format", "numeric", "flags", "stage", "root_cause", "action")
        },
        # Across all k samples.
        "reward_mean_all": mean(r.total for row in rows for r in row["scored"]),
        "completion_tokens_mean": mean(lengths) if any(lengths) else None,
        "completion_chars_mean": mean(
            len(s.completion) for row in rows for s in row["samples"]
        ),
    }

    if k > 1:
        metrics["pass_at_k"] = mean(
            float(any(r.diagnostics["exact_match"] for r in row["scored"])) for row in rows
        )
        metrics["unique_answers_mean"] = mean(row["unique_answers"] for row in rows)
        metrics["unique_completions_mean"] = mean(row["unique_completions"] for row in rows)
        metrics["distinct_4"] = _distinct_4(
            [s.completion for row in rows for s in row["samples"]]
        )
        # A group where every completion earns the same reward gives GRPO no
        # gradient at all. Pre-registered as `adv_zero_frac`.
        metrics["adv_zero_frac"] = mean(
            float(len({round(r.total, 9) for r in row["scored"]}) == 1) for row in rows
        )

    # Failure taxonomy, so "validity is low" comes with a reason attached.
    metrics["parse_errors"] = dict(
        Counter(r.parse_error for r in first if r.parse_error is not None)
    )
    metrics["schema_errors"] = dict(
        Counter(e.split(":")[0] for r in first for e in r.schema_errors).most_common(8)
    )
    metrics["predicted_cause_hist"] = dict(
        Counter(r.diagnostics.get("predicted_cause") for r in first)
    )
    return metrics


def breakdown(results: list[CaseResult], weights, key: str) -> dict[str, dict]:
    groups: dict[str, list[CaseResult]] = {}
    for result in results:
        groups.setdefault(result.case[key], []).append(result)
    return {name: summarise(rows, weights) for name, rows in sorted(groups.items())}


# --- reporting ------------------------------------------------------------------


def print_report(label: str, metrics: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"  cases              {metrics['n_cases']}   k={metrics['k']}")
    print(f"  reward             {metrics['reward']:.3f}")
    print(f"  exact match        {metrics['exact_match']:.3f}")
    if "pass_at_k" in metrics:
        print(f"  pass@{metrics['k']}             {metrics['pass_at_k']:.3f}")
        print(f"  unique answers     {metrics['unique_answers_mean']:.2f} / {metrics['k']}")
        print(f"  distinct-4         {metrics['distinct_4']:.3f}")
        print(f"  zero-variance grp  {metrics['adv_zero_frac']:.3f}")
    print(f"  validity (gate)    {metrics['validity_gate']:.3f}")
    print(f"  schema ok          {metrics['schema_ok']:.3f}")
    print(f"  cause accuracy     {metrics['cause_acc']:.3f}")
    print(f"  cause | flags ok   {metrics['cause_given_flags']:.3f}")
    print(f"  numeric (of 3)     {metrics['numeric_acc']:.3f}")
    print(f"  flags   (of 3)     {metrics['flags_acc']:.3f}")
    tokens = metrics.get("completion_tokens_mean")
    if tokens:
        print(f"  length             {tokens:.0f} tok")
    else:
        print(f"  length             {metrics['completion_chars_mean']:.0f} chars")
    if metrics["parse_errors"]:
        print(f"  parse errors       {metrics['parse_errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["openai", "hf"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("-k", type=int, default=8, help="completions per prompt in sample mode")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--weights", default="MAIN", choices=["MAIN", "PROBE"])
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, choices=list(VERSIONS))
    parser.add_argument("--run-name", default=None)
    # openai backend
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--thinking", action="store_true", help="leave the reasoning mode on")
    # hf backend
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory")
    args = parser.parse_args()

    path = DATA / f"{args.split}.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    if args.limit:
        cases = cases[: args.limit]
    split_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    temperature = 0.0 if args.mode == "greedy" else 1.0
    n = 1 if args.mode == "greedy" else args.k
    weights = getattr(reward_module, args.weights)

    print(
        f"{args.backend} | {args.model} | {args.split} ({len(cases)} cases) | "
        f"{args.mode} n={n} T={temperature} | prompt {args.prompt_version}"
    )

    started = time.perf_counter()
    if args.backend == "openai":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"set {args.api_key_env} (the key is never written to disk here)")
        results = generate_openai(
            cases,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            n=n,
            temperature=temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            concurrency=args.concurrency,
            timeout=args.timeout,
            thinking=args.thinking,
            prompt_version=args.prompt_version,
        )
    else:
        import torch

        device = args.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        results = generate_hf(
            cases,
            model=args.model,
            device=device,
            dtype=args.dtype,
            n=n,
            temperature=temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            batch_size=args.batch_size,
            adapter=args.adapter,
            prompt_version=args.prompt_version,
            thinking=args.thinking,
        )
    elapsed = time.perf_counter() - started

    overall = summarise(results, weights)
    by_tier = breakdown(results, weights, "tier")
    by_slice = breakdown(results, weights, "slice")

    print_report(f"{args.split} overall", overall)
    for tier, metrics in by_tier.items():
        print_report(f"{args.split} / tier={tier}", metrics)
    if len(by_slice) > 1:
        for name, metrics in by_slice.items():
            print_report(f"{args.split} / slice={name}", metrics)

    run_name = args.run_name or f"{Path(args.model).name}-{args.split}-{args.mode}"
    out = ROOT / "runs" / run_name / f"eval_{args.split}_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "backend": args.backend,
                "split": args.split,
                "split_sha256": split_sha,
                "prompt_version": args.prompt_version,
                "thinking_requested": args.thinking,
                "weights": args.weights,
                "mode": args.mode,
                "k": n,
                "temperature": temperature,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "adapter": args.adapter,
                "elapsed_seconds": round(elapsed, 1),
                "overall": overall,
                "by_tier": by_tier,
                "by_slice": by_slice,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"\nwrote {out.relative_to(ROOT)}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
