"""Compare prompt versions on the same cases, and write the numbers to a file.

This exists because of a process failure worth naming. The most important
findings of the 9B reference run -- that the model estimates rather than
computes under v1, and that the `dp` field is ambiguous -- were produced by
ad-hoc inline scripts during a fast exploration. They left no artifact. The
README quoted them and nothing in the repository could reproduce or check them,
which is exactly the standard `probe_throughput.py` sets for itself two
directories away: "so the README's numbers have a file behind them".

So: same cases, same model, same decoding, one variable. Every number the README
cites about prompt versions should come from here.

    .venv/bin/python prompt_ab.py --model qwen3.5-9b --base-url http://HOST:8000/v1 -n 40

What it reports beyond `eval.py`'s summary, because the v1 diagnosis turned on
these specifically:

* **per-field numeric accuracy** -- the three computations fail at different
  rates and for different reasons, and averaging them hid that;
* **error quantiles** -- v1's errors were bimodal, near-exact or wildly wrong,
  which is what proved they were mistakes rather than noise;
* **a tolerance sweep** -- widening the band from 0.5 pp to 5.0 pp barely moved
  the score, which is how we know loosening the reward would not have helped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from eval import generate_hf, generate_openai
from reward import MAIN, NUMERIC_TOLERANCE_PP, score
from task.prompt import VERSIONS
from task.schema import NUMERIC_KEYS, parse_answer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

TOLERANCE_SWEEP = (0.5, 1.0, 2.0, 5.0)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def analyse(results, tier: str | None = None) -> dict[str, Any]:
    """Field-level accuracy and error shape for one prompt version."""
    rows = [r for r in results if tier is None or r.case["tier"] == tier]
    if not rows:
        return {}

    errors: dict[str, list[float]] = {key: [] for key in NUMERIC_KEYS}
    hits: dict[str, int] = {key: 0 for key in NUMERIC_KEYS}
    per_case_errors: list[list[float]] = []
    scored = []
    lengths = []
    gated = 0

    for result in rows:
        completion = result.samples[0].completion
        lengths.append(result.samples[0].completion_tokens or 0)
        parsed = parse_answer(completion)
        outcome = score(completion, result.case["answer"], MAIN)
        scored.append(outcome)
        if parsed.obj is None:
            continue
        gated += 1

        case_errors = []
        for key in NUMERIC_KEYS:
            value = parsed.obj.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                err = abs(float(value) - result.case["answer"][key])
                errors[key].append(err)
                hits[key] += err <= NUMERIC_TOLERANCE_PP
                case_errors.append(err)
        if len(case_errors) == len(NUMERIC_KEYS):
            per_case_errors.append(case_errors)

    n = len(rows)

    def frac(values) -> float:
        return sum(values) / n if n else float("nan")

    return {
        "n": n,
        "validity": gated / n,
        "reward": frac(o.total for o in scored),
        "exact_match": frac(float(o.diagnostics["exact_match"]) for o in scored),
        "cause_acc": frac(float(o.diagnostics.get("root_cause_correct", False)) for o in scored),
        "action_acc": frac(float(o.diagnostics.get("action_correct", False)) for o in scored),
        "all_flags": frac(float(o.diagnostics.get("flags_correct", 0) == 3) for o in scored),
        "tokens_mean": statistics.mean(lengths) if any(lengths) else None,
        "per_field": {
            key: {
                "within_tolerance": hits[key] / n,
                "median_error": statistics.median(errors[key]) if errors[key] else float("nan"),
                "p90_error": _quantile(errors[key], 0.9),
                "answered": len(errors[key]),
            }
            for key in NUMERIC_KEYS
        },
        # If widening this barely moves the number, the errors are mistakes and
        # not noise -- and loosening the reward's tolerance would not help.
        "tolerance_sweep": {
            str(tol): sum(max(e) <= tol for e in per_case_errors) / n for tol in TOLERANCE_SWEEP
        },
    }


def print_table(by_version: dict[str, dict[str, Any]]) -> None:
    versions = list(by_version)
    width = 14

    def row(label: str, fmt, pick) -> str:
        cells = "".join(f"{fmt(pick(by_version[v])):>{width}}" for v in versions)
        return f"  {label:<30}{cells}"

    header = "".join(f"{v:>{width}}" for v in versions)
    print(f"\n  {'':<30}{header}")
    print("  " + "-" * (30 + width * len(versions)))
    for label, key in (
        ("validity (gate)", "validity"),
        ("reward", "reward"),
        ("exact match", "exact_match"),
        ("all three flags", "all_flags"),
        ("cause accuracy", "cause_acc"),
        ("action accuracy", "action_acc"),
    ):
        print(row(label, lambda x: f"{x:.3f}", lambda m, k=key: m[k]))
    print(row("completion tokens", lambda x: f"{x:.0f}" if x else "-", lambda m: m["tokens_mean"]))

    print(f"\n  numeric, within {NUMERIC_TOLERANCE_PP} pp")
    for key in NUMERIC_KEYS:
        print(row(f"  {key}", lambda x: f"{x:.3f}", lambda m, k=key: m["per_field"][k]["within_tolerance"]))
    print("\n  median absolute error")
    for key in NUMERIC_KEYS:
        print(row(f"  {key}", lambda x: f"{x:.2f}", lambda m, k=key: m["per_field"][k]["median_error"]))
    print("\n  p90 absolute error")
    for key in NUMERIC_KEYS:
        print(row(f"  {key}", lambda x: f"{x:.2f}", lambda m, k=key: m["per_field"][k]["p90_error"]))

    print("\n  all three numbers within")
    for tol in TOLERANCE_SWEEP:
        print(row(f"  {tol} pp", lambda x: f"{x:.3f}", lambda m, t=tol: m["tolerance_sweep"][str(t)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["openai", "hf"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("-n", "--limit", type=int, default=40)
    parser.add_argument("--versions", default=",".join(VERSIONS))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "prompt-ab")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",")]
    for version in versions:
        if version not in VERSIONS:
            raise SystemExit(f"unknown prompt version {version!r}")

    path = DATA / f"{args.split}.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines()][: args.limit]
    split_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    print(f"{args.model} | {args.split} | {len(cases)} cases | versions {versions}")

    by_version: dict[str, dict[str, Any]] = {}
    by_tier: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for version in versions:
        print(f"\n  running {version} ...")
        if args.backend == "openai":
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise SystemExit(f"set {args.api_key_env}")
            results = generate_openai(
                cases,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                n=1,
                temperature=0.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
                concurrency=args.concurrency,
                timeout=args.timeout,
                thinking=False,
                prompt_version=version,
            )
        else:
            results = generate_hf(
                cases,
                model=args.model,
                device=args.device,
                dtype=args.dtype,
                n=1,
                temperature=0.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
                batch_size=args.batch_size,
                adapter=None,
                prompt_version=version,
            )
        by_version[version] = analyse(results)
        by_tier[version] = {
            tier: analyse(results, tier) for tier in ("easy", "hard")
        }

    print_table(by_version)
    print("\n  by tier, exact match")
    for version in versions:
        parts = " ".join(
            f"{tier}={by_tier[version][tier].get('exact_match', float('nan')):.3f}"
            for tier in ("easy", "hard")
        )
        print(f"    {version}: {parts}")

    elapsed = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / f"{Path(args.model).name}_{args.split}_n{len(cases)}.json"
    outfile.write_text(
        json.dumps(
            {
                "model": args.model,
                "backend": args.backend,
                "split": args.split,
                "split_sha256": split_sha,
                "n_cases": len(cases),
                "versions": versions,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "temperature": 0.0,
                "tolerance_pp": NUMERIC_TOLERANCE_PP,
                "elapsed_seconds": round(elapsed, 1),
                "overall": by_version,
                "by_tier": by_tier,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {outfile.relative_to(ROOT)}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
