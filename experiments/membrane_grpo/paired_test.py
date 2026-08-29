"""McNemar's exact test on two evaluations of the same frozen split.

Written because a claim in this project was resting on the wrong comparison.
After 200 GRPO steps on Qwen3-1.7B, held-out `root_cause` accuracy moved
0.235 -> 0.290. Against the independent-sample standard error for a proportion
near 0.26 on 200 cases -- 0.031 -- that is 1.8 SE, which is not much.

But the two evaluations are not independent samples. They are the *same 200
cases*, in the same order, decoded greedily. Almost every case is answered the
same way by both policies, and those cases carry no information about the
difference. Only the disagreements do. McNemar's test uses exactly them, and it
is the test the design called for all along.

    python3 paired_test.py before.json after.json --metric cause

Requires `per_case` in both files, which `eval.py` writes. Earlier runs predate
that and cannot be tested this way -- which is why the field exists now.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def exact_binomial_two_sided(b: int, c: int) -> float:
    """P(|disagreements at least this lopsided|) under H0: b and c equiprobable.

    The exact test rather than the chi-square approximation, because the
    discordant counts here are small enough that the approximation is not
    trustworthy.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text())
    per_case = payload["overall"].get("per_case")
    if not per_case:
        raise SystemExit(
            f"{path} has no per_case block — it predates eval.py writing one, "
            "so it cannot support a paired test"
        )
    return payload, {row["id"]: row for row in per_case}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument(
        "--metric", default="cause", choices=["cause", "exact", "schema"]
    )
    args = parser.parse_args()

    before_payload, before = load(args.before)
    after_payload, after = load(args.after)

    if before_payload.get("split_sha256") != after_payload.get("split_sha256"):
        raise SystemExit("the two evaluations used different data; not comparable")

    shared = sorted(set(before) & set(after))
    if len(shared) != len(before) or len(shared) != len(after):
        print(f"  warning: comparing {len(shared)} shared cases only")

    both = only_before = only_after = neither = 0
    for case_id in shared:
        x = bool(before[case_id][args.metric])
        y = bool(after[case_id][args.metric])
        if x and y:
            both += 1
        elif x and not y:
            only_before += 1
        elif y and not x:
            only_after += 1
        else:
            neither += 1

    n = len(shared)
    p = exact_binomial_two_sided(only_before, only_after)
    rate_before = (both + only_before) / n
    rate_after = (both + only_after) / n

    print(f"metric: {args.metric}   n = {n}   split {before_payload['split']}")
    print(f"  {args.before.parent.name:>28}: {rate_before:.3f}")
    print(f"  {args.after.parent.name:>28}: {rate_after:.3f}")
    print(f"  difference                  : {rate_after - rate_before:+.3f}")
    print()
    print("  agreement table")
    print(f"    both correct              : {both}")
    print(f"    only before               : {only_before}")
    print(f"    only after                : {only_after}")
    print(f"    neither                   : {neither}")
    print()
    print(f"  discordant pairs            : {only_before + only_after} of {n}")
    print(f"  McNemar exact two-sided p   : {p:.4f}")
    verdict = "significant at 0.05" if p < 0.05 else "not significant at 0.05"
    print(f"  verdict                     : {verdict}")

    # The contrast with the unpaired reading, stated so it cannot be missed --
    # though note it does not always favour pairing. Pairing helps when the two
    # policies agree on most cases; where the discordant pairs are themselves
    # balanced, the paired test is the *more* conservative of the two, and that
    # is the honest answer rather than a worse one.
    independent_se = (rate_after * (1 - rate_after) / n) ** 0.5
    if independent_se == 0:
        print("\n  both rates are 0; there is nothing to compare")
        return
    print(
        f"\n  for contrast, the independent-sample SE is {independent_se:.3f}, "
        f"so the same difference reads as {abs(rate_after - rate_before) / independent_se:.1f} SE"
        "\n  the paired test uses only the "
        f"{only_before + only_after} discordant pairs; the {both + neither} cases "
        "both policies answered alike carry no information"
    )


if __name__ == "__main__":
    main()
