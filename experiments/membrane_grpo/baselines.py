"""Degenerate answering strategies, and what the reward pays them.

Two uses. As test fixtures they pin down how much a given cheat is worth, so a
change to the weights cannot quietly make one of them profitable. As reference
lines on the training curves they answer the question every reward plot should
have to answer: *is the number going up better than a strategy that is not
solving the problem at all?*

The interesting one is `skip_correction`. It is not a cheat -- it reads the
right values, does the right divisions, and follows the table correctly. Its
only flaw is ignoring the temperature correction. On `easy` cases, where the two
temperatures are equal and the factor cancels, it is exactly right and scores a
perfect 1.0. On `hard` cases it is wrong from the first field onward. It is
therefore the sharpest available prediction about what a model that learns the
easy tier and nothing else will look like on the reward curve.

Every strategy takes a case's `record` and its `answer` and returns a string, so
they plug into `reward.score` exactly where a model completion would.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from task.decision_table import (
    ACTIONS,
    CAUSES,
    SEVERE_ACTION,
    SEVERE_FLOW_LOSS_PCT,
    UncoveredSymptoms,
    classify,
)
from task.prompt import SCHEMA_EXAMPLE

Strategy = Callable[[dict[str, Any], dict[str, Any]], str]

#: Used when a strategy's own (wrong) numbers land on a symptom combination the
#: table does not cover. A real model would guess something; this guesses the
#: same thing every time so the score stays reproducible.
FALLBACK_CAUSE = "compaction"


def empty(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """The model emitted nothing. Fails the gate."""
    return ""


def prose(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """Fluent, on-topic, no JSON anywhere. Fails the gate."""
    return (
        "Looking at the two readings, the permeate flow has dropped noticeably "
        "while the differential pressure has risen. This pattern is consistent "
        "with fouling in the affected stage, and I would recommend a clean."
    )


def schema_template(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """The requested schema echoed back verbatim, placeholders and all.

    A real and common failure for small instruct models. It parses -- the
    placeholders are unquoted, so this is not even valid JSON... which is the
    point of including it.
    """
    return SCHEMA_EXAMPLE


def constant(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """Schema-perfect, content-free: valid JSON, same guess every time.

    The floor a reward has to beat. Labels are balanced across seven causes, so
    this lands the cause about 1/7 of the time by luck.
    """
    return json.dumps(
        {
            "normalized_flow_change_pct": 0.0,
            "salt_passage_change_pct": 0.0,
            "dp_change_pct": 0.0,
            "flags": {"flow": "down", "salt_passage": "up", "dp": "up"},
            "stage": "tail",
            "root_cause": CAUSES[0],
            "action": ACTIONS[CAUSES[0]],
        }
    )


def copy_stage_only(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """`constant`, but copying the one field the prompt states outright.

    Isolates what the trivially copyable field is worth on its own.
    """
    payload = json.loads(constant(record, answer))
    payload["stage"] = record["anomaly_stage"]
    return json.dumps(payload)


def skip_correction(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """Everything right except the temperature correction.

    Perfect on `easy` cases, where TCF cancels. Wrong on `hard` ones.
    """
    t0, t1 = record["t0"], record["t1"]

    flow_pct = (
        (t1["permeate_flow_m3_h"] - t0["permeate_flow_m3_h"]) / t0["permeate_flow_m3_h"] * 100.0
    )
    sp0 = t0["permeate_conductivity_uS_cm"] / t0["feed_conductivity_uS_cm"] * 100.0
    sp1 = t1["permeate_conductivity_uS_cm"] / t1["feed_conductivity_uS_cm"] * 100.0
    sp_pct = (sp1 - sp0) / sp0 * 100.0
    dp0 = t0["dp_lead_bar"] + t0["dp_tail_bar"]
    dp1 = t1["dp_lead_bar"] + t1["dp_tail_bar"]
    dp_pct = (dp1 - dp0) / dp0 * 100.0

    stage = record["anomaly_stage"]
    try:
        diagnosis = classify(flow_pct, sp_pct, dp_pct, stage)
        flags, cause, action = diagnosis.flags, diagnosis.root_cause, diagnosis.action
    except UncoveredSymptoms:
        from task.decision_table import flags_for

        flags = flags_for(flow_pct, sp_pct, dp_pct)
        cause = FALLBACK_CAUSE
        action = SEVERE_ACTION if flow_pct <= SEVERE_FLOW_LOSS_PCT else ACTIONS[cause]

    return json.dumps(
        {
            "normalized_flow_change_pct": round(flow_pct, 1),
            "salt_passage_change_pct": round(sp_pct, 1),
            "dp_change_pct": round(dp_pct, 1),
            "flags": flags,
            "stage": stage,
            "root_cause": cause,
            "action": action,
        }
    )


def oracle(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """The answer key. Must score exactly 1.0, or the reward is miscalibrated."""
    return json.dumps(answer)


def oracle_verbose(record: dict[str, Any], answer: dict[str, Any]) -> str:
    """Right content, wrapped in chatter and carrying an extra key.

    Should lose the format component and nothing else -- the check that the
    lenient gate and the strict validator are wired up the way they are meant
    to be.
    """
    payload = dict(answer, confidence=0.87)
    return (
        "Here is my analysis of the two readings.\n\n```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n\nLet me know if you need the cleaning procedure."
    )


STRATEGIES: dict[str, Strategy] = {
    "empty": empty,
    "prose": prose,
    "schema_template": schema_template,
    "constant": constant,
    "copy_stage_only": copy_stage_only,
    "skip_correction": skip_correction,
    "oracle_verbose": oracle_verbose,
    "oracle": oracle,
}


def evaluate(cases: list[dict[str, Any]], weights) -> dict[str, dict[str, float]]:
    """Score every strategy over every case. Returns {strategy: metrics}."""
    from reward import score

    report: dict[str, dict[str, float]] = {}
    for name, strategy in STRATEGIES.items():
        results = [
            (case, score(strategy(case["record"], case["answer"]), case["answer"], weights))
            for case in cases
        ]
        n = len(results)
        by_tier = {
            tier: [r for c, r in results if c["tier"] == tier] for tier in ("easy", "hard")
        }
        report[name] = {
            "reward": sum(r.total for _, r in results) / n,
            "reward_easy": (
                sum(r.total for r in by_tier["easy"]) / len(by_tier["easy"])
                if by_tier["easy"]
                else float("nan")
            ),
            "reward_hard": (
                sum(r.total for r in by_tier["hard"]) / len(by_tier["hard"])
                if by_tier["hard"]
                else float("nan")
            ),
            "exact_match": sum(r.diagnostics["exact_match"] for _, r in results) / n,
            "validity": sum(r.gate_passed for _, r in results) / n,
            "cause_acc": sum(
                bool(r.diagnostics.get("root_cause_correct")) for _, r in results
            )
            / n,
        }
    return report


def main() -> None:
    import argparse
    import json as _json
    from pathlib import Path

    import reward as reward_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--weights", default="MAIN", choices=["MAIN", "PROBE"])
    args = parser.parse_args()

    path = Path(__file__).resolve().parent / "data" / f"{args.split}.jsonl"
    cases = [_json.loads(line) for line in path.read_text().splitlines()]
    weights = getattr(reward_module, args.weights)

    print(f"{args.split}: {len(cases)} cases, {args.weights} weights\n")
    header = f"{'strategy':<18}{'reward':>8}{'easy':>8}{'hard':>8}{'EM':>8}{'valid':>8}{'cause':>8}"
    print(header)
    print("-" * len(header))
    for name, m in evaluate(cases, weights).items():
        print(
            f"{name:<18}{m['reward']:>8.3f}{m['reward_easy']:>8.3f}{m['reward_hard']:>8.3f}"
            f"{m['exact_match']:>8.2f}{m['validity']:>8.2f}{m['cause_acc']:>8.2f}"
        )


if __name__ == "__main__":
    main()
