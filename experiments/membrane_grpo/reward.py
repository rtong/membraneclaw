"""The deterministic reward. Every training signal in this experiment comes from here.

There is no LLM judge and no fuzzy matching: a completion is parsed, compared
field by field against `task.decision_table`, and turned into a scalar. Run it
twice on the same string and it returns the same number.

## Shape of the reward

A loose gate and a strict scorer, for reasons argued in `task/schema.py`: the
gate only asks whether a JSON object came out at all, so a nearly-right answer
keeps earning partial credit instead of collapsing to zero. Gating on strict
schema validity would zero almost every rollout from a cold 0.5B model, every
group would have zero reward variance, and GRPO's advantages would be
identically zero -- a run that cannot start.

Six components, summing to 1.0 when everything is right:

    format      0.10   exactly the requested keys, types and vocabularies
    numeric     0.15   three percent changes, 0.05 each, +/- 0.5 pp
    flags       0.12   three trend flags, 0.04 each
    stage       0.03   a copy of a value stated in the prompt
    root_cause  0.45   the actual diagnosis
    action      0.15   the lookup, plus the severity override

`stage` gets its own tiny weight rather than being folded in anywhere, because
watching the most trivially copyable field saturate first is part of what this
experiment is for.

## The components are not independent

Worth stating plainly, because the curves will invite the opposite reading. One
arithmetic slip on `dp_change_pct` costs its 0.05, then flips the `dp` flag
(-0.04), which changes the table lookup (-0.45), which changes the action
(-0.15). A single wrong number costs 0.69 of a possible 1.0. The numeric
component's real influence is far larger than its 0.15 weight suggests, and
`root_cause` accuracy is not a clean measure of whether the model can read the
table.

`diagnostics["cause_given_flags"]` exists to separate those two. Conditioned on
the flags being right, did the model pick the right row? That is the lookup
ability with the arithmetic cascade divided out.

## No length penalty

Deliberately. Completion length is a *measurement* in this experiment, not
something to control. A length term in the reward would suppress exactly the
drift worth observing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from task.decision_table import CAUSES
from task.schema import FLAG_KEYS, NUMERIC_KEYS, parse_answer, validate

# How close a percent change has to be. The answer key is rounded to one
# decimal, and generated cases sit at least 2 pp from any flag threshold, so
# 0.5 pp is loose enough to forgive a last-digit disagreement and far too tight
# to forgive a missing temperature correction.
NUMERIC_TOLERANCE_PP = 0.5


@dataclass(frozen=True)
class Weights:
    format: float = 0.10
    numeric: float = 0.15
    flags: float = 0.12
    stage: float = 0.03
    root_cause: float = 0.45
    action: float = 0.15

    def total(self) -> float:
        return self.format + self.numeric + self.flags + self.stage + self.root_cause + self.action


#: The reward the main run is trained against.
MAIN = Weights()

#: The control. Same task, same model, same code -- only the weights move, and
#: they move away from the field that requires actually solving the problem. If
#: reward climbs faster here while held-out exact match climbs slower, "reward
#: went up" has been shown to be separable from "the model got better" within a
#: single experiment rather than across two.
PROBE = Weights(
    format=0.35,
    numeric=0.35,
    flags=0.12,
    stage=0.03,
    root_cause=0.10,
    action=0.05,
)


@dataclass
class Reward:
    total: float
    components: dict[str, float]
    gate_passed: bool
    parse_error: str | None = None
    schema_errors: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_hits(obj: dict[str, Any], answer: dict[str, Any]) -> list[bool]:
    hits = []
    for key in NUMERIC_KEYS:
        got = obj.get(key)
        hits.append(_is_number(got) and abs(float(got) - answer[key]) <= NUMERIC_TOLERANCE_PP)
    return hits


def _flag_hits(obj: dict[str, Any], answer: dict[str, Any]) -> list[bool]:
    got = obj.get("flags")
    if not isinstance(got, dict):
        return [False] * len(FLAG_KEYS)
    return [got.get(key) == answer["flags"][key] for key in FLAG_KEYS]


def _zero(weights: Weights) -> dict[str, float]:
    return {name: 0.0 for name in asdict(weights)}


def score(
    completion: str, answer: dict[str, Any], weights: Weights = MAIN
) -> Reward:
    """Grade one completion against one answer key.

    `answer` is the `answer` field of a case from `data/*.jsonl`, which
    `task.generate.truth_from_record` guarantees is derivable from the record
    the model was shown.
    """
    parsed = parse_answer(completion)
    if parsed.obj is None:
        return Reward(
            total=0.0,
            components=_zero(weights),
            gate_passed=False,
            parse_error=parsed.error,
            diagnostics={"exact_match": False, "cause_given_flags": None},
        )

    obj = parsed.obj
    schema = validate(obj)

    numeric = _numeric_hits(obj, answer)
    flags = _flag_hits(obj, answer)
    stage_ok = obj.get("stage") == answer["stage"]
    cause_ok = obj.get("root_cause") == answer["root_cause"]
    action_ok = obj.get("action") == answer["action"]

    components = {
        "format": weights.format * float(schema.ok),
        "numeric": weights.numeric * (sum(numeric) / len(numeric)),
        "flags": weights.flags * (sum(flags) / len(flags)),
        "stage": weights.stage * float(stage_ok),
        "root_cause": weights.root_cause * float(cause_ok),
        "action": weights.action * float(action_ok),
    }

    flags_ok = all(flags)
    exact = all(numeric) and flags_ok and stage_ok and cause_ok and action_ok

    return Reward(
        total=sum(components.values()),
        components=components,
        gate_passed=True,
        schema_errors=schema.errors,
        diagnostics={
            "exact_match": exact,
            "schema_ok": schema.ok,
            "numeric_correct": sum(numeric),
            "flags_correct": sum(flags),
            "stage_correct": stage_ok,
            "root_cause_correct": cause_ok,
            "action_correct": action_ok,
            # Table-reading ability with the arithmetic cascade divided out.
            # None when the flags are wrong, so averaging skips those cases
            # rather than scoring them as failures of a lookup never reached.
            "cause_given_flags": cause_ok if flags_ok else None,
            "predicted_cause": obj.get("root_cause") if obj.get("root_cause") in CAUSES else None,
            "completion_chars": len(completion),
        },
    )


def score_batch(
    completions: list[str], answer: dict[str, Any], weights: Weights = MAIN
) -> list[Reward]:
    """Grade one GRPO group: many completions, one prompt, one answer key."""
    return [score(completion, answer, weights) for completion in completions]


def group_advantages(rewards: list[float]) -> tuple[list[float], bool]:
    """GRPO's group-normalised advantages, and whether the group is degenerate.

    When every completion in a group earns the same reward the standard
    deviation is zero, every advantage is zero, and the group contributes no
    gradient at all. That is not an edge case to be smoothed over with an
    epsilon -- it is a real and frequent event on an easy prompt with a small
    model, and `adv_zero_frac` is one of the pre-registered things to watch. The
    flag is returned so the caller can count it instead of silently dividing by
    a fudge factor.
    """
    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    if variance <= 0.0:
        return [0.0] * n, True
    std = variance**0.5
    return [(r - mean) / std for r in rewards], False
