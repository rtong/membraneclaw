"""The answer schema, and a parser tolerant enough to be informative.

Two jobs, deliberately kept apart:

`parse_answer()` tries to get *a JSON object* out of whatever the model emitted.
It is lenient -- it will dig the object out of a code fence or out of a sentence
of preamble -- because a model that produced the right diagnosis and wrapped it
in "Here is my analysis:" has not made the same mistake as one that produced
prose. The reward gate sits here, and it is the loosest check in the pipeline.

`validate()` then asks whether that object is actually the requested schema:
exactly these keys, these types, these vocabularies. This is strict, and it is
*not* the gate. A response that parses but has a stray extra key still gets
graded on the fields it did supply.

Splitting them this way is a deliberate choice about gradient shape. Gating the
entire reward on strict validity would hand a 0.5B model a reward of zero on
essentially every rollout early in training, every group would have zero
variance, and GRPO's advantages would be identically zero -- no gradient, no
run. Partial credit on a nearly-right object is what keeps the signal alive.
The cost is that "reward went up" and "the answer got better" come apart very
easily, which is exactly the phenomenon this experiment is built to look at.
"""
from __future__ import annotations

import json
from typing import Any, NamedTuple

from .decision_table import ACTION_SET, CAUSES, STAGES

NUMERIC_KEYS = (
    "normalized_flow_change_pct",
    "salt_passage_change_pct",
    "dp_change_pct",
)

FLAG_KEYS = ("flow", "salt_passage", "dp")

ANSWER_KEYS = (*NUMERIC_KEYS, "flags", "stage", "root_cause", "action")

FLAG_VOCAB = {
    "flow": ("down", "flat", "up"),
    "salt_passage": ("down", "flat", "up", "sharp_up"),
    "dp": ("down", "flat", "up"),
}


class ParseResult(NamedTuple):
    obj: dict[str, Any] | None
    error: str | None  # None on success; a short slug otherwise


class Validation(NamedTuple):
    ok: bool
    errors: tuple[str, ...]


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} span, ignoring braces inside strings."""
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


def parse_answer(completion: str) -> ParseResult:
    """Extract a JSON object from a raw completion.

    Tries the whole string first, then the first balanced brace span, which
    covers both ```json fences and a bare object preceded by chatter.
    """
    text = completion.strip()
    if not text:
        return ParseResult(None, "empty")

    for candidate in (text, _first_json_object(text)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return ParseResult(obj, None)
        return ParseResult(None, "not_an_object")

    return ParseResult(None, "no_json")


def _is_number(value: Any) -> bool:
    # bool is an int subclass in Python, and `true` is not a percentage.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate(obj: dict[str, Any]) -> Validation:
    """Strict schema check. Returns every problem found, not just the first."""
    errors: list[str] = []

    missing = [key for key in ANSWER_KEYS if key not in obj]
    extra = [key for key in obj if key not in ANSWER_KEYS]
    errors += [f"missing:{key}" for key in missing]
    errors += [f"extra:{key}" for key in extra]

    for key in NUMERIC_KEYS:
        if key in obj and not _is_number(obj[key]):
            errors.append(f"not_a_number:{key}")

    flags = obj.get("flags")
    if "flags" in obj:
        if not isinstance(flags, dict):
            errors.append("flags:not_an_object")
        else:
            errors += [f"flags.missing:{key}" for key in FLAG_KEYS if key not in flags]
            errors += [f"flags.extra:{key}" for key in flags if key not in FLAG_KEYS]
            for key in FLAG_KEYS:
                if key in flags and flags[key] not in FLAG_VOCAB[key]:
                    errors.append(f"flags.bad_value:{key}")

    for key, vocab in (("stage", STAGES), ("root_cause", CAUSES), ("action", ACTION_SET)):
        if key in obj and obj[key] not in vocab:
            errors.append(f"bad_value:{key}")

    return Validation(ok=not errors, errors=tuple(errors))


def canonical(answer: dict[str, Any]) -> str:
    """Stable serialisation, used for exact-match and diversity counting."""
    return json.dumps(answer, sort_keys=True, separators=(",", ":"))
