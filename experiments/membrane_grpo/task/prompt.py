"""The frozen prompt template.

Every fact needed to answer is in the prompt: the correction formula, the flag
thresholds, the diagnosis rules, the action lookup, and the output schema. The
task is deliberately **closed-book** -- a 0.5B model has no reliable RO domain
knowledge, and an experiment about reinforcement learning should not be
bottlenecked on knowledge the base model was never going to have. What is being
measured is whether RL improves the model's ability to *execute a stated
procedure*: read the right numbers, compute, threshold, look up, and emit valid
structure.

The same template serves both difficulty tiers. `easy` and `hard` cases differ
only in whether the two feed temperatures happen to be equal -- when they are,
the correction factor cancels and the first computation collapses to a plain
percent change. Nothing in the wording changes. That keeps tier difficulty a
purely arithmetic variable rather than a confound with prompt format.

`PROMPT_VERSION` is part of every result record. Changing the template without
bumping it would silently invalidate comparisons against an earlier baseline,
which is the single easiest way to ruin a frozen evaluation.
"""
from __future__ import annotations

from typing import Any

from .decision_table import (
    ACTIONS,
    DP_DOWN_PCT,
    DP_UP_PCT,
    FLOW_DOWN_PCT,
    FLOW_UP_PCT,
    SEVERE_ACTION,
    SEVERE_FLOW_LOSS_PCT,
    SP_DOWN_PCT,
    SP_SHARP_UP_PCT,
    SP_UP_PCT,
)

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a reverse-osmosis membrane troubleshooting assistant. "
    "You reply with exactly one JSON object and no other text."
)

# The 17 rows of `decision_table.COVERED`, compressed to one line per cause.
# `test_prompt.py` checks that this compression is lossless: every covered key
# must match exactly one rule and get back the same cause. Without that test the
# prompt could drift from the answer key and every reward in the run would be
# graded against a procedure the model was never shown.
RULES: tuple[tuple[str, dict[str, tuple[str, ...]]], ...] = (
    ("scaling", {"flow": ("down",), "dp": ("up",), "stage": ("tail",)}),
    (
        "biofouling",
        {
            "flow": ("down",),
            "dp": ("up",),
            "stage": ("lead",),
            "salt_passage": ("up", "sharp_up"),
        },
    ),
    (
        "colloidal_fouling",
        {"flow": ("down",), "dp": ("up",), "stage": ("lead",), "salt_passage": ("flat",)},
    ),
    ("organic_fouling", {"flow": ("down",), "dp": ("flat",), "salt_passage": ("down",)}),
    (
        "compaction",
        {"flow": ("down",), "dp": ("flat",), "salt_passage": ("flat", "up")},
    ),
    ("oxidation_damage", {"flow": ("up",), "salt_passage": ("sharp_up",)}),
    (
        "mechanical_leak",
        {"flow": ("flat",), "salt_passage": ("sharp_up",), "dp": ("flat",)},
    ),
)


def _num(value: float | int) -> str:
    return f"{value:g}"


def _reading_block(reading: dict[str, Any]) -> str:
    rows = (
        ("feed temperature", f"{_num(reading['feed_temp_C'])} C"),
        ("feed pressure", f"{_num(reading['feed_pressure_bar'])} bar"),
        ("feed conductivity", f"{_num(reading['feed_conductivity_uS_cm'])} uS/cm"),
        ("permeate conductivity", f"{_num(reading['permeate_conductivity_uS_cm'])} uS/cm"),
        ("permeate flow", f"{_num(reading['permeate_flow_m3_h'])} m3/h"),
        (
            "differential pressure",
            f"lead {_num(reading['dp_lead_bar'])} bar, tail {_num(reading['dp_tail_bar'])} bar",
        ),
    )
    return "\n".join(f"  {label:<22}: {value}" for label, value in rows)


def _rule_line(cause: str, constraints: dict[str, tuple[str, ...]]) -> str:
    order = ("flow", "salt_passage", "dp", "stage")
    parts = [
        f"{field}={' or '.join(constraints[field])}"
        for field in order
        if field in constraints
    ]
    return f"  {cause:<18}: " + ", ".join(parts)


def _rules_block() -> str:
    return "\n".join(_rule_line(cause, c) for cause, c in RULES)


def _actions_block() -> str:
    return "\n".join(f"  {cause:<18}-> {action}" for cause, action in ACTIONS.items())


SCHEMA_EXAMPLE = """{
  "normalized_flow_change_pct": <number>,
  "salt_passage_change_pct": <number>,
  "dp_change_pct": <number>,
  "flags": {"flow": <string>, "salt_passage": <string>, "dp": <string>},
  "stage": <string>,
  "root_cause": <string>,
  "action": <string>
}"""


def build_user_prompt(record: dict[str, Any]) -> str:
    """Render the user turn for one operating record."""
    return f"""Reverse-osmosis train performance record.

Baseline (t0):
{_reading_block(record["t0"])}

Current (t1):
{_reading_block(record["t1"])}

Recovery is {_num(record["recovery_pct"])} % at both readings.
Vessel probing places the anomaly in the {record["anomaly_stage"].upper()} stage.

Step 1 -- compute three percent changes, each rounded to one decimal place.

  TCF(T) = 1.03 ** (25 - T)
  normalized_flow(t)  = permeate_flow(t) * TCF(feed_temperature(t))
  salt_passage(t)     = permeate_conductivity(t) / feed_conductivity(t) * 100
  dp(t)               = dp_lead(t) + dp_tail(t)

  normalized_flow_change_pct = (normalized_flow(t1) - normalized_flow(t0)) \
/ normalized_flow(t0) * 100
  salt_passage_change_pct    = (salt_passage(t1) - salt_passage(t0)) \
/ salt_passage(t0) * 100
  dp_change_pct              = (dp(t1) - dp(t0)) / dp(t0) * 100

Step 2 -- turn each change into a flag.

  flow          : down if <= {_num(FLOW_DOWN_PCT)}, up if >= +{_num(FLOW_UP_PCT)}, else flat
  salt_passage  : down if <= {_num(SP_DOWN_PCT)}, sharp_up if >= +{_num(SP_SHARP_UP_PCT)}, \
up if >= +{_num(SP_UP_PCT)}, else flat
  dp            : down if <= {_num(DP_DOWN_PCT)}, up if >= +{_num(DP_UP_PCT)}, else flat

Step 3 -- read the root cause off this table.

{_rules_block()}

Step 4 -- choose the action.

  If normalized_flow_change_pct <= {_num(SEVERE_FLOW_LOSS_PCT)}, the action is
  {SEVERE_ACTION} whatever the cause. Otherwise:

{_actions_block()}

Reply with exactly this JSON object and nothing else:

{SCHEMA_EXAMPLE}"""


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]
