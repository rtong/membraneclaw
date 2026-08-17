"""The answer key: normalized symptom flags -> root cause and corrective action.

RO troubleshooting guides all carry some version of the same table. Three
normalized trends -- permeate flow, salt passage, differential pressure -- plus
where in the train the anomaly sits, and the combination narrows the cause to
one of a handful. This module is that table, written out in full and made
deterministic so it can serve as a reward signal.

Determinism is the whole point. Every reward in this experiment is computed by
`classify()` and nothing else: no LLM judge, no fuzzy matching. That means the
table must be *total* over the symptom combinations the generator emits and
*silent* about the rest -- an uncovered combination raises rather than guessing,
so a generator bug shows up as a crash instead of a quietly wrong label.

Like `toy_mdp`, the thresholds here are teaching parameters chosen by hand. They
are in the right ballpark for a brackish-water RO train, but nothing in this file
came from plant data and none of it is a substitute for a vendor's guide.
"""
from __future__ import annotations

from typing import NamedTuple

# --- flag thresholds, in percent change from baseline -----------------------
# These are stated verbatim in the prompt, so the model is not being asked to
# guess them. Getting a flag right is therefore a reading-comprehension task on
# top of the arithmetic, not a second inference.
FLOW_DOWN_PCT = -10.0
FLOW_UP_PCT = 10.0

SP_DOWN_PCT = -15.0
SP_UP_PCT = 15.0
SP_SHARP_UP_PCT = 50.0

DP_DOWN_PCT = -15.0
DP_UP_PCT = 15.0

# Past this much normalized flow loss the element is treated as spent: cleaning
# chemistry is no longer the first move, whatever the cause. This is what stops
# `action` from being a pure lookup on `root_cause` -- see ACTIONS below.
SEVERE_FLOW_LOSS_PCT = -30.0

STAGES = ("lead", "tail")

CAUSES = (
    "scaling",
    "biofouling",
    "colloidal_fouling",
    "organic_fouling",
    "compaction",
    "oxidation_damage",
    "mechanical_leak",
)

# One default action per cause. Balanced labels mean a constant guesser scores
# 1/7 = 14.3% on either field, which keeps both worth learning.
ACTIONS = {
    "scaling": "acid_clean_low_ph",
    "biofouling": "alkaline_clean_and_sanitize",
    "colloidal_fouling": "alkaline_clean_and_check_sdi",
    "organic_fouling": "alkaline_clean_high_ph",
    "compaction": "no_clean_evaluate_replacement",
    "oxidation_damage": "check_oxidant_and_replace_elements",
    "mechanical_leak": "probe_vessels_and_replace_orings",
}

SEVERE_ACTION = "isolate_and_evaluate_replacement"

ACTION_SET = tuple(sorted({*ACTIONS.values(), SEVERE_ACTION}))


class UncoveredSymptoms(KeyError):
    """The flag combination has no entry in the table.

    Raised rather than defaulted. A missing entry means either the generator
    produced a case the table was never meant to cover, or the table lost a row;
    both are bugs, and neither should be papered over with a fallback label.
    """


# --- the table --------------------------------------------------------------
# Keyed by (flow, salt_passage, dp, stage). Written out one row at a time
# instead of as nested conditionals: 17 rows is small enough to read, and a flat
# dict makes "is this combination covered?" a membership test rather than an
# argument about rule precedence.
#
# The two disambiguations worth naming:
#
#   * Scaling and biofouling look identical in the three trends -- flow down, dp
#     up, salt passage up. Location separates them. Scale precipitates where the
#     brine is most concentrated (tail); biofilm grows where the nutrients enter
#     (lead).
#   * Biofouling and colloidal fouling both foul the lead stage. Biofilm drives
#     salt passage up through enhanced concentration polarisation; colloidal
#     matter mostly just plugs the feed channel, so dp climbs while salt passage
#     stays put.
COVERED: dict[tuple[str, str, str, str], str] = {
    # tail-end deposition: dp up, flow down, whatever salt passage does
    ("down", "flat", "up", "tail"): "scaling",
    ("down", "up", "up", "tail"): "scaling",
    ("down", "sharp_up", "up", "tail"): "scaling",
    # lead-end biofilm: same trends, front of the train, salt passage moves
    ("down", "up", "up", "lead"): "biofouling",
    ("down", "sharp_up", "up", "lead"): "biofouling",
    # lead-end plugging: dp up but the membrane surface is still rejecting
    ("down", "flat", "up", "lead"): "colloidal_fouling",
    # adsorbed organics: flow lost with no channel blockage, rejection improves
    ("down", "down", "flat", "lead"): "organic_fouling",
    ("down", "down", "flat", "tail"): "organic_fouling",
    # compaction: flow lost, channel clear, rejection flat or slightly worse
    ("down", "flat", "flat", "lead"): "compaction",
    ("down", "flat", "flat", "tail"): "compaction",
    ("down", "up", "flat", "lead"): "compaction",
    ("down", "up", "flat", "tail"): "compaction",
    # oxidation: the barrier layer is damaged, so flow *rises* as rejection dies
    ("up", "sharp_up", "flat", "lead"): "oxidation_damage",
    ("up", "sharp_up", "flat", "tail"): "oxidation_damage",
    ("up", "sharp_up", "down", "lead"): "oxidation_damage",
    # mechanical bypass: feed short-circuits to permeate, flow barely moves
    ("flat", "sharp_up", "flat", "lead"): "mechanical_leak",
    ("flat", "sharp_up", "flat", "tail"): "mechanical_leak",
}


def flow_flag(pct: float) -> str:
    """Trend flag for normalized permeate flow change."""
    if pct <= FLOW_DOWN_PCT:
        return "down"
    if pct >= FLOW_UP_PCT:
        return "up"
    return "flat"


def salt_passage_flag(pct: float) -> str:
    """Trend flag for salt passage change.

    Four levels, not three. A 20% rise in salt passage and a doubling of it mean
    different things -- the first is fouling, the second is a hole -- and
    collapsing them would make oxidation damage and mechanical leaks
    indistinguishable from ordinary fouling.
    """
    if pct >= SP_SHARP_UP_PCT:
        return "sharp_up"
    if pct >= SP_UP_PCT:
        return "up"
    if pct <= SP_DOWN_PCT:
        return "down"
    return "flat"


def dp_flag(pct: float) -> str:
    """Trend flag for differential-pressure change across the train."""
    if pct <= DP_DOWN_PCT:
        return "down"
    if pct >= DP_UP_PCT:
        return "up"
    return "flat"


class Diagnosis(NamedTuple):
    flags: dict[str, str]
    root_cause: str
    action: str


def flags_for(flow_pct: float, sp_pct: float, dp_pct: float) -> dict[str, str]:
    return {
        "flow": flow_flag(flow_pct),
        "salt_passage": salt_passage_flag(sp_pct),
        "dp": dp_flag(dp_pct),
    }


def classify(flow_pct: float, sp_pct: float, dp_pct: float, stage: str) -> Diagnosis:
    """Map three normalized percent changes and a stage to the answer key.

    Raises `UncoveredSymptoms` if the resulting flag combination is not in the
    table, and `ValueError` for a stage outside `STAGES`.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    flags = flags_for(flow_pct, sp_pct, dp_pct)
    key = (flags["flow"], flags["salt_passage"], flags["dp"], stage)
    try:
        cause = COVERED[key]
    except KeyError:
        raise UncoveredSymptoms(key) from None

    action = SEVERE_ACTION if flow_pct <= SEVERE_FLOW_LOSS_PCT else ACTIONS[cause]
    return Diagnosis(flags=flags, root_cause=cause, action=action)


def keys_for_cause(cause: str) -> list[tuple[str, str, str, str]]:
    """Every symptom combination the table maps to `cause`.

    The generator works backwards from a cause, so it needs this inverse.
    """
    return [key for key, value in COVERED.items() if value == cause]
