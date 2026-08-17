"""Tests that a generated case is consistent with its own label.

The generator constructs cases backwards from an answer, which makes one failure
mode both easy to hit and invisible: a record whose rounded readings no longer
imply the label that was used to build it. Every test here is ultimately the
same assertion -- recomputing the answer from the record must give the answer
that shipped with it.
"""
from __future__ import annotations

import random

import pytest

from task.decision_table import (
    ACTIONS,
    CAUSES,
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
from task.generate import (
    BOUNDARY_MARGIN_PP,
    DEFAULT_MARGIN_PP,
    SHIFT_TEMP_RANGES,
    TRAIN_TEMP_RANGE,
    build_split,
    generate_case,
    tcf,
    truth_from_record,
)
from task.schema import validate

TIERS = ("easy", "hard")

# Rounding the readings moves the recomputed percentages off the sampled values,
# and the answer itself is rounded to one decimal. Measured over 140 cases the
# closest survivor sits exactly `margin` away in both slices, so the drift the
# resample loop lets through is under a rounding step. One step of slack is
# enough; anything looser stops testing the margin at all.
ROUNDING_SLACK_PP = 0.15


def _distance_to_thresholds(answer: dict) -> float:
    checks = (
        (answer["normalized_flow_change_pct"], (FLOW_DOWN_PCT, FLOW_UP_PCT, SEVERE_FLOW_LOSS_PCT)),
        (answer["salt_passage_change_pct"], (SP_DOWN_PCT, SP_UP_PCT, SP_SHARP_UP_PCT)),
        (answer["dp_change_pct"], (DP_DOWN_PCT, DP_UP_PCT)),
    )
    return min(abs(value - t) for value, thresholds in checks for t in thresholds)


@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("tier", TIERS)
def test_every_cause_and_tier_is_constructible(cause, tier):
    record, answer, meta = generate_case(random.Random(hash((cause, tier)) % 2**32), cause, tier)
    assert answer["root_cause"] == cause
    assert meta["intended_cause"] == cause
    assert validate(answer).ok, validate(answer).errors


@pytest.mark.parametrize("cause", CAUSES)
def test_answer_is_reproducible_from_the_record_alone(cause):
    rng = random.Random(7)
    for _ in range(20):
        record, answer, _ = generate_case(rng, cause, "hard")
        assert truth_from_record(record) == answer


def test_easy_tier_cancels_the_temperature_correction():
    rng = random.Random(11)
    for cause in CAUSES:
        record, answer, _ = generate_case(rng, cause, "easy")
        assert record["t0"]["feed_temp_C"] == record["t1"]["feed_temp_C"]

        # With equal temperatures the correction factor divides out, so the
        # normalized change is the raw flow change. Compared with a tolerance
        # rather than exactly: (Q1*k - Q0*k)/(Q0*k) and (Q1 - Q0)/Q0 agree only
        # to within floating-point noise, which is enough to flip `round` when
        # the true value lands on a .x5 boundary. The reward tolerance is 0.5 pp,
        # so a last-digit disagreement is below anything the pipeline resolves.
        raw = (
            (record["t1"]["permeate_flow_m3_h"] - record["t0"]["permeate_flow_m3_h"])
            / record["t0"]["permeate_flow_m3_h"]
            * 100.0
        )
        assert raw == pytest.approx(answer["normalized_flow_change_pct"], abs=0.06)


def test_hard_tier_actually_needs_the_correction():
    """The uncorrected answer must be wrong by more than the reward tolerance."""
    rng = random.Random(13)
    differences = []
    for cause in CAUSES:
        record, answer, _ = generate_case(rng, cause, "hard")
        assert record["t0"]["feed_temp_C"] != record["t1"]["feed_temp_C"]
        raw = (
            (record["t1"]["permeate_flow_m3_h"] - record["t0"]["permeate_flow_m3_h"])
            / record["t0"]["permeate_flow_m3_h"]
            * 100.0
        )
        differences.append(abs(raw - answer["normalized_flow_change_pct"]))

    assert min(differences) > 0.5, "a hard case that ignoring TCF would still get right"


def test_feed_pressure_is_a_distractor():
    """It appears in both readings and is never used; it must not move."""
    rng = random.Random(17)
    for cause in CAUSES:
        record, _, _ = generate_case(rng, cause, "hard")
        assert record["t0"]["feed_pressure_bar"] == record["t1"]["feed_pressure_bar"]


def test_salt_passage_is_a_real_division():
    """Feed conductivity must drift, or salt passage collapses to a permeate ratio."""
    rng = random.Random(19)
    moved = 0
    for _ in range(40):
        record, _, _ = generate_case(rng, "scaling", "easy")
        if record["t0"]["feed_conductivity_uS_cm"] != record["t1"]["feed_conductivity_uS_cm"]:
            moved += 1
    assert moved > 30


def test_no_case_sits_on_a_threshold():
    cases = build_split(140, seed=23, split="probe")
    worst = min(_distance_to_thresholds(case["answer"]) for case in cases)
    assert worst >= DEFAULT_MARGIN_PP - ROUNDING_SLACK_PP, f"closest case is {worst:.2f} pp away"


def test_boundary_slice_is_deliberately_closer():
    """Closer to the thresholds than the main slice ever gets, but still off them.

    Both bounds matter. Without the lower one the slice could degenerate into
    coin-flip labels, which would make a poor score there uninterpretable rather
    than informative.
    """
    cases = build_split(140, seed=29, split="probe", margin=BOUNDARY_MARGIN_PP)
    worst = min(_distance_to_thresholds(case["answer"]) for case in cases)
    assert worst < DEFAULT_MARGIN_PP - ROUNDING_SLACK_PP, "the hard slice is not actually harder"
    assert worst >= BOUNDARY_MARGIN_PP - ROUNDING_SLACK_PP, f"label is a coin toss at {worst:.2f}"


def test_shift_slice_leaves_the_training_temperature_range():
    cases = build_split(
        40, seed=31, split="probe", temp_ranges=SHIFT_TEMP_RANGES, hard_only=True
    )
    for case in cases:
        temp = case["record"]["t0"]["feed_temp_C"]
        assert not TRAIN_TEMP_RANGE[0] <= temp <= TRAIN_TEMP_RANGE[1]
        assert case["tier"] == "hard"


def test_labels_are_close_to_balanced():
    cases = build_split(210, seed=37, split="probe")
    counts = {cause: 0 for cause in CAUSES}
    for case in cases:
        counts[case["answer"]["root_cause"]] += 1
    assert set(counts.values()) == {30}, counts


def test_severe_override_appears_but_does_not_dominate():
    cases = build_split(210, seed=41, split="probe")
    severe = [c for c in cases if c["answer"]["action"] == SEVERE_ACTION]
    assert 0.05 < len(severe) / len(cases) < 0.30, f"{len(severe)}/{len(cases)}"
    for case in severe:
        assert case["answer"]["normalized_flow_change_pct"] <= SEVERE_FLOW_LOSS_PCT
    for case in cases:
        if case["answer"]["normalized_flow_change_pct"] > SEVERE_FLOW_LOSS_PCT:
            assert case["answer"]["action"] == ACTIONS[case["answer"]["root_cause"]]


def test_splits_are_reproducible_from_the_seed():
    assert build_split(30, seed=43, split="probe") == build_split(30, seed=43, split="probe")
    assert build_split(30, seed=43, split="probe") != build_split(30, seed=44, split="probe")


def test_tcf_matches_the_formula_quoted_in_the_prompt():
    assert tcf(25.0) == pytest.approx(1.0)
    assert tcf(15.0) == pytest.approx(1.03**10)
    assert tcf(30.0) == pytest.approx(1.03**-5)
