"""Tests for the answer key itself.

If this file is wrong, every reward in the experiment is wrong and nothing
downstream can detect it -- the reward function has no second opinion to check
against. So the table gets tested harder than the code that uses it.
"""
from __future__ import annotations

import itertools

import pytest

from task.decision_table import (
    ACTION_SET,
    ACTIONS,
    CAUSES,
    COVERED,
    SEVERE_ACTION,
    SEVERE_FLOW_LOSS_PCT,
    STAGES,
    UncoveredSymptoms,
    classify,
    dp_flag,
    flow_flag,
    keys_for_cause,
    salt_passage_flag,
)

FLOW_LEVELS = ("down", "flat", "up")
SP_LEVELS = ("down", "flat", "up", "sharp_up")
DP_LEVELS = ("down", "flat", "up")


def test_every_cause_is_reachable():
    reachable = set(COVERED.values())
    assert reachable == set(CAUSES), f"unreachable causes: {set(CAUSES) - reachable}"


def test_every_cause_has_an_action():
    assert set(ACTIONS) == set(CAUSES)
    assert SEVERE_ACTION not in ACTIONS.values(), "the override must be distinguishable"
    assert set(ACTION_SET) == {*ACTIONS.values(), SEVERE_ACTION}


def test_table_keys_are_well_formed():
    for flow, sp, dp, stage in COVERED:
        assert flow in FLOW_LEVELS
        assert sp in SP_LEVELS
        assert dp in DP_LEVELS
        assert stage in STAGES


def test_uncovered_combinations_raise_rather_than_guess():
    covered = set(COVERED)
    uncovered = [
        key
        for key in itertools.product(FLOW_LEVELS, SP_LEVELS, DP_LEVELS, STAGES)
        if key not in covered
    ]
    assert uncovered, "the table would be total; this test would be vacuous"

    # A representative one: flow down with salt passage down and dp up is not a
    # pattern the table claims to explain.
    with pytest.raises(UncoveredSymptoms):
        classify(-20.0, -30.0, +30.0, "tail")


def test_flag_boundaries_are_inclusive_on_the_named_side():
    assert flow_flag(-10.0) == "down"
    assert flow_flag(-9.99) == "flat"
    assert flow_flag(10.0) == "up"
    assert flow_flag(9.99) == "flat"

    assert salt_passage_flag(-15.0) == "down"
    assert salt_passage_flag(15.0) == "up"
    assert salt_passage_flag(49.99) == "up"
    assert salt_passage_flag(50.0) == "sharp_up"

    assert dp_flag(-15.0) == "down"
    assert dp_flag(15.0) == "up"
    assert dp_flag(0.0) == "flat"


def test_scaling_and_biofouling_are_separated_only_by_stage():
    """The one distinction the three trends cannot make on their own."""
    tail = classify(-18.0, +25.0, +30.0, "tail")
    lead = classify(-18.0, +25.0, +30.0, "lead")
    assert tail.flags == lead.flags
    assert tail.root_cause == "scaling"
    assert lead.root_cause == "biofouling"


def test_biofouling_and_colloidal_are_separated_only_by_salt_passage():
    bio = classify(-18.0, +25.0, +30.0, "lead")
    colloidal = classify(-18.0, +5.0, +30.0, "lead")
    assert bio.root_cause == "biofouling"
    assert colloidal.root_cause == "colloidal_fouling"


def test_severe_flow_loss_overrides_the_action_but_not_the_cause():
    mild = classify(-20.0, +25.0, +30.0, "tail")
    severe = classify(-40.0, +25.0, +30.0, "tail")

    assert mild.root_cause == severe.root_cause == "scaling"
    assert mild.action == ACTIONS["scaling"]
    assert severe.action == SEVERE_ACTION


def test_severity_threshold_is_inclusive():
    assert classify(SEVERE_FLOW_LOSS_PCT, +25.0, +30.0, "tail").action == SEVERE_ACTION
    assert classify(-29.99, +25.0, +30.0, "tail").action == ACTIONS["scaling"]


def test_action_is_not_a_pure_function_of_cause():
    """Otherwise the action field would carry no information beyond root_cause."""
    actions_seen = {
        classify(-20.0, +25.0, +30.0, "tail").action,
        classify(-40.0, +25.0, +30.0, "tail").action,
    }
    assert len(actions_seen) == 2


def test_keys_for_cause_is_the_inverse_of_the_table():
    for cause in CAUSES:
        keys = keys_for_cause(cause)
        assert keys, f"{cause} has no symptom combination"
        assert all(COVERED[key] == cause for key in keys)
    assert sum(len(keys_for_cause(cause)) for cause in CAUSES) == len(COVERED)


def test_stage_must_be_valid():
    with pytest.raises(ValueError):
        classify(-18.0, +25.0, +30.0, "middle")
