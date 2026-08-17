"""Tests for the parser's leniency and the validator's strictness.

These pull in opposite directions on purpose, and the tests are written to pin
down exactly where the line is. `parse_answer` is the reward gate, so anything
it rejects scores zero no matter how good the content was; `validate` only costs
the format component.
"""
from __future__ import annotations

import json

import pytest

from task.schema import ANSWER_KEYS, canonical, parse_answer, validate

GOOD = {
    "normalized_flow_change_pct": -14.3,
    "salt_passage_change_pct": 38.0,
    "dp_change_pct": 22.1,
    "flags": {"flow": "down", "salt_passage": "up", "dp": "up"},
    "stage": "tail",
    "root_cause": "scaling",
    "action": "acid_clean_low_ph",
}


def test_a_bare_object_parses():
    assert parse_answer(json.dumps(GOOD)).obj == GOOD


def test_a_fenced_object_parses():
    completion = "```json\n" + json.dumps(GOOD) + "\n```"
    assert parse_answer(completion).obj == GOOD


def test_an_object_after_preamble_parses():
    completion = "Here is my analysis:\n\n" + json.dumps(GOOD) + "\n\nHope that helps."
    assert parse_answer(completion).obj == GOOD


def test_braces_inside_strings_do_not_confuse_the_scanner():
    payload = dict(GOOD, root_cause="scaling")
    completion = 'The rule was "if {flow down} then": ' + json.dumps(payload)
    assert parse_answer(completion).obj == payload


@pytest.mark.parametrize(
    "completion,error",
    [
        ("", "empty"),
        ("   \n ", "empty"),
        ("The membrane is scaled.", "no_json"),
        ("{not: valid json,,}", "no_json"),
        ("[1, 2, 3]", "not_an_object"),
    ],
)
def test_unparseable_completions_report_why(completion, error):
    result = parse_answer(completion)
    assert result.obj is None
    assert result.error == error


def test_a_correct_answer_validates():
    assert validate(GOOD).ok


def test_missing_and_extra_keys_are_both_reported():
    obj = {k: v for k, v in GOOD.items() if k != "action"}
    obj["confidence"] = 0.9
    errors = validate(obj).errors
    assert "missing:action" in errors
    assert "extra:confidence" in errors


def test_a_stringified_number_is_not_a_number():
    obj = dict(GOOD, dp_change_pct="22.1")
    assert "not_a_number:dp_change_pct" in validate(obj).errors


def test_booleans_are_not_numbers():
    obj = dict(GOOD, dp_change_pct=True)
    assert "not_a_number:dp_change_pct" in validate(obj).errors


def test_integers_are_numbers():
    assert validate(dict(GOOD, dp_change_pct=22)).ok


def test_out_of_vocabulary_labels_are_rejected():
    assert "bad_value:root_cause" in validate(dict(GOOD, root_cause="fouling")).errors
    assert "bad_value:stage" in validate(dict(GOOD, stage="middle")).errors
    assert "bad_value:action" in validate(dict(GOOD, action="clean it")).errors


def test_flag_problems_are_reported_per_field():
    obj = dict(GOOD, flags={"flow": "downward", "dp": "up", "extra": 1})
    errors = validate(obj).errors
    assert "flags.bad_value:flow" in errors
    assert "flags.missing:salt_passage" in errors
    assert "flags.extra:extra" in errors


def test_sharp_up_is_only_valid_for_salt_passage():
    assert validate(dict(GOOD, flags={**GOOD["flags"], "salt_passage": "sharp_up"})).ok
    assert not validate(dict(GOOD, flags={**GOOD["flags"], "flow": "sharp_up"})).ok


def test_validation_is_not_the_gate():
    """A nearly-right object still parses, so it can still earn partial credit."""
    obj = dict(GOOD, confidence=0.9)
    assert parse_answer(json.dumps(obj)).obj is not None
    assert not validate(obj).ok


def test_canonical_is_order_independent():
    shuffled = {key: GOOD[key] for key in reversed(ANSWER_KEYS)}
    assert canonical(shuffled) == canonical(GOOD)
