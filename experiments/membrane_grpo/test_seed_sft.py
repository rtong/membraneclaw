"""Tests for the seed's data handling. No model, no GPU.

The shuffled seed is only a control if two things are exactly true: every label
occurs as often as it does in the correct seed, and the mapping from each
record's flags to its label is broken. Both are checked here rather than
assumed, along with the mask and the seed data's own labels.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from seed_sft import SLOTS, is_dead, shuffle_labels, target_text, value_spans
from task.decision_table import ACTION_SET, CAUSES, SEVERE_ACTION
from task.generate import truth_from_record

SEED_CASES = Path(__file__).resolve().parent / "seed" / "seed_cases.jsonl"


@pytest.fixture(scope="module")
def cases():
    return [json.loads(line) for line in SEED_CASES.read_text().splitlines() if line.strip()]


def test_every_seed_label_is_what_the_grader_derives(cases):
    """The seed carries no label the grader would not itself produce."""
    assert len(cases) == 387
    assert all(truth_from_record(c["record"]) == c["answer"] for c in cases)


def test_shuffling_keeps_every_label_count_exactly(cases):
    shuffled = shuffle_labels(cases, seed=0)
    for slot in SLOTS:
        assert Counter(c["answer"][slot] for c in shuffled) == Counter(
            c["answer"][slot] for c in cases
        )
    # which also means the dead-label loss weight applies to as many cases
    assert sum(is_dead(c["answer"]) for c in shuffled) == sum(is_dead(c["answer"]) for c in cases)


def test_shuffling_touches_nothing_but_the_two_slots(cases):
    shuffled = shuffle_labels(cases, seed=0)
    for before, after in zip(cases, shuffled):
        assert after["record"] == before["record"]
        for key in before["answer"]:
            if key not in SLOTS:
                assert after["answer"][key] == before["answer"][key], key


def test_shuffling_breaks_the_mapping(cases):
    """Most labels must move, or the control is mostly the correct seed.

    With these label frequencies a random permutation leaves roughly the sum of
    squared shares in place -- about a fifth for both slots.
    """
    shuffled = shuffle_labels(cases, seed=0)
    for slot in SLOTS:
        kept = sum(a["answer"][slot] == b["answer"][slot] for a, b in zip(cases, shuffled))
        assert kept / len(cases) < 0.30, (slot, kept)


def test_shuffling_is_deterministic_and_does_not_modify_its_input(cases):
    snapshot = json.dumps(cases)
    assert shuffle_labels(cases, seed=3) == shuffle_labels(cases, seed=3)
    assert shuffle_labels(cases, seed=3) != shuffle_labels(cases, seed=4)
    assert json.dumps(cases) == snapshot


def test_the_mask_covers_exactly_the_two_slot_values(cases):
    answer = cases[0]["answer"]
    text = target_text(answer)
    spans = value_spans(text)
    assert [text[a:b] for a, b in spans] == sorted(
        [answer["action"], answer["root_cause"]], key=lambda v: text.index(f'"{v}"')
    )
    assert {text[a:b] for a, b in spans} <= set(CAUSES) | set(ACTION_SET)


def test_the_mask_ignores_anything_before_the_offset():
    """The prompt's schema example names both keys; they must not be supervised."""
    answer = {
        "action": SEVERE_ACTION, "dp_change_pct": 1.0,
        "flags": {"dp": "flat", "flow": "down", "salt_passage": "flat"},
        "normalized_flow_change_pct": -31.0, "root_cause": "compaction",
        "salt_passage_change_pct": 2.0, "stage": "lead",
    }
    prefix = 'reply with {"root_cause": "x", "action": "y"}\n'
    full = prefix + target_text(answer)
    spans = value_spans(full, offset=len(prefix))
    assert [full[a:b] for a, b in spans] == [SEVERE_ACTION, "compaction"]


def test_the_target_states_the_correct_flags_before_either_supervised_value(cases):
    """The property that makes the correct seed a lookup and not only vocabulary."""
    answer = cases[0]["answer"]
    text = target_text(answer)
    first_value = min(a for a, _ in value_spans(text))
    flags = answer["flags"]
    assert text.index(f"Flow {flags['flow']}, salt passage {flags['salt_passage']}") < first_value
