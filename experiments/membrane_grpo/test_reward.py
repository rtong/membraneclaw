"""Tests for the reward, including what it pays for not solving the problem.

The adversarial half of this file is the important half. A reward function is
only as good as the cheapest way to score well on it, and the way to find that
out is to write the cheats down and price them before training rather than
after. Each `test_<strategy>_is_worth` below is a quantitative prediction: if a
training curve later parks on one of these numbers, the model found that
strategy.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import baselines
from reward import MAIN, PROBE, NUMERIC_TOLERANCE_PP, Weights, group_advantages, score
from task.decision_table import classify

DATA = Path(__file__).resolve().parent / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "dev.jsonl").exists(), reason="run `python3 -m task.generate` first"
)


def cases(split: str = "dev") -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


def one(tier: str | None = None) -> dict:
    for case in cases():
        if tier is None or case["tier"] == tier:
            return case
    raise AssertionError(f"no {tier} case in dev")


# --- calibration -------------------------------------------------------------


def test_weights_sum_to_one():
    assert MAIN.total() == pytest.approx(1.0)
    assert PROBE.total() == pytest.approx(1.0)


def test_the_answer_key_scores_exactly_one():
    for case in cases()[:40]:
        result = score(baselines.oracle(case["record"], case["answer"]), case["answer"])
        assert result.total == pytest.approx(1.0), case["id"]
        assert result.diagnostics["exact_match"]


def test_a_correct_answer_in_a_code_fence_loses_only_the_format_component():
    case = one()
    result = score(baselines.oracle_verbose(case["record"], case["answer"]), case["answer"])

    assert result.gate_passed
    assert result.total == pytest.approx(1.0 - MAIN.format)
    assert result.components["format"] == 0.0
    assert result.diagnostics["exact_match"], "content was right; only the schema was not"
    assert "extra:confidence" in result.schema_errors


# --- what the cheats are worth ------------------------------------------------


@pytest.mark.parametrize("name", ["empty", "prose", "schema_template"])
def test_ungradeable_completions_score_zero(name):
    case = one()
    result = score(baselines.STRATEGIES[name](case["record"], case["answer"]), case["answer"])

    assert not result.gate_passed
    assert result.total == 0.0
    assert result.parse_error is not None


def test_the_schema_template_is_not_valid_json():
    """It looks like the answer and parses as nothing. Worth pinning explicitly."""
    case = one()
    assert score(baselines.schema_template({}, {}), case["answer"]).parse_error == "no_json"


def test_constant_guessing_is_worth_less_than_a_third():
    """The floor. Any reward gain has to be read against this line, not against zero."""
    scored = [
        score(baselines.constant(c["record"], c["answer"]), c["answer"]).total for c in cases()
    ]
    mean = sum(scored) / len(scored)

    # Always-valid JSON collects `format` outright, plus luck on the balanced
    # seven-way label and the two-way stage.
    assert mean > MAIN.format, "a schema-perfect constant should at least bank the format credit"
    assert mean < 0.33, f"constant guessing scores {mean:.3f}; the reward is too easy"


def test_copying_the_stage_is_worth_its_weight_and_no_more():
    plain = [score(baselines.constant(c["record"], c["answer"]), c["answer"]).total for c in cases()]
    copied = [
        score(baselines.copy_stage_only(c["record"], c["answer"]), c["answer"]).total
        for c in cases()
    ]
    gain = sum(copied) / len(copied) - sum(plain) / len(plain)

    # `constant` already guesses one of two stages, so copying buys the half it
    # was getting wrong.
    assert 0.3 * MAIN.stage < gain < 0.7 * MAIN.stage, f"gain {gain:.4f}"


def test_skipping_the_correction_is_free_on_easy_and_costly_on_hard():
    """The sharpest prediction available about a model that learns only the easy tier."""
    by_tier: dict[str, list[float]] = {"easy": [], "hard": []}
    for case in cases():
        result = score(
            baselines.skip_correction(case["record"], case["answer"]), case["answer"]
        )
        by_tier[case["tier"]].append(result.total)

    easy = sum(by_tier["easy"]) / len(by_tier["easy"])
    hard = sum(by_tier["hard"]) / len(by_tier["hard"])

    assert easy == pytest.approx(1.0), "TCF cancels on easy cases; this should be exactly right"
    assert hard < 0.75, f"hard tier scores {hard:.3f}; the correction is not doing enough work"


# --- the cascade --------------------------------------------------------------


def test_one_wrong_number_costs_far_more_than_its_weight():
    """A self-consistent answer built on one bad figure loses 0.69 of 1.0.

    This is why `numeric` cannot be read as a 0.15-weight component and why
    `root_cause` accuracy is not a clean measure of table-reading.
    """
    case = next(c for c in cases() if c["answer"]["flags"]["dp"] == "up")
    answer = case["answer"]

    # Same model, one arithmetic slip, then everything downstream derived
    # correctly *from the slip*.
    bad_dp = 5.0
    derived = classify(
        answer["normalized_flow_change_pct"],
        answer["salt_passage_change_pct"],
        bad_dp,
        answer["stage"],
    )
    completion = json.dumps(
        {
            **answer,
            "dp_change_pct": bad_dp,
            "flags": derived.flags,
            "root_cause": derived.root_cause,
            "action": derived.action,
        }
    )

    result = score(completion, answer)
    lost = 1.0 - result.total
    one_number = MAIN.numeric / 3

    assert result.diagnostics["numeric_correct"] == 2
    assert not result.diagnostics["root_cause_correct"]
    assert lost == pytest.approx(one_number + MAIN.flags / 3 + MAIN.root_cause + MAIN.action)
    assert lost > 8 * one_number, f"one number cost {lost:.3f}, {lost / one_number:.0f}x its weight"


def test_cause_given_flags_isolates_the_lookup():
    case = one()
    answer = case["answer"]

    # Flags right, cause wrong: the lookup itself failed.
    wrong_row = json.dumps(dict(answer, root_cause="oxidation_damage"))
    assert score(wrong_row, answer).diagnostics["cause_given_flags"] is False

    # Flags wrong: the lookup was never reached, so it is not scored.
    wrong_flags = json.dumps(dict(answer, flags={**answer["flags"], "dp": "down"}))
    assert score(wrong_flags, answer).diagnostics["cause_given_flags"] is None

    assert score(json.dumps(answer), answer).diagnostics["cause_given_flags"] is True


# --- tolerance ----------------------------------------------------------------


def test_the_numeric_tolerance_is_inclusive():
    case = one()
    answer = case["answer"]
    target = answer["dp_change_pct"]

    inside = json.dumps(dict(answer, dp_change_pct=target + NUMERIC_TOLERANCE_PP))
    outside = json.dumps(dict(answer, dp_change_pct=target + NUMERIC_TOLERANCE_PP + 0.01))

    assert score(inside, answer).diagnostics["numeric_correct"] == 3
    assert score(outside, answer).diagnostics["numeric_correct"] == 2


def test_a_stringified_number_earns_nothing_numeric():
    case = one()
    answer = case["answer"]
    completion = json.dumps(dict(answer, dp_change_pct=str(answer["dp_change_pct"])))

    result = score(completion, answer)
    assert result.diagnostics["numeric_correct"] == 2
    assert not result.diagnostics["schema_ok"]


# --- the probe control ---------------------------------------------------------


def test_the_probe_weights_pay_more_for_form_and_less_for_the_diagnosis():
    case = one()
    answer = case["answer"]

    # Right shape, right arithmetic, wrong diagnosis.
    shallow = json.dumps(
        dict(answer, root_cause="oxidation_damage", action="check_oxidant_and_replace_elements")
    )

    assert score(shallow, answer, PROBE).total > score(shallow, answer, MAIN).total
    assert score(shallow, answer, PROBE).total > 0.8
    assert score(shallow, answer, MAIN).total < 0.5


# --- group advantages ----------------------------------------------------------


def test_a_group_that_all_scores_the_same_produces_no_gradient():
    advantages, degenerate = group_advantages([0.4] * 8)
    assert degenerate
    assert advantages == [0.0] * 8


def test_advantages_are_standardised():
    advantages, degenerate = group_advantages([0.0, 0.25, 0.5, 0.75, 1.0])
    assert not degenerate
    assert sum(advantages) == pytest.approx(0.0)
    assert max(advantages) > 0 > min(advantages)


def test_custom_weights_still_grade_consistently():
    case = one()
    flat = Weights(format=1.0, numeric=0.0, flags=0.0, stage=0.0, root_cause=0.0, action=0.0)
    result = score(baselines.oracle(case["record"], case["answer"]), case["answer"], flat)
    assert result.total == pytest.approx(1.0)
