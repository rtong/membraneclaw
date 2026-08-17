"""Tests that the prompt describes the procedure the reward actually grades.

The prompt compresses the table's 17 rows into 7 lines. If that compression is
lossy in either direction, the model is being shown one procedure and scored
against another -- and the symptom would be a training run that plateaus for no
visible reason. This is the cheapest test in the project and the one most likely
to catch a silent disaster.
"""
from __future__ import annotations

import random
import re

from task.decision_table import ACTION_SET, CAUSES, COVERED, SEVERE_ACTION
from task.generate import generate_case
from task.prompt import RULES, build_messages, build_user_prompt
from task.schema import ANSWER_KEYS


def _rule_matches(constraints: dict[str, tuple[str, ...]], key: tuple[str, ...]) -> bool:
    flow, sp, dp, stage = key
    got = {"flow": flow, "salt_passage": sp, "dp": dp, "stage": stage}
    return all(got[field] in allowed for field, allowed in constraints.items())


def test_prompt_rules_reproduce_the_table_exactly():
    for key, cause in COVERED.items():
        matched = [name for name, constraints in RULES if _rule_matches(constraints, key)]
        assert matched == [cause], f"{key}: table says {cause}, prompt says {matched}"


def test_prompt_rules_cover_every_cause_once():
    assert [cause for cause, _ in RULES] == list(CAUSES)


def test_prompt_names_every_label_the_model_may_need():
    record = generate_case(random.Random(0), "scaling", "hard")[0]
    text = build_user_prompt(record)

    for cause in CAUSES:
        assert cause in text, f"{cause} is gradeable but never mentioned"
    for action in ACTION_SET:
        assert action in text, f"{action} is gradeable but never mentioned"
    assert SEVERE_ACTION in text
    for key in ANSWER_KEYS:
        assert key in text, f"{key} is required by the schema but not shown"


def test_prompt_shows_every_reading_the_answer_depends_on():
    record = generate_case(random.Random(1), "biofouling", "hard")[0]
    text = build_user_prompt(record)

    for reading in (record["t0"], record["t1"]):
        for field in (
            "feed_temp_C",
            "feed_conductivity_uS_cm",
            "permeate_conductivity_uS_cm",
            "permeate_flow_m3_h",
            "dp_lead_bar",
            "dp_tail_bar",
        ):
            assert f"{reading[field]:g}" in text, f"{field}={reading[field]} missing"


def test_both_tiers_use_an_identical_template():
    """Tier difficulty must be arithmetic, not a change in wording.

    `colloidal_fouling` has exactly one symptom combination, so the stage is
    fixed and the two prompts can differ in nothing but their numbers.
    """
    easy = build_user_prompt(generate_case(random.Random(2), "colloidal_fouling", "easy")[0])
    hard = build_user_prompt(generate_case(random.Random(2), "colloidal_fouling", "hard")[0])

    def skeleton(text: str) -> str:
        return re.sub(r"[-+0-9.]+", "#", text)

    assert skeleton(easy) == skeleton(hard)


def test_messages_are_a_system_user_pair():
    record = generate_case(random.Random(3), "oxidation_damage", "easy")[0]
    messages = build_messages(record)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(m["content"].strip() for m in messages)
