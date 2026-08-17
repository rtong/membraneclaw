"""Tests for the frozen splits on disk, as opposed to the code that made them.

`test_generate.py` checks the generator. This file checks the artefact: the four
files that every number in the memo will eventually be traced back to. It is the
guard against the two ways a frozen evaluation quietly stops being frozen --
someone regenerates the data and forgets to say so, or a case leaks from test
into train.

Skips rather than fails when the data has not been generated yet, so a fresh
clone can run the suite before `python3 -m task.generate`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from task.generate import truth_from_record
from task.schema import canonical, validate

DATA = Path(__file__).resolve().parent / "data"
SPLITS = ("train", "dev", "test", "holdout_shift")

pytestmark = pytest.mark.skipif(
    not (DATA / "SHA256SUMS").exists(),
    reason="run `python3 -m task.generate` first",
)


def _load(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


def test_checksums_match_the_files_on_disk():
    """A regenerated split with a stale SHA256SUMS is the failure this catches."""
    expected = {}
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        expected[name] = digest

    assert set(expected) == {f"{split}.jsonl" for split in SPLITS}
    for name, digest in expected.items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} has changed since SHA256SUMS was written"


@pytest.mark.parametrize("split", SPLITS)
def test_every_shipped_answer_is_derivable_from_its_record(split):
    for case in _load(split):
        assert truth_from_record(case["record"]) == case["answer"], case["id"]


@pytest.mark.parametrize("split", SPLITS)
def test_every_shipped_answer_matches_the_schema(split):
    for case in _load(split):
        result = validate(case["answer"])
        assert result.ok, (case["id"], result.errors)


def test_ids_are_unique_across_every_split():
    ids = [case["id"] for split in SPLITS for case in _load(split)]
    assert len(ids) == len(set(ids))


def test_no_record_leaks_between_splits():
    seen: dict[str, str] = {}
    for split in SPLITS:
        for case in _load(split):
            key = canonical(case["record"])
            assert key not in seen, f"{case['id']} duplicates a case in {seen.get(key)}"
            seen[key] = split


@pytest.mark.parametrize("split", SPLITS)
def test_split_field_agrees_with_the_filename(split):
    assert all(case["split"] == split for case in _load(split))


def test_holdout_shift_carries_both_slices():
    slices = {case["slice"] for case in _load("holdout_shift")}
    assert slices == {"shift_temp", "shift_boundary"}
