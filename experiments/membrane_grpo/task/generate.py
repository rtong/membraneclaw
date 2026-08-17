"""Synthetic operating records, generated backwards from the answer.

Forward generation -- sample plausible readings, then see what they diagnose --
gives an unusable label distribution: most random readings land on flag
combinations the table does not cover, and the ones that do land are dominated
by whichever cause happens to occupy the most volume. So this goes the other
way. Pick a cause, pick one of its symptom combinations, pick percent changes
inside each flag's band, and solve backwards for the t1 readings that produce
them.

Two rules keep that honest:

**Ground truth is always recomputed from the finished record.** The percent
changes that go into the answer are not the ones that were sampled -- they are
re-derived from the rounded readings the model will actually see. A record whose
rounded numbers no longer produce the intended flags is thrown away and
resampled rather than shipped with a label its own numbers contradict.

**Every sampled change sits at least `margin_pp` away from a threshold.** A case
sitting at -10.02% flow change would have a label that a hundredth of a
percentage point of rounding could flip, which makes it a coin toss rather than
a question. The default margin is 2 pp. The `shift_boundary` slice deliberately
drops it to 0.6 pp -- those cases are *supposed* to be hard, and they are held
out rather than trained on.

Every number here is a teaching parameter. The ranges are in the neighbourhood
of a brackish-water RO train, but nothing came from plant data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_table import (
    CAUSES,
    DP_DOWN_PCT,
    DP_UP_PCT,
    FLOW_DOWN_PCT,
    FLOW_UP_PCT,
    SEVERE_FLOW_LOSS_PCT,
    SP_DOWN_PCT,
    SP_SHARP_UP_PCT,
    SP_UP_PCT,
    classify,
    keys_for_cause,
)
from .schema import validate

DEFAULT_MARGIN_PP = 2.0
BOUNDARY_MARGIN_PP = 0.6

# Feed temperature at t0. Training stays in a narrow band so that the
# `shift_temp` holdout can sit well outside it.
TRAIN_TEMP_RANGE = (20.0, 25.0)
SHIFT_TEMP_RANGES = ((10.0, 15.0), (28.0, 33.0))

P_SEVERE = 0.22  # of the flow=down cases; ~14% of the dataset overall
P_HARD_TIER = 0.4

MAX_RESAMPLE = 200


def tcf(temp_c: float) -> float:
    """Temperature correction factor to 25 C, the form quoted in the prompt."""
    return 1.03 ** (25.0 - temp_c)


# --- sampling bands ---------------------------------------------------------
# Derived from the thresholds rather than written out, so that moving a
# threshold in decision_table.py cannot leave a stale band behind here.


def _flow_band(flag: str, severe: bool, margin: float) -> tuple[float, float]:
    if flag == "flat":
        return (FLOW_DOWN_PCT + margin, FLOW_UP_PCT - margin)
    if flag == "up":
        return (FLOW_UP_PCT + margin, 35.0)
    if severe:
        return (-50.0, SEVERE_FLOW_LOSS_PCT - margin)
    return (SEVERE_FLOW_LOSS_PCT + margin, FLOW_DOWN_PCT - margin)


def _sp_band(flag: str, margin: float) -> tuple[float, float]:
    return {
        "down": (-45.0, SP_DOWN_PCT - margin),
        "flat": (SP_DOWN_PCT + margin, SP_UP_PCT - margin),
        "up": (SP_UP_PCT + margin, SP_SHARP_UP_PCT - margin),
        "sharp_up": (SP_SHARP_UP_PCT + margin, 160.0),
    }[flag]


def _dp_band(flag: str, margin: float) -> tuple[float, float]:
    return {
        "down": (-45.0, DP_DOWN_PCT - margin),
        "flat": (DP_DOWN_PCT + margin, DP_UP_PCT - margin),
        "up": (DP_UP_PCT + margin, 60.0),
    }[flag]


def truth_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Compute the answer key from a finished record. The only grader there is.

    Returns a dict in exactly the answer schema, with the three percent changes
    rounded to one decimal -- the same precision the prompt asks the model for.
    """
    t0, t1 = record["t0"], record["t1"]

    nf0 = t0["permeate_flow_m3_h"] * tcf(t0["feed_temp_C"])
    nf1 = t1["permeate_flow_m3_h"] * tcf(t1["feed_temp_C"])
    flow_pct = (nf1 - nf0) / nf0 * 100.0

    sp0 = t0["permeate_conductivity_uS_cm"] / t0["feed_conductivity_uS_cm"] * 100.0
    sp1 = t1["permeate_conductivity_uS_cm"] / t1["feed_conductivity_uS_cm"] * 100.0
    sp_pct = (sp1 - sp0) / sp0 * 100.0

    dp0 = t0["dp_lead_bar"] + t0["dp_tail_bar"]
    dp1 = t1["dp_lead_bar"] + t1["dp_tail_bar"]
    dp_pct = (dp1 - dp0) / dp0 * 100.0

    stage = record["anomaly_stage"]
    diagnosis = classify(flow_pct, sp_pct, dp_pct, stage)

    return {
        "normalized_flow_change_pct": round(flow_pct, 1),
        "salt_passage_change_pct": round(sp_pct, 1),
        "dp_change_pct": round(dp_pct, 1),
        "flags": dict(diagnosis.flags),
        "stage": stage,
        "root_cause": diagnosis.root_cause,
        "action": diagnosis.action,
    }


def _sample_t0(rng: random.Random, temp_range: tuple[float, float]) -> dict[str, Any]:
    feed_cond = round(rng.uniform(2000.0, 6000.0), -1)
    sp0_pct = rng.uniform(1.0, 3.5)
    return {
        "feed_temp_C": round(rng.uniform(*temp_range), 1),
        "feed_pressure_bar": round(rng.uniform(12.0, 20.0), 1),
        "feed_conductivity_uS_cm": feed_cond,
        "permeate_conductivity_uS_cm": round(feed_cond * sp0_pct / 100.0, 1),
        "permeate_flow_m3_h": round(rng.uniform(20.0, 60.0), 1),
        "dp_lead_bar": round(rng.uniform(0.35, 1.10), 2),
        "dp_tail_bar": round(rng.uniform(0.30, 1.00), 2),
    }


def _attempt(
    rng: random.Random,
    cause: str,
    tier: str,
    margin: float,
    temp_range: tuple[float, float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """One backward-construction attempt. Returns (record, answer, meta) or None."""
    flow_flag, sp_flag, dp_flag, stage = rng.choice(keys_for_cause(cause))
    severe = flow_flag == "down" and rng.random() < P_SEVERE

    flow_pct = rng.uniform(*_flow_band(flow_flag, severe, margin))
    sp_pct = rng.uniform(*_sp_band(sp_flag, margin))
    dp_pct = rng.uniform(*_dp_band(dp_flag, margin))

    t0 = _sample_t0(rng, temp_range)

    # Temperature is the only thing separating the tiers. When it is unchanged
    # TCF cancels and step 1 is a plain percent change.
    if tier == "easy":
        temp1 = t0["feed_temp_C"]
    else:
        delta = rng.choice((-1.0, 1.0)) * rng.uniform(3.0, 8.0)
        temp1 = round(min(max(t0["feed_temp_C"] + delta, 8.0), 36.0), 1)
        if temp1 == t0["feed_temp_C"]:
            return None

    flow1 = (
        t0["permeate_flow_m3_h"]
        * tcf(t0["feed_temp_C"])
        / tcf(temp1)
        * (1.0 + flow_pct / 100.0)
    )

    # Feed conductivity drifts a little, so salt passage is a real division
    # rather than a disguised permeate-conductivity comparison.
    feed_cond1 = round(t0["feed_conductivity_uS_cm"] * rng.uniform(0.95, 1.05), -1)
    sp0_pct = t0["permeate_conductivity_uS_cm"] / t0["feed_conductivity_uS_cm"] * 100.0
    perm_cond1 = feed_cond1 * (sp0_pct * (1.0 + sp_pct / 100.0)) / 100.0

    # The whole dp change lands on the stage the probe flagged; the other stage
    # holds still. The target percentage is on the train total either way.
    dp0_total = t0["dp_lead_bar"] + t0["dp_tail_bar"]
    delta_dp = dp0_total * dp_pct / 100.0
    dp_lead1 = t0["dp_lead_bar"] + (delta_dp if stage == "lead" else 0.0)
    dp_tail1 = t0["dp_tail_bar"] + (delta_dp if stage == "tail" else 0.0)

    if min(flow1, perm_cond1, dp_lead1, dp_tail1) <= 0.06:
        return None

    t1 = {
        "feed_temp_C": temp1,
        "feed_pressure_bar": t0["feed_pressure_bar"],  # held fixed; a distractor
        "feed_conductivity_uS_cm": feed_cond1,
        "permeate_conductivity_uS_cm": round(perm_cond1, 1),
        "permeate_flow_m3_h": round(flow1, 1),
        "dp_lead_bar": round(dp_lead1, 2),
        "dp_tail_bar": round(dp_tail1, 2),
    }

    record = {
        "t0": t0,
        "t1": t1,
        "recovery_pct": round(rng.uniform(68.0, 78.0), 1),
        "anomaly_stage": stage,
    }

    # Re-derive the label from the rounded record. If rounding moved anything
    # across a threshold, this case is discarded rather than mislabelled.
    try:
        answer = truth_from_record(record)
    except KeyError:
        return None

    intended = {"flow": flow_flag, "salt_passage": sp_flag, "dp": dp_flag}
    if answer["flags"] != intended or answer["root_cause"] != cause:
        return None
    if severe != (answer["normalized_flow_change_pct"] <= SEVERE_FLOW_LOSS_PCT):
        return None

    meta = {
        "intended_cause": cause,
        "intended_key": [flow_flag, sp_flag, dp_flag, stage],
        "severe": severe,
        "margin_pp": margin,
    }
    return record, answer, meta


def generate_case(
    rng: random.Random,
    cause: str,
    tier: str,
    margin: float = DEFAULT_MARGIN_PP,
    temp_range: tuple[float, float] = TRAIN_TEMP_RANGE,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Construct one case for `cause`, resampling until the label holds up."""
    for _ in range(MAX_RESAMPLE):
        attempt = _attempt(rng, cause, tier, margin, temp_range)
        if attempt is not None:
            return attempt
    raise RuntimeError(f"could not construct a {tier}/{cause} case in {MAX_RESAMPLE} tries")


def build_split(
    n: int,
    seed: int,
    split: str,
    slice_name: str = "main",
    margin: float = DEFAULT_MARGIN_PP,
    temp_ranges: Sequence[tuple[float, float]] = (TRAIN_TEMP_RANGE,),
    hard_only: bool = False,
) -> list[dict[str, Any]]:
    """Build `n` cases with causes balanced round-robin, then shuffled."""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []

    for i in range(n):
        cause = CAUSES[i % len(CAUSES)]
        tier = "hard" if hard_only or rng.random() < P_HARD_TIER else "easy"
        record, answer, meta = generate_case(
            rng, cause, tier, margin, rng.choice(tuple(temp_ranges))
        )
        cases.append(
            {
                "id": f"{split}-{slice_name}-{i:04d}",
                "split": split,
                "slice": slice_name,
                "tier": tier,
                "record": record,
                "answer": answer,
                "meta": meta,
            }
        )

    rng.shuffle(cases)
    return cases


def _write_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for case in cases:
            fh.write(json.dumps(case, sort_keys=True) + "\n")


def _summarise(name: str, cases: list[dict[str, Any]]) -> str:
    causes = Counter(case["answer"]["root_cause"] for case in cases)
    tiers = Counter(case["tier"] for case in cases)
    severe = sum(case["meta"]["severe"] for case in cases)
    spread = ", ".join(f"{cause}={causes[cause]}" for cause in CAUSES)
    return (
        f"{name:<16} n={len(cases):<5} easy={tiers['easy']:<4} hard={tiers['hard']:<4} "
        f"severe={severe:<4}\n{'':<16} {spread}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--dev", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--shift", type=int, default=25, help="cases per holdout shift slice")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": build_split(args.train, args.seed + 1, "train"),
        "dev": build_split(args.dev, args.seed + 2, "dev"),
        "test": build_split(args.test, args.seed + 3, "test"),
        "holdout_shift": (
            build_split(
                args.shift,
                args.seed + 4,
                "holdout_shift",
                slice_name="shift_temp",
                temp_ranges=SHIFT_TEMP_RANGES,
                hard_only=True,
            )
            + build_split(
                args.shift,
                args.seed + 5,
                "holdout_shift",
                slice_name="shift_boundary",
                margin=BOUNDARY_MARGIN_PP,
            )
        ),
    }

    for name, cases in splits.items():
        bad = [case["id"] for case in cases if not validate(case["answer"]).ok]
        if bad:
            raise RuntimeError(f"{name}: {len(bad)} answers fail their own schema, e.g. {bad[0]}")
        _write_jsonl(args.out / f"{name}.jsonl", cases)
        print(_summarise(name, cases))

    digests = []
    for name in sorted(splits):
        path = args.out / f"{name}.jsonl"
        digests.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.out / "SHA256SUMS").write_text("\n".join(digests) + "\n")

    print("\nwrote SHA256SUMS:")
    print("\n".join(digests))


if __name__ == "__main__":
    main()
