"""Generate chain-of-thought SFT data: explicit reasoning trace + JSON answer.

The trace is rendered from the repo's own ground-truth computation
(task.generate.truth_from_record / task.decision_table.classify), so it is
faithful by construction. The pedagogical content is the contrastive
table-lookup note: for each cause the trace names the near-miss row it is
*not*, which is exactly where GRPO collapsed (scaling/biofouling,
mechanical_leak/oxidation_damage, compaction/biofouling).

Output: data/train_cot.jsonl -- same schema as train.jsonl plus
"cot_completion" (trace + canonical JSON).
"""
from __future__ import annotations

import json
from pathlib import Path

from task.decision_table import (
    DP_DOWN_PCT, DP_UP_PCT, FLOW_DOWN_PCT, FLOW_UP_PCT,
    SEVERE_FLOW_LOSS_PCT, SP_DOWN_PCT, SP_SHARP_UP_PCT, SP_UP_PCT,
    classify, flags_for,
)
from task.generate import tcf
from task.schema import canonical

ROOT = Path(__file__).resolve().parent

# The disambiguation worth stating, per cause. This is the whole point:
# the model reads flags and stage fine but fails the table lookup.
WHY_NOT = {
    "scaling": (
        "Biofouling shares the (down, up/sharp_up, up) trends; "
        "stage tail separates them -- scale precipitates where the brine "
        "is most concentrated."
    ),
    "biofouling": (
        "Scaling shares these trends; stage lead separates them -- biofilm "
        "grows where the nutrients enter."
    ),
    "colloidal_fouling": (
        "Biofouling also fouls the lead end, but it drives salt passage up; "
        "here salt passage is flat, so colloidal plugging."
    ),
    "organic_fouling": (
        "Flow lost with no channel blockage (dp flat) and rejection "
        "improving (sp down) -- adsorbed organics, not a blocked channel."
    ),
    "compaction": (
        "Flow lost, channel clear (dp flat), rejection flat or slightly "
        "worse -- compaction, not biofouling, which needs salt passage up."
    ),
    "oxidation_damage": (
        "Flow rises as rejection dies (sp sharp_up) -- the barrier layer is "
        "damaged, not fouled."
    ),
    "mechanical_leak": (
        "Flow barely moves (flat) while salt passage jumps (sharp_up) -- feed "
        "bypassing the membrane, not oxidation damage, which raises flow."
    ),
}


def _fmt_pct(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f}%"


def render_trace(record: dict) -> tuple[str, dict]:
    t0, t1 = record["t0"], record["t1"]
    temp0, temp1 = t0["feed_temp_C"], t1["feed_temp_C"]
    k0, k1 = tcf(temp0), tcf(temp1)

    f0, f1 = t0["permeate_flow_m3_h"], t1["permeate_flow_m3_h"]
    nf0, nf1 = f0 * k0, f1 * k1
    flow_pct = (nf1 - nf0) / nf0 * 100.0

    c0, c1 = t0["permeate_conductivity_uS_cm"], t1["permeate_conductivity_uS_cm"]
    fc0, fc1 = t0["feed_conductivity_uS_cm"], t1["feed_conductivity_uS_cm"]
    sp0, sp1 = c0 / fc0 * 100.0, c1 / fc1 * 100.0
    sp_pct = (sp1 - sp0) / sp0 * 100.0

    dp0 = t0["dp_lead_bar"] + t0["dp_tail_bar"]
    dp1 = t1["dp_lead_bar"] + t1["dp_tail_bar"]
    dp_pct = (dp1 - dp0) / dp0 * 100.0

    flags = flags_for(flow_pct, sp_pct, dp_pct)
    stage = record["anomaly_stage"]
    diagnosis = classify(flow_pct, sp_pct, dp_pct, stage)
    severe = flow_pct <= SEVERE_FLOW_LOSS_PCT

    lines = [
        f"Flow: TCF({temp0}C)={k0:.3f}, TCF({temp1}C)={k1:.3f}; "
        f"normalized {nf0:.1f} -> {nf1:.1f} m3/h = {_fmt_pct(flow_pct)} "
        f"-> {flags['flow']} (down <= {FLOW_DOWN_PCT}%, up >= {FLOW_UP_PCT}%).",
        f"Salt passage: {sp0:.2f}% -> {sp1:.2f}% = {_fmt_pct(sp_pct)} "
        f"-> {flags['salt_passage']} (down <= {SP_DOWN_PCT}%, up >= {SP_UP_PCT}%, "
        f"sharp_up >= {SP_SHARP_UP_PCT}%).",
        f"DP: {dp0:.2f} -> {dp1:.2f} bar = {_fmt_pct(dp_pct)} "
        f"-> {flags['dp']} (down <= {DP_DOWN_PCT}%, up >= {DP_UP_PCT}%).",
        f"Stage: {stage} (given).",
        f"Lookup: ({flags['flow']}, {flags['salt_passage']}, {flags['dp']}, {stage}) "
        f"-> {diagnosis.root_cause}. {WHY_NOT[diagnosis.root_cause]}",
        f"Action: {diagnosis.root_cause} -> {diagnosis.action}"
        + (" (severe flow loss, isolate)." if severe else "."),
    ]
    answer = {
        "normalized_flow_change_pct": round(flow_pct, 1),
        "salt_passage_change_pct": round(sp_pct, 1),
        "dp_change_pct": round(dp_pct, 1),
        "flags": dict(diagnosis.flags),
        "stage": stage,
        "root_cause": diagnosis.root_cause,
        "action": diagnosis.action,
    }
    return "\n".join(lines), answer


def main() -> None:
    src = ROOT / "data" / "train.jsonl"
    dst = ROOT / "data" / "train_cot.jsonl"
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            case = json.loads(line)
            trace, answer = render_trace(case["record"])
            # Faithfulness gate: the rendered answer must match the stored one.
            assert answer == case["answer"], f"mismatch on {case['id']}"
            case["cot_completion"] = trace + "\n" + canonical(answer)
            fout.write(json.dumps(case) + "\n")
            n += 1
    print(f"wrote {dst} ({n} cases, all answers verified)")


if __name__ == "__main__":
    main()
