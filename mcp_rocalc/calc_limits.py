"""check_limits: every candidate's values against a question's limits, with each comparison written out.

A limit is "op number unit"; the number is converted to the field's own unit (the unit in the field's name, as
the RO simulators return it) with convert_units' factors. Copied verbatim from the smoke_test35 notebooks
(06_direction_and_compare.py introduced it), so the deployed tool and the trained one are the same code.
"""
from __future__ import annotations

import json
import re
from typing import Any

from calc_units import convert_units, parse_number

METRIC_TOOL_UNIT = {"permeate.flow_kg_s": "kg/s", "permeate.nacl_ppm": "mg/L", "flux.water_LMH": "LMH",
                    "costing.LCOW_usd_m3": "$/m³", "performance.product_flow_m3_s": "m³/s",
                    "costing.specific_energy_kWh_m3": "kWh/m³", "performance.water_recovery_pct": "%",
                    "performance.salt_rejection_pct": "%", "performance.system_recovery_pct": "%"}


COMPARE_OPS = {"<=": lambda v, l: v <= l, ">=": lambda v, l: v >= l, "<": lambda v, l: v < l, ">": lambda v, l: v > l}


_LIMIT_SPEC = re.compile(r"\s*(<=|>=|<|>|≤|≥|=<|=>)\s*(-?[\d.,]+(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def limit_in_field_unit(field: str, number: float, unit: str) -> tuple[float | None, str | None]:
    """A limit stated as ``number unit`` in the field's own unit (converted with convert_units' factors
    when the unit differs): (value, None) or (None, error)."""
    if not unit:
        return number, None
    own = METRIC_TOOL_UNIT.get(field)
    if own is None:
        return None, f"{field!r} is not a known field: give its limit without a unit, in the value's own unit"
    got = json.loads(convert_units(number, unit, own))
    if "error" in got:
        return None, got["error"]
    return got["value"], None


def check_limits(limits: Any, values: Any) -> str:
    """The tool, one call per question. ``limits`` = {field: "op number unit"}, the number as the user
    stated it (the tool converts it to the field's own unit); ``values`` = {candidate: {field: number}}.
    Returns a JSON list with one line per candidate -- "candidate: field value meets|breaks op limit;
    ... -> pass|fail", the limit in the field's own unit, a field given without a limit written out as
    "field value". A limit the tool cannot read -- an unknown field, a unit that does not convert to the
    field's own, no "op number" -- is written into every line as "field value (...)" and left out of the
    verdict; a candidate without a number for a limited field gets "field ? (no value)". The verdict is
    fail when a read limit breaks, pass when every limit is read and met, unchecked otherwise. A slip
    costs that limit, not the call: {"error": ...} only when limits or values are not objects of that
    shape."""
    if isinstance(limits, str):
        try:
            limits = json.loads(limits)
        except ValueError:
            pass
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except ValueError:
            pass
    if not isinstance(limits, dict) or not limits:
        return json.dumps({"error": f"limits must be an object of field: \"op number unit\", got {limits!r}"})
    if not isinstance(values, dict) or not values:
        return json.dumps({"error": f"values must be an object of candidate: {{field: number}}, got {values!r}"})
    parsed: dict[str, tuple[str | None, Any]] = {}  # field -> (op, limit) when read, (None, note) when not
    for field, spec in limits.items():
        field = str(field)
        m = _LIMIT_SPEC.fullmatch(str(spec))
        num = parse_number(m.group(2)) if m else None
        if m is None or num is None:
            parsed[field] = (None, f"limit {str(spec)!r} not read")
            continue
        o = m.group(1).replace("≤", "<=").replace("≥", ">=").replace("=<", "<=").replace("=>", ">=")
        own = METRIC_TOOL_UNIT.get(field)
        if own is None:
            parsed[field] = (None, "unknown field")
            continue
        lim, err = limit_in_field_unit(field, num, m.group(3))
        if err:
            parsed[field] = (None, f"limit {str(spec)!r} not in {own}")
            continue
        parsed[field] = (o, lim)
    lines = []
    for label, vals in values.items():
        if not isinstance(vals, dict):
            return json.dumps({"error": f"the values of {label!r} must be an object of field: number, got {vals!r}"})
        parts, broken, unchecked = [], False, False
        for field, (o, lim) in parsed.items():
            x = parse_number(vals[field]) if field in vals else None
            if o is None:
                parts.append(f"{field} {_fmt(x) if x is not None else '?'} ({lim})")
                unchecked = True
            elif x is None:
                parts.append(f"{field} ? (no value)")
                unchecked = True
            else:
                meets = COMPARE_OPS[o](x, lim)
                broken |= not meets
                parts.append(f"{field} {_fmt(x)} {'meets' if meets else 'breaks'} {o} {lim:g}")
        for field, v in vals.items():  # a field without a limit: written out
            if str(field) not in parsed:
                x = parse_number(v)
                parts.append(f"{field} {_fmt(x) if x is not None else '? (no value)'}")
        lines.append(f"{label}: " + "; ".join(parts) + f" -> {'fail' if broken else 'unchecked' if unchecked else 'pass'}")
    return json.dumps(lines, ensure_ascii=False)
