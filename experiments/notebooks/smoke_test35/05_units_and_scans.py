#!/usr/bin/env python
"""Notebook 05 -- 04's task with a unit-conversion tool and a scan-aggregation line.

04's converged policy lost most of its remaining decisions to two things a 0.8B model cannot do in
its head: unit conversion (limits stated in gfd, t/h or m3/h were copied unconverted into the
comparison lines on 91 of 102 occasions; arguments in F, ft2 or t/h landed 1-2.5% off, outside the
0.5% tolerance) and picking the right end of a scan (asked for the highest passing setting, it
answered the first). This notebook adds one tool, ``convert_units``, and one line to the answer's
reasoning -- the passing candidates and the one the question's objective picks -- and reruns the
recipe: seed, then PPO. Everything else -- the questions' generator, the reward, the harness --
is 04's, copied (no code is shared between notebooks) with these changes:

  * ``convert_units(value, from_unit, to_unit)``: exact factors, the same the generator uses. It is
    served by ToolHost here; the deployed MCP server does not have it yet.
  * The generator records the unit every input was stated in, so a reference episode knows what to
    convert. 04's factor for A in LMH/bar was 1000x too large (3.6e14; 1 m/s = 3.6e6 LMH and 1 bar
    = 1e5 Pa give 3.6e11); corrected, and the questions regenerated. Each case draws its own random
    stream, so only the cases that state A in LMH/bar change; ``check`` counts the identical ones.
  * Reference episodes convert first (one turn), simulate every candidate (one turn), then answer.
    Convert calls earn nothing and cost nothing: they are outside the calls credit, the efficiency
    count and the simulation budget.
  * The policy's token budget per episode is 2,048 (04: 1,536; six long scans were cut off).

Nothing here trains on the MembraneClaw benchmark (D1-D6): it is evaluation-only, and the only
code that opens it is the decontamination check. Every gold answer comes from running the real
simulator on the generated scenario, through the same argument handling the deployed MCP server
applies (verified bit-identical to the server on temur: same code hashes, same package versions,
worst relative difference 0.0).

Subcommands (all but ``check`` need ``.venv-watertap``; scoring and ``check`` are pure Python and
also run in the training venv):

  dump-tools  freeze the served MCP declarations of the two tools, plus convert_units, into 05_units_and_scans_tools.json
  generate    sample scenarios, solve every point, fit thresholds, render the prompts, write splits
  elasticity  worst output change per +-0.5% input error, for both tools (sets the gold margins)
  worker      JSON-lines tool server for rollouts (deployed argument handling, no rate limit)
  check       decontamination against the benchmark, reward sanity (oracle = 1), null baselines
  xcheck      re-solve a sample of gold points through the deployed MCP on temur
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PREFIX = "05_units_and_scans"
DATA = HERE / f"{PREFIX}_data"
DATA_04 = HERE / "04_natural_watertap_data"  # 04's questions, for the identity count in ``check``
TOOLS_JSON = HERE / f"{PREFIX}_tools.json"
REPO = HERE.parents[2]
MCP_DIR = REPO / "mcp_watertap"
AGENT_CORE = REPO / "agent" / "core.py"
BENCH_DIR = REPO / "experiments" / "auto-evaluate" / "benchmarks" / "Datasets Harness"

RO, SWRO, CONVERT = "simulate_ro", "simulate_swro_system", "convert_units"
TOOLS = (RO, SWRO)              # the simulators, declared exactly as the deployed server serves them
ALL_TOOLS = (RO, SWRO, CONVERT)  # plus this notebook's conversion tool

# ---------------------------------------------------------------------------------------------
# The fixed parts of every episode: system prompt, tool defaults, the metrics an answer may cite.
# ---------------------------------------------------------------------------------------------

#: agent/core.py's SYSTEM_MESSAGE, verbatim (``check`` asserts it still matches).
DEPLOYED_SYSTEM_MESSAGE = (
    "You are a capable assistant running on a local Qwen3.5 deployment. "
    "Use the provided tools when they give you a more accurate answer than reasoning alone; "
    "otherwise answer directly. When documents are attached, ground your answer in them and "
    "say so if the answer is not present."
)
#: The one addition to the deployed prompt: an answer contract that code can grade.
ANSWER_CONTRACT = (
    "Finish your reply with one JSON object and nothing after it:\n"
    '{"decision": ..., "violations": [...], "values": {...}}\n'
    '- "decision": the direct answer to the question. "pass" or "fail" when asked whether '
    "something meets its limits; otherwise the option or setting you recommend, written the way "
    "the user named it; null if none of the options works.\n"
    '- "violations": when asked whether something meets its limits, the tool-result fields whose '
    "limits are broken; otherwise [].\n"
    '- "values": the numbers the user asked for, keyed by the tool-result field they come from '
    "as section.field, in that field's units."
)
SYSTEM_PROMPT = DEPLOYED_SYSTEM_MESSAGE + "\n\n" + ANSWER_CONTRACT

#: mcp_watertap ro_model.DEFAULTS / swro_model.DEFAULTS (``generate`` asserts they still match).
DEFAULTS: dict[str, dict[str, Any]] = {
    RO: {
        "feed_flow_mass_kg_s": 1.0, "feed_nacl_mass_frac": 0.035, "feed_pressure_bar": 50.0,
        "feed_temperature_c": 25.0, "membrane_area_m2": 50.0, "A_comp": 4.2e-12, "B_comp": 3.5e-8,
        "permeate_pressure_bar": 1.01325, "pressure_drop_bar": 3.0, "channel_height_m": 0.002,
        "spacer_porosity": 0.75, "module_length_m": 20.0, "cp_modulus": 1.1,
        "mass_transfer_coeff": 2e-5, "concentration_polarization": "calculated",
        "mass_transfer_coefficient": "calculated",
    },
    SWRO: {
        "erd_type": "pressure_exchanger", "feed_flow_m3_s": 0.3092, "feed_tds_g_L": 35.0,
        "feed_tss_g_L": 0.03, "feed_temperature_c": 24.85, "ro_area_m2": None, "A_comp": 4.2e-12,
        "B_comp": 3.5e-8, "p1_pressure_bar": 70.0, "p1_efficiency": 0.80, "pxr_efficiency": 0.95,
        "p2_efficiency": 0.80, "erd_efficiency": 0.95,
    },
}

#: Result fields an answer may cite, as section.field paths into the tool's JSON.
METRICS = {
    RO: ("permeate.flow_kg_s", "permeate.nacl_ppm", "performance.water_recovery_pct",
         "performance.salt_rejection_pct", "flux.water_LMH"),
    SWRO: ("costing.specific_energy_kWh_m3", "costing.LCOW_usd_m3",
           "performance.product_flow_m3_s", "performance.system_recovery_pct"),
}
#: Smallest |value - threshold| / value any gold point may have. A 0.5% error in one input moves
#: RO outputs by at most 1.43% (measured over 12 scenarios x 7 inputs), so 2% keeps every gold
#: pass/fail stable under the argument tolerance; rejection moves 0.02%, hence its smaller margin.
MARGIN = defaultdict(lambda: 0.02, {"performance.salt_rejection_pct": 0.001})
#: How far past the nearest value an open-ended threshold may sit (keeps limits plausible).
SPREAD = {
    "permeate.flow_kg_s": 0.25, "permeate.nacl_ppm": 0.30, "performance.water_recovery_pct": 0.15,
    "performance.salt_rejection_pct": 0.006, "flux.water_LMH": 0.25,
    "costing.specific_energy_kWh_m3": 0.25, "costing.LCOW_usd_m3": 0.25,
    "performance.product_flow_m3_s": 0.25, "performance.system_recovery_pct": 0.15,
}
PCT_METRICS = {"performance.water_recovery_pct", "performance.salt_rejection_pct",
               "performance.system_recovery_pct"}
METRIC_NAMES = {
    "permeate.flow_kg_s": ["permeate flow", "product water flow"],
    "permeate.nacl_ppm": ["permeate NaCl", "permeate salinity"],
    "performance.water_recovery_pct": ["recovery", "water recovery"],
    "performance.salt_rejection_pct": ["salt rejection"],
    "flux.water_LMH": ["water flux", "flux"],
    "costing.specific_energy_kWh_m3": ["specific energy", "energy use per cubic metre"],
    "costing.LCOW_usd_m3": ["levelized cost of water", "water cost per cubic metre"],
    "performance.product_flow_m3_s": ["product flow", "plant output"],
    "performance.system_recovery_pct": ["overall plant recovery", "system recovery"],
}

# ---------------------------------------------------------------------------------------------
# Units. Every quantity is sampled in the user's unit on a round grid and converted to the tool's
# unit exactly, so the gold is unique. Level 0 = the tool's own unit, 1 = a decimal shift or a
# relabel, 2 = a multiplicative conversion. Only conversions with exact factors are used.
# ---------------------------------------------------------------------------------------------

def _u(name: str, level: int, factor: float, grid: float, fmt: str, offset: float = 0.0) -> dict:
    return {"name": name, "level": level, "factor": factor, "offset": offset, "grid": grid, "fmt": fmt}


PSI_PER_BAR = 1e5 / 6894.757293168
FT2_PER_M2 = 1 / 0.09290304
MGD_PER_M3S = 86400 / 3785.411784
GFD_PER_LMH = 0.09290304 * 24 / 3.785411784
INPUT_UNITS = {
    "ro_flow": [_u("kg/s", 0, 1, 0.01, "{:.2f} kg/s"), _u("t/h", 2, 3.6, 0.1, "{:.1f} t/h"),
                _u("kg/h", 2, 3600, 10, "{:,.0f} kg/h")],
    "salinity": [_u("mass fraction", 0, 1, 0.0005, "a NaCl mass fraction of {:.4f}"),
                 _u("g/kg", 1, 1000, 0.1, "{:.1f} g/kg"), _u("permille", 1, 1000, 0.1, "{:.1f} ‰"),
                 _u("pct", 1, 100, 0.01, "{:.2f} % salt by mass"),
                 _u("mg/kg", 1, 1e6, 100, "{:,.0f} mg/kg")],
    "temp": [_u("C", 0, 1, 1, "{:.0f} °C"), _u("F", 2, 1.8, 1, "{:.0f} °F", offset=32)],
    "area": [_u("m2", 0, 1, 5, "{:,.0f} m²"), _u("ft2", 2, FT2_PER_M2, 50, "{:,.0f} ft²")],
    # 1 m/s = 3.6e6 LMH and 1 bar = 1e5 Pa: A in LMH/bar is A in m/s/Pa times 3.6e11 (04 had 3.6e14).
    "A": [_u("SI", 0, 1e12, 0.1, "{:.1f}e-12 m/s/Pa"), _u("LMH/bar", 2, 3.6e11, 0.01, "{:.2f} LMH/bar")],
    "B": [_u("SI", 0, 1e8, 0.1, "{:.1f}e-8 m/s"), _u("LMH", 2, 3.6e6, 0.001, "{:.3f} LMH")],
    "pressure": [_u("bar", 0, 1, 0.5, "{:g} bar"), _u("kPa", 1, 100, 50, "{:,.0f} kPa"),
                 _u("MPa", 1, 0.1, 0.05, "{:.2f} MPa"), _u("psi", 2, PSI_PER_BAR, 5, "{:,.0f} psi")],
    "swro_flow": [_u("m3/s", 0, 1, 0.001, "{:.3f} m³/s"), _u("L/s", 1, 1000, 1, "{:,.0f} L/s"),
                  _u("m3/h", 2, 3600, 10, "{:,.0f} m³/h"), _u("m3/d", 2, 86400, 100, "{:,.0f} m³/d"),
                  _u("MGD", 2, MGD_PER_M3S, 0.01, "{:.2f} MGD")],
    "tds": [_u("g/L", 0, 1, 0.1, "{:.1f} g/L"), _u("mg/L", 1, 1000, 100, "{:,.0f} mg/L"),
            _u("kg/m3", 1, 1, 0.1, "{:.1f} kg/m³")],
    "eff": [_u("fraction", 0, 1, 0.01, "{:.2f}"), _u("pct", 1, 100, 1, "{:.0f} %")],
}
#: Units a threshold on each metric may be stated in ("{}" is filled with a formatted number).
METRIC_UNITS = {
    "permeate.flow_kg_s": [_u("kg/s", 0, 1, 0, "{} kg/s"), _u("m3/h", 2, 3.6, 0, "{} m³/h"),
                           _u("m3/d", 2, 86.4, 0, "{} m³/d"), _u("t/h", 2, 3.6, 0, "{} t/h")],
    "permeate.nacl_ppm": [_u("mg/L", 0, 1, 0, "{} mg/L"), _u("ppm", 0, 1, 0, "{} ppm")],
    "performance.water_recovery_pct": [_u("%", 0, 1, 0, "{} %")],
    "performance.salt_rejection_pct": [_u("%", 0, 1, 0, "{} %")],
    "flux.water_LMH": [_u("LMH", 0, 1, 0, "{} LMH"), _u("L/m2/h", 1, 1, 0, "{} L/m²/h"),
                       _u("gfd", 2, GFD_PER_LMH, 0, "{} gfd")],
    "costing.specific_energy_kWh_m3": [_u("kWh/m3", 0, 1, 0, "{} kWh/m³")],
    "costing.LCOW_usd_m3": [_u("$/m3", 0, 1, 0, "${} per m³"),
                            _u("$/kgal", 2, 3.785411784, 0, "${} per 1,000 gallons")],
    "performance.product_flow_m3_s": [_u("m3/s", 0, 1, 0, "{} m³/s"), _u("L/s", 1, 1000, 0, "{} L/s"),
                                      _u("m3/d", 2, 86400, 0, "{} m³/d"), _u("m3/h", 2, 3600, 0, "{} m³/h"),
                                      _u("MGD", 2, MGD_PER_M3S, 0, "{} MGD")],
    "performance.system_recovery_pct": [_u("%", 0, 1, 0, "{} %")],
}
#: Scan steps in user units, per input unit.
SCAN_STEPS = {
    ("pressure", "bar"): [1.5, 2, 2.5, 3], ("pressure", "kPa"): [150, 200, 250, 300],
    ("pressure", "MPa"): [0.15, 0.2, 0.25, 0.3], ("pressure", "psi"): [25, 30, 40, 50],
    ("area", "m2"): [25, 30, 40, 50], ("area", "ft2"): [300, 400, 500],
}


#: How a unit is written in a convert_units call: the way the prompts write it.
UNIT_SYMBOL = {
    "kg/s": "kg/s", "t/h": "t/h", "kg/h": "kg/h", "mass fraction": "mass fraction", "g/kg": "g/kg",
    "permille": "‰", "pct": "%", "mg/kg": "mg/kg", "C": "C", "F": "°F", "m2": "m²", "ft2": "ft²",
    "SI": "SI", "LMH/bar": "LMH/bar", "LMH": "LMH", "bar": "bar", "kPa": "kPa", "MPa": "MPa", "psi": "psi",
    "m3/s": "m³/s", "L/s": "L/s", "m3/h": "m³/h", "m3/d": "m³/d", "MGD": "MGD", "g/L": "g/L", "mg/L": "mg/L",
    "kg/m3": "kg/m³", "fraction": "fraction", "%": "%", "ppm": "ppm", "L/m2/h": "L/m²/h", "gfd": "gfd",
    "kWh/m3": "kWh/m³", "$/m3": "$/m³", "$/kgal": "$/kgal",
}
#: The simulators' own unit of each input quantity and of each result metric (the convert target).
TOOL_UNIT = {"ro_flow": "kg/s", "salinity": "mass fraction", "temp": "C", "area": "m²", "A": "m/s/Pa",
             "B": "m/s", "pressure": "bar", "swro_flow": "m³/s", "tds": "g/L", "eff": "fraction"}
METRIC_TOOL_UNIT = {"permeate.flow_kg_s": "kg/s", "permeate.nacl_ppm": "mg/L", "flux.water_LMH": "LMH",
                    "costing.LCOW_usd_m3": "$/m³", "performance.product_flow_m3_s": "m³/s",
                    "costing.specific_energy_kWh_m3": "kWh/m³", "performance.water_recovery_pct": "%",
                    "performance.salt_rejection_pct": "%", "performance.system_recovery_pct": "%"}


def unit_spec(q: str, name: str) -> dict:
    return next(u for u in INPUT_UNITS[q] if u["name"] == name)


def metric_unit_spec(path: str, name: str) -> dict:
    return next(u for u in METRIC_UNITS[path] if u["name"] == name)


def needs_conversion(u: dict) -> bool:
    """A stated unit whose number differs from the tool's (level 0 is the tool's own unit; a relabel
    such as L/m²/h for LMH changes nothing)."""
    return u["level"] > 0 and (u["factor"] != 1 or u["offset"] != 0)


def to_user(u: dict, tool_value: float) -> float:
    return tool_value * u["factor"] + u["offset"]


def to_tool(u: dict, user_value: float) -> float:
    return (user_value - u["offset"]) / u["factor"]


def round_grid(x: float, grid: float) -> float:
    return round(round(x / grid) * grid, 10)


def fmt_step(x: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return f"{x:,.{decimals}f}"


# ---------------------------------------------------------------------------------------------
# convert_units: the one tool this notebook adds. Exact factors, the generator's own; every unit
# the prompts use, and the usual spellings of each. Served by ToolHost like the simulators.
# ---------------------------------------------------------------------------------------------

#: kind -> unit -> (multiplier, offset): value_in_base = (value - offset) * multiplier.
UNIT_KINDS: dict[str, dict[str, tuple[float, float]]] = {
    "mass flow": {"kg/s": (1, 0), "t/h": (1 / 3.6, 0), "kg/h": (1 / 3600, 0),
                  # water taken at 1000 kg/m3, as the generator does for permeate flow
                  "m3/h": (1 / 3.6, 0), "m3/d": (1 / 86.4, 0)},
    "volume flow": {"m3/s": (1, 0), "l/s": (1e-3, 0), "m3/h": (1 / 3600, 0), "m3/d": (1 / 86400, 0),
                    "mgd": (1 / MGD_PER_M3S, 0)},
    "salinity": {"mass fraction": (1, 0), "g/kg": (1e-3, 0), "‰": (1e-3, 0), "%": (1e-2, 0),
                 "mg/kg": (1e-6, 0), "ppm": (1e-6, 0)},
    "concentration": {"g/l": (1, 0), "mg/l": (1e-3, 0), "kg/m3": (1, 0), "ppm": (1e-3, 0)},
    "temperature": {"c": (1, 0), "f": (1 / 1.8, 32)},
    "area": {"m2": (1, 0), "ft2": (1 / FT2_PER_M2, 0)},
    "water permeability": {"m/s/pa": (1, 0), "lmh/bar": (1 / 3.6e11, 0)},
    "salt permeability": {"m/s": (1, 0), "lmh": (1 / 3.6e6, 0)},
    "flux": {"lmh": (1, 0), "l/m2/h": (1, 0), "gfd": (1 / GFD_PER_LMH, 0)},
    "pressure": {"bar": (1, 0), "kpa": (1e-2, 0), "mpa": (10, 0), "psi": (1 / PSI_PER_BAR, 0), "pa": (1e-5, 0)},
    "cost": {"$/m3": (1, 0), "$/kgal": (1 / 3.785411784, 0)},
    "efficiency": {"fraction": (1, 0), "%": (1e-2, 0)},
}
_UNIT_SYNONYMS = {
    "degc": "c", "celsius": "c", "°c": "c", "℃": "c", "deg c": "c", "degf": "f", "fahrenheit": "f", "°f": "f",
    "deg f": "f", "sqm": "m2", "sq m": "m2", "m^2": "m2", "sqft": "ft2", "sq ft": "ft2", "ft^2": "ft2",
    "m^3/s": "m3/s", "m^3/h": "m3/h", "m^3/d": "m3/d", "m3/hr": "m3/h", "m3/day": "m3/d", "m3/hour": "m3/h",
    "l/h": None, "lps": "l/s", "liters/s": "l/s", "litres/s": "l/s", "t/hr": "t/h", "tonne/h": "t/h",
    "tonnes/h": "t/h", "ton/h": "t/h", "tons/h": "t/h", "tph": "t/h", "kg/hr": "kg/h", "kg/hour": "kg/h",
    "permille": "‰", "per mille": "‰", "promille": "‰", "percent": "%", "pct": "%", "% salt by mass": "%",
    "w/w": "mass fraction", "massfraction": "mass fraction", "mass frac": "mass fraction",
    "mg/kg": "mg/kg", "l/m2/hr": "l/m2/h", "l/(m2 h)": "l/m2/h", "l/(m2·h)": "l/m2/h", "lm2h": "lmh",
    "gal/ft2/d": "gfd", "gal/ft2/day": "gfd", "gfd": "gfd", "kilopascal": "kpa", "megapascal": "mpa",
    "usd/m3": "$/m3", "$ per m3": "$/m3", "$/1000 gal": "$/kgal", "$/1,000 gal": "$/kgal",
    "$/1,000 gallons": "$/kgal", "$/1000 gallons": "$/kgal", "$/1,000 gal": "$/kgal", "$/kgal": "$/kgal",
    "usd/kgal": "$/kgal", "$/thousand gallons": "$/kgal",
    "$/1000gal": "$/kgal", "lmh/bar": "lmh/bar", "l/m2/h/bar": "lmh/bar", "m/s/pa": "m/s/pa", "m/(s pa)": "m/s/pa",
    "m/s": "m/s", "mgd": "mgd", "million gallons per day": "mgd",
}


def _norm_unit(text: str) -> str:
    t = str(text).strip().lower().replace("³", "3").replace("²", "2").replace("μ", "u").replace("·", " ")
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s+per\s+", "/", t)
    t = re.sub(r"\s+", " ", t)
    if t.startswith("degrees "):
        t = t[8:]
    t = t.replace("°", "")
    return _UNIT_SYNONYMS.get(t, t) or t


def convert_units(value: Any, from_unit: str, to_unit: str) -> str:
    """The tool: JSON {"value": ..., "unit": ...}, or {"error": ...}."""
    x = parse_number(value)
    if x is None:
        return json.dumps({"error": f"value must be a number, got {value!r}"})
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    if f == t:
        return json.dumps({"value": float(f"{x:.6g}"), "unit": to_unit}, ensure_ascii=False)
    kinds = [k for k, units in UNIT_KINDS.items() if f in units and t in units]
    if not kinds:
        known = sorted({u for units in UNIT_KINDS.values() for u in units})
        return json.dumps({"error": f"cannot convert {from_unit!r} to {to_unit!r}; units: {', '.join(known)}"})
    (mf, of), (mt, ot) = UNIT_KINDS[kinds[0]][f], UNIT_KINDS[kinds[0]][t]
    base = (x - of) * mf
    return json.dumps({"value": float(f"{base / mt + ot:.6g}"), "unit": to_unit}, ensure_ascii=False)


CONVERT_DECL = {"type": "function", "function": {
    "name": CONVERT,
    "description": (
        "Convert one number between units.\n\n"
        "Supported: mass flow (kg/s, t/h, kg/h; m3/h and m3/d of water at 1000 kg/m3), volume flow "
        "(m3/s, L/s, m3/h, m3/d, MGD), salinity as a mass fraction (mass fraction, g/kg, \u2030, %, mg/kg, "
        "ppm), concentration (g/L, mg/L, kg/m3, ppm), temperature (C, F), area (m2, ft2), pressure "
        "(bar, kPa, MPa, psi), water permeability A (m/s/Pa, LMH/bar), salt permeability B (m/s, LMH), "
        "flux (LMH, gfd), cost ($/m3, $/kgal), efficiency (fraction, %).\n\n"
        "Returns JSON {\"value\": ..., \"unit\": ...}. On failure returns {\"error\": ...} rather than raising.\n"),
    "parameters": {"properties": {"value": {"title": "Value", "type": "number"},
                                  "from_unit": {"title": "From Unit", "type": "string"},
                                  "to_unit": {"title": "To Unit", "type": "string"}},
                   "required": ["value", "from_unit", "to_unit"], "title": "convert_unitsArguments",
                   "type": "object"}}}


def nice_in(lo: float, hi: float, rng: random.Random) -> tuple[float, float] | None:
    """A round number in [lo, hi], preferring coarse steps: (value, step)."""
    if not (hi >= lo > 0):
        return None
    top = math.floor(math.log10(hi))
    for exp in range(top, top - 9, -1):
        for mult in (5, 2, 1):
            step = mult * 10.0 ** exp
            a, b = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
            if a <= b:
                return round(rng.randint(a, min(b, a + 3)) * step, 10), step
    return None


# ---------------------------------------------------------------------------------------------
# Phrase banks: (train/dev/test phrasings, holdout_shift-only phrasings).
# ---------------------------------------------------------------------------------------------

OPENERS = {
    "ro_check": (["Quick check on one of our seawater RO trains.",
                  "Can you sanity-check this operating point for me?",
                  "Night shift here, I want to confirm we're inside our limits.",
                  "We're reviewing a setpoint before handing it to operations.",
                  "Need a compliance check on one RO unit.",
                  "I'm looking at one of the RO trains and want a second opinion."],
                 ["Operations flagged this train for a limits check.",
                  "Before tomorrow's audit, please verify this unit."]),
    "ro_scan_pressure": (["We need to pick a pump setpoint for one of our RO trains.",
                          "Feed conditions changed and we're re-tuning the high-pressure pump.",
                          "We're commissioning a new train and choosing its operating pressure.",
                          "Our pump runs on a VFD and we want to settle on a setpoint."],
                         ["The pump contractor is asking us for a setpoint.",
                          "Seasonal re-tuning time for the high-pressure pump."]),
    "ro_scan_area": (["We're sizing the membrane area for an RO train.",
                      "Design question: how much membrane should this train get?",
                      "We're deciding how many elements to load into a new train.",
                      "Expansion project, and the membrane area isn't fixed yet."],
                     ["The design review wants the membrane area settled.",
                      "We have to order elements soon and need the area."]),
    "ro_membrane": (["Procurement sent us membrane offers for a retrofit.",
                     "We're replacing the elements on one train and have a shortlist.",
                     "Several suppliers quoted replacement membranes.",
                     "We have to choose replacement elements for an existing train."],
                    ["Vendor bids came in for the replacement membranes.",
                     "The element tender closed and we have the finalists."]),
    "swro_check": (["Looking at the whole desalination plant now.",
                    "Management wants plant-level energy and cost numbers.",
                    "Plant-wide check, please.",
                    "I need the economics of our seawater plant as it runs today."],
                   ["Finance is asking whether the plant hits its targets.",
                    "Board meeting prep: plant energy and cost."]),
    "swro_option": (["We're comparing configurations for a seawater RO plant.",
                     "We have a few options for the plant's pump and energy-recovery setup.",
                     "Choosing between plant configurations.",
                     "Our consultant laid out some plant configurations."],
                    ["Several plant configurations are on the table.",
                     "The EPC contractor gave us alternatives to compare."]),
}
FACTS = {
    "feed_flow_mass_kg_s": (["Feed flow to the train is {v}.", "The train gets {v} of feed.",
                             "We feed it {v}.", "It runs on {v} of seawater feed."],
                            ["Intake delivers {v} to this train.", "Feed rate sits at {v}."]),
    "feed_nacl_mass_frac": (["Feed salinity is {v}.", "The seawater comes in at {v}.",
                             "The lab puts the feed at {v}.", "Our intake water measures {v}."],
                            ["The latest grab sample reads {v}.", "Salinity at the intake is {v}."]),
    "feed_temperature_c": (["Water temperature is {v}.", "The feed is at {v}.",
                            "It's {v} at the intake.", "Feed temperature is {v}."],
                           ["We're seeing {v} water today.", "Intake temperature is holding at {v}."]),
    "membrane_area_m2": (["The train has {v} of membrane.", "Installed membrane area is {v}.",
                          "There's {v} of active membrane area.", "The rack holds {v} of membrane."],
                         ["Total membrane area comes to {v}.", "It's built with {v} of membrane."]),
    "AB": (["The elements have a water permeability of {A} and a salt permeability of {B}.",
            "Membrane specs: A = {A}, B = {B}.", "The datasheet gives A as {A} and B as {B}.",
            "Water permeability is {A}; salt permeability is {B}."],
           ["Per the vendor, water permeability is {A} and salt permeability {B}.",
            "The elements are rated at A = {A} and B = {B}."]),
    "feed_pressure_bar": (["The high-pressure pump delivers {v}.", "Feed pressure is {v}.",
                           "We run it at {v}.", "Pump discharge is {v}."],
                          ["Pressure at the membrane inlet is {v}.", "The pump is set to {v}."]),
    "feed_flow_m3_s": (["The plant takes in {v} of seawater.", "Plant feed is {v}.",
                        "Intake flow to the plant is {v}.", "We treat {v} of seawater."],
                       ["The intake pumps supply {v}.", "Raw-water flow is {v}."]),
    "feed_tds_g_L": (["Feed TDS is {v}.", "The seawater carries {v} of dissolved solids.",
                      "Raw water TDS is {v}."], ["Dissolved solids in the feed come to {v}."]),
    "swro_temp": (["Seawater temperature is {v}.", "The intake is at {v}.", "Feed temperature is {v}."],
                  ["The sea is at {v} this month."]),
    "p1_pressure_bar": (["The high-pressure pump runs at {v}.", "Pump discharge pressure is {v}.",
                         "The HP pump is set to {v}."], ["The main pump delivers {v}."]),
    "p1_efficiency": (["The high-pressure pump is {v} efficient.", "Pump efficiency is {v}."],
                      ["The HP pump runs at {v} efficiency."]),
    "erd_px": (["Brine energy goes back through a pressure exchanger at {v} efficiency.",
                "Energy recovery is a pressure exchanger, {v} efficient."],
               ["A pressure exchanger recovers the brine energy at {v} efficiency."]),
    "erd_pat": (["Brine energy is recovered by a pump-as-turbine at {v} efficiency.",
                 "The energy-recovery device is a pump-as-turbine running at {v} efficiency."],
                ["A pump-as-turbine on the brine line recovers energy at {v} efficiency."]),
}
DISTURBANCE = (["{name} just moved from {old} to {new}.", "{name} went from {old} to {new} after {event}.",
                "Since {event}, {lname} is {new} (it was {old})."],
               ["{name} shifted from {old} to {new} overnight."])
DISTURB_NAMES = {"feed_nacl_mass_frac": "Feed salinity", "feed_temperature_c": "Water temperature",
                 "feed_flow_mass_kg_s": "Feed flow"}
EVENTS = {"feed_nacl_mass_frac": ["the storm", "the tide turned", "the dry spell set in", "the river plume moved off",
                                  "the spring tide"],
          "feed_temperature_c": ["the cold front came through", "the heatwave started", "the upwelling started",
                                 "the storm", "the seasonal turnover"],
          "feed_flow_mass_kg_s": ["the intake screen clogged", "one intake pump tripped", "the strainer fouled",
                                  "the storm", "the bypass opened"]}
PROPOSAL = (["We're thinking of running the pump at {v}.", "The proposal is to set feed pressure to {v}."],
            ["Operations wants to try {v} on the pump."])
CONSTRAINTS = {
    ("permeate.flow_kg_s", ">="): (["We need at least {v} of permeate.",
                                    "Production can't drop below {v} of permeate.",
                                    "The permeate target is {v} minimum."],
                                   ["Downstream needs no less than {v} of product water."]),
    ("permeate.nacl_ppm", "<="): (["Permeate NaCl has to stay at or below {v}.",
                                   "Product water must be under {v} NaCl.",
                                   "The permeate salinity limit is {v}."],
                                  ["Our quality cap on permeate NaCl is {v}."]),
    ("performance.water_recovery_pct", "<="): (["Recovery can't exceed {v}.",
                                                "We don't run recovery above {v}.",
                                                "The recovery limit is {v}."],
                                               ["Keep recovery at {v} or less."]),
    ("flux.water_LMH", "<="): (["Flux must stay at or below {v}.",
                                "The membrane warranty caps flux at {v}.",
                                "Don't go past {v} of flux."], ["Average flux is limited to {v}."]),
    ("performance.salt_rejection_pct", ">="): (["Salt rejection needs to be at least {v}.",
                                                "We require {v} rejection or better."],
                                               ["Rejection must not fall under {v}."]),
    ("costing.specific_energy_kWh_m3", "<="): (["Specific energy has to stay at or below {v}.",
                                                "The energy budget is {v}.",
                                                "We can't exceed {v} of specific energy."],
                                               ["Our SEC ceiling is {v}."]),
    ("costing.LCOW_usd_m3", "<="): (["The levelized cost of water must be {v} or lower.",
                                     "Water cost can be at most {v}."],
                                    ["The tariff only works if LCOW stays at or under {v}."]),
    ("performance.product_flow_m3_s", ">="): (["The plant must deliver at least {v} of product water.",
                                               "Contract output is {v} minimum.",
                                               "We need {v} of product water or more."],
                                              ["We're committed to supplying no less than {v}."]),
    ("performance.system_recovery_pct", ">="): (["Overall plant recovery has to be at least {v}.",
                                                 "System recovery can't fall below {v}."],
                                                ["We need plant recovery of {v} or better."]),
}
QUESTIONS = {
    "ro_check": (["Does it meet all of these limits, and if not, which ones fail?",
                  "Are we within every limit? Tell me which ones we break, if any.",
                  "Is this point compliant with all the limits above?", "Does this operating point pass?"],
                 ["Which of these limits, if any, would be violated?"]),
    ("feed_pressure_bar", "lowest"): (["What's the lowest setting that meets all the limits? If none does, say so.",
                                       "Which is the lowest pressure that satisfies everything, if any?",
                                       "How low can we set the pump and still meet every limit? If no setting works, tell me."],
                                      ["Give me the minimum setpoint that keeps us compliant, or tell me it can't be done."]),
    ("feed_pressure_bar", "highest"): (["What's the highest setting we can run without breaking any limit? If none works, say so.",
                                        "How far up can we push the pressure and still meet every limit, if at all?",
                                        "Which is the highest pressure that still satisfies everything?"],
                                       ["What's the top setpoint that keeps us inside every limit, if any?"]),
    ("membrane_area_m2", "lowest"): (["What's the smallest area that meets all the limits? If none does, say so.",
                                      "Which is the least membrane area that satisfies everything, if any?"],
                                     ["How little membrane can we get away with and stay within every limit? Say so if none works."]),
    ("membrane_area_m2", "highest"): (["What's the largest area we can install without breaking a limit? If none works, say so.",
                                       "Which is the biggest area that still meets every limit, if any?"],
                                      ["How much membrane can we put in before a limit breaks? Tell me if no option works."]),
    "ro_membrane": (["Which is the cheapest one that meets all the limits? If none does, say so.",
                     "Pick the least expensive membrane that keeps us within every limit, or tell me none qualifies."],
                    ["Which of them is the lowest-cost element that passes everything, if any?"]),
    "swro_check": (["Does the plant meet these targets, and which ones does it miss if not?",
                    "Are we hitting every target? Tell me which ones we miss, if any."],
                   ["Which of these targets, if any, does the plant fail?"]),
    ("swro_option", "costing.specific_energy_kWh_m3"): (
        ["Which option has the lowest specific energy while meeting the targets? If none meets them, say so.",
         "Of the options that meet the targets, which uses the least energy per cubic metre?"],
        ["Which configuration meets the targets at the lowest SEC, if any does?"]),
    ("swro_option", "costing.LCOW_usd_m3"): (
        ["Which option gives the lowest levelized cost of water while meeting the targets? If none meets them, say so.",
         "Of the options that meet the targets, which one makes the cheapest water?"],
        ["Which configuration meets the targets at the lowest LCOW, if any does?"]),
}
CANDIDATES = {
    ("feed_pressure_bar", "range"): (["The pump can be set from {a} to {b} in steps of {s}.",
                                      "Allowed setpoints run from {a} to {b} in {s} steps."],
                                     ["The VFD goes from {a} to {b} in increments of {s}."]),
    ("feed_pressure_bar", "list"): (["We can choose among {list}.", "The candidate setpoints are {list}."],
                                    ["The options on the table are {list}."]),
    ("membrane_area_m2", "range"): (["We could install anywhere from {a} to {b} in steps of {s}.",
                                     "Area options run from {a} to {b} in {s} increments."],
                                    ["The rack allows {a} to {b} in steps of {s}."]),
    ("membrane_area_m2", "list"): (["The area options are {list}.", "We're considering {list}."],
                                   ["The layouts on the table give {list}."]),
}
MEMBRANE_LIST = (["The offers, cheapest first: {items}.", "In order of price, lowest first: {items}."],
                 ["Ranked by cost, cheapest first: {items}."])
VALUES_REQUEST = (["Also give me the {m}{where}.", "Please include the {m}{where}.",
                   "I'd also like the {m}{where}."], ["And what {m} would we see{where}?"])
DISTRACTORS = (["SDI15 has been around {sdi:.1f}.", "Feed pH is {ph:.1f}.", "Turbidity is under {ntu:.1f} NTU.",
                "Antiscalant dosing is {asc:.1f} mg/L.", "The cartridge filters were changed {days} days ago."],
               ["Chlorine residual is zero after the SBS dose.", "Permeate boron was {boron:.1f} mg/L last week."])

RANGES = {
    "base": {"ro": {"flow": (1.5, 6.0), "sal": (0.028, 0.042), "temp": (15, 35), "area": (150, 600),
                    "A": (3.0e-12, 5.5e-12), "B": (2.0e-8, 5.0e-8), "p": (45, 70)},
             "swro": {"flow": (0.10, 0.40), "tds": (30, 42), "temp": (15, 32), "p": (60, 80),
                      "p1_eff": (0.75, 0.88), "pxr": (0.88, 0.97), "pat": (0.75, 0.92)}},
    # holdout_shift: brinier, colder feed and bigger trains / smaller plants, unseen phrasings.
    "shift": {"ro": {"flow": (6.0, 8.0), "sal": (0.041, 0.046), "temp": (8, 14), "area": (500, 800),
                     "A": (3.0e-12, 5.5e-12), "B": (2.0e-8, 5.0e-8), "p": (60, 80)},
              "swro": {"flow": (0.05, 0.10), "tds": (42, 45), "temp": (10, 15), "p": (68, 82),
                       "p1_eff": (0.75, 0.88), "pxr": (0.88, 0.97), "pat": (0.75, 0.92)}},
}
FAMILIES = ("ro_check", "ro_scan_pressure", "ro_scan_area", "ro_membrane", "swro_check", "swro_option")
SPLITS = {"train": (1200, 101), "dev": (240, 202), "test": (240, 303), "holdout_shift": (120, 404)}
LEVEL_MIX = (0.40, 0.35, 0.25)


# ---------------------------------------------------------------------------------------------
# Tool host: the deployed MCP argument handling, minus the rate limiter.
# ---------------------------------------------------------------------------------------------

class ToolHost:
    """Runs a tool call exactly as mcp_watertap/server.py serves it.

    FastMCP validates arguments with the tool's pydantic model (unknown keys are dropped, numeric
    strings coerced), calls the function and wraps any exception as "Error executing tool ...".
    The server's own function is used, unwrapped from the rate limiter. server.py imports
    reaktoro_model, which is not installed here; a stub stands in for it -- neither exposed tool
    touches it.
    """

    def __init__(self) -> None:
        import logging
        import types
        logging.disable(logging.CRITICAL)
        if "reaktoro_model" not in sys.modules:
            stub = types.ModuleType("reaktoro_model")
            stub.ReaktoroSimulationError = type("ReaktoroSimulationError", (RuntimeError,), {})
            stub.MOLAR_MASS, stub.DEFAULT_COMPOSITION, stub.DEFAULT_MINERALS = {}, {}, ()
            stub.available_minerals = lambda: []
            stub.equilibrate = stub.composition_salinity = None
            sys.modules["reaktoro_model"] = stub
        sys.path.insert(0, str(MCP_DIR))
        import server  # noqa: E402
        self.server = server
        self.tools = {name: server.mcp._tool_manager.get_tool(name) for name in TOOLS}

    def call(self, name: str, arguments: Any) -> str:
        if name == CONVERT:
            a = arguments if isinstance(arguments, dict) else {}
            try:
                return convert_units(a.get("value"), str(a.get("from_unit", "")), str(a.get("to_unit", "")))
            except Exception as exc:  # noqa: BLE001
                return f"Error executing tool {name}: {exc}"
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        try:
            meta = tool.fn_metadata
            parsed = meta.arg_model.model_validate(meta.pre_parse_json(arguments or {}))
            return tool.fn.__wrapped__(**parsed.model_dump_one_level())
        except Exception as exc:  # noqa: BLE001 -- mirrors Tool.run
            return f"Error executing tool {name}: {exc}"


def result_is_error(text: str | None) -> bool:
    if not text:
        return True
    if text.startswith("Error executing tool") or text.startswith("Unknown tool"):
        return True
    try:
        return "error" in json.loads(text)
    except ValueError:
        return True


def metric(result: dict, path: str) -> float:
    node: Any = result
    for part in path.split("."):
        node = node[part]
    return float(node)


# ---------------------------------------------------------------------------------------------
# Scenario generation.
# ---------------------------------------------------------------------------------------------

class Retry(Exception):
    """The sampled scenario cannot carry the target answer; draw another."""


class Gen:
    def __init__(self, rng: random.Random, split: str, level: int, host: ToolHost, u: float = 0.5):
        self.rng, self.split, self.level, self.host = rng, split, level, host
        #: One uniform draw per case, fixed across attempts, that picks the target answer. Redrawing
        #: it on every retry would favour whichever answers are easiest to fit.
        self.u = u
        self.hold = split == "holdout_shift"
        self.R = RANGES["shift" if self.hold else "base"]
        self.levels: list[int] = []
        self.units: dict[str, dict] = {}  # input key -> unit its value was stated in
        self.inputs: dict[str, dict] = {}  # input key -> {"q", "user_value", "unit"} for the shared inputs
        self.last_input: dict | None = None
        self.n_solves = 0

    def target(self, k: int, p_none: float) -> int | None:
        """The target option: none with probability p_none, else a position spread evenly over k."""
        return None if self.u < p_none else min(k - 1, int((self.u - p_none) / (1 - p_none) * k))

    def pick(self, bank: tuple[list[str], list[str]]) -> str:
        return self.rng.choice(bank[1] if self.hold else bank[0])

    def unit(self, options: list[dict]) -> dict:
        u = self.rng.choice([u for u in options if u["level"] <= self.level])
        self.levels.append(u["level"])
        return u

    def quantity(self, q: str, lo: float, hi: float, u: dict | None = None,
                 key: str | None = None) -> tuple[float, str, dict]:
        u = u or self.unit(INPUT_UNITS[q])
        user = round_grid(to_user(u, self.rng.uniform(lo, hi)), u["grid"])
        self.last_input = {"q": q, "user_value": user, "unit": u["name"]}
        if key:
            self.inputs[key] = self.last_input
        return to_tool(u, user), u["fmt"].format(user), u

    def solve(self, tool: str, args: dict) -> dict:
        self.n_solves += 1
        text = self.host.call(tool, args)
        if result_is_error(text):
            raise Retry(f"{tool} failed: {text[:120]}")
        result = json.loads(text)
        return {"args": args, "result": text, "metrics": {m: metric(result, m) for m in METRICS[tool]}}

    # -- thresholds ----------------------------------------------------------------------------

    def threshold_options(self, path: str, op: str, values: list[float]) -> list[dict]:
        """Every margin-respecting way one threshold can split these values."""
        m, spread = MARGIN[path], SPREAD[path]
        vs = sorted(set(values))
        out = []
        for i in range(len(vs) + 1):
            below = vs[i - 1] if i > 0 else None
            above = vs[i] if i < len(vs) else None
            lo = below * (1 + m) if below is not None else above * (1 - spread)
            hi = above * (1 - m) if above is not None else below * (1 + spread)
            if path in PCT_METRICS:
                hi = min(hi, 99.9)
            if lo > hi:
                continue
            # The threshold sits strictly between `below` and `above`.
            if op == ">=":
                passing = [above is not None and v >= above for v in values]
            else:
                passing = [below is not None and v <= below for v in values]
            out.append({"lo": lo, "hi": hi, "pass": passing})
        return out

    def fit_thresholds(self, cons: list[tuple[str, str]], points: list[dict], accept) -> list[dict]:
        """Choose thresholds so the points' pass/fail pattern satisfies ``accept``; render them."""
        options = [self.threshold_options(p, op, [pt["metrics"][p] for pt in points]) for p, op in cons]
        combos = [[]]
        for opts in options:
            combos = [c + [o] for c in combos for o in opts]
        good = [c for c in combos if accept([[o["pass"][j] for o in c] for j in range(len(points))])]
        if not good:
            raise Retry("no threshold design gives the target answer")
        chosen = self.rng.choice(good)
        rendered = []
        for (path, op), o in zip(cons, chosen):
            u = self.unit(METRIC_UNITS[path])
            nice = nice_in(to_user(u, o["lo"]), to_user(u, o["hi"]), self.rng)
            if nice is None:
                raise Retry("no round threshold fits")
            user, step = nice
            tool_t = to_tool(u, user)
            for pt in points:  # the rendered, rounded threshold must keep every margin
                v = pt["metrics"][path]
                if abs(v - tool_t) < MARGIN[path] * abs(v) * 0.999:
                    raise Retry("rounded threshold broke a margin")
            text = u["fmt"].format(fmt_step(user, step))
            rendered.append({"metric": path, "op": op, "threshold": tool_t, "user_value": user,
                             "user_unit": u["name"], "text": text})
        return rendered

    @staticmethod
    def passes(con: dict, v: float) -> bool:
        return v >= con["threshold"] if con["op"] == ">=" else v <= con["threshold"]

    def violations(self, cons: list[dict], pt: dict) -> list[str]:
        return sorted(c["metric"] for c in cons if not self.passes(c, pt["metrics"][c["metric"]]))

    # -- shared rendering ------------------------------------------------------------------------

    def fact(self, key: str, value_text: str) -> str:
        return self.pick(FACTS[key]).format(v=value_text)

    def constraint_sentences(self, cons: list[dict]) -> list[str]:
        return [self.pick(CONSTRAINTS[(c["metric"], c["op"])]).format(v=c["text"]) for c in cons]

    def values_request(self, paths: list[str], where: str) -> str:
        names = [self.rng.choice(METRIC_NAMES[p]) for p in paths]
        m = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return self.pick(VALUES_REQUEST).format(m=m, where=where)

    def distractor(self) -> str:
        r = self.rng
        return self.pick(DISTRACTORS).format(sdi=r.uniform(1.5, 4.5), ph=r.uniform(7.6, 8.3),
                                             ntu=r.uniform(0.1, 0.9), asc=r.uniform(1.0, 4.0),
                                             days=r.randint(2, 30), boron=r.uniform(0.6, 1.4))

    def assemble(self, opener: str, facts: list[str], cons: list[str], extra: list[str],
                 question: str, values: str | None) -> str:
        r = self.rng
        r.shuffle(facts)
        r.shuffle(cons)
        if r.random() < 0.25:
            facts.insert(r.randrange(len(facts) + 1), self.distractor())
        ask = " ".join(extra + [question] + ([values] if values else []))
        body = " ".join(facts)
        limits = " ".join(cons)
        if r.random() < 0.5:
            return f"{opener} {body} {limits} {ask}"
        return f"{opener} {body}\n\n{limits} {ask}"

    def pick_values(self, tool: str, cons: list[dict]) -> list[str]:
        constrained = [c["metric"] for c in cons]
        n = self.rng.choice([1, 1, 2])
        pool = list(METRICS[tool])
        first = self.rng.choice(constrained) if constrained and self.rng.random() < 0.6 else self.rng.choice(pool)
        rest = [p for p in pool if p != first]
        return [first] + self.rng.sample(rest, n - 1)

    # -- RO scenario pieces ----------------------------------------------------------------------

    def ro_base(self, skip: tuple[str, ...] = ()) -> tuple[dict, dict]:
        R, r = self.R["ro"], self.rng
        stated, texts = {}, {}
        for key, q, rng_key in (("feed_flow_mass_kg_s", "ro_flow", "flow"),
                                ("feed_nacl_mass_frac", "salinity", "sal"),
                                ("membrane_area_m2", "area", "area"),
                                ("feed_pressure_bar", "pressure", "p")):
            if key not in skip:
                stated[key], texts[key], self.units[key] = self.quantity(q, *R[rng_key], key=key)
        if r.random() < 0.85:
            key = "feed_temperature_c"
            stated[key], texts[key], self.units[key] = self.quantity("temp", *R["temp"], key=key)
        if "A_comp" not in skip and r.random() < 0.75:
            stated["A_comp"], a_txt, _ = self.quantity("A", *R["A"], key="A_comp")
            stated["B_comp"], b_txt, _ = self.quantity("B", *R["B"], key="B_comp")
            texts["AB"] = (a_txt, b_txt)
        return stated, texts

    def ro_fact_sentences(self, texts: dict) -> list[str]:
        out = []
        for key, txt in texts.items():
            if key == "AB":
                out.append(self.pick(FACTS["AB"]).format(A=txt[0], B=txt[1]))
            else:
                out.append(self.fact(key, txt))
        return out

    def ro_constraint_pool(self) -> list[tuple[str, str]]:
        return [("permeate.flow_kg_s", ">="), ("permeate.nacl_ppm", "<="),
                ("performance.water_recovery_pct", "<="), ("flux.water_LMH", "<="),
                ("performance.salt_rejection_pct", ">=")]


def check_family(g: Gen, tool: str) -> dict:
    """ro_check / swro_check: one operating point, pass or fail, which limits break."""
    r = g.rng
    if tool == RO:
        stated, texts = g.ro_base()
        extra = []
        story = r.random()
        if story < 0.35:  # a disturbance: the old value is a distractor, the new one is stated
            key = r.choice([k for k in DISTURB_NAMES if k in texts])
            u = g.units[key]
            old_tool = stated[key] * r.choice([0.85, 0.9, 0.93, 1.07, 1.1, 1.15])
            old_txt = u["fmt"].format(round_grid(to_user(u, old_tool), u["grid"]))
            sentence = g.pick(DISTURBANCE).format(name=DISTURB_NAMES[key], lname=DISTURB_NAMES[key].lower(),
                                                  old=old_txt, new=texts[key], event=r.choice(EVENTS[key]))
            del texts[key]
            facts = g.ro_fact_sentences(texts) + [sentence]
        elif story < 0.6:  # a proposed setpoint
            p_txt = texts.pop("feed_pressure_bar")
            facts = g.ro_fact_sentences(texts) + [g.pick(PROPOSAL).format(v=p_txt)]
        else:
            facts = g.ro_fact_sentences(texts)
        pt = g.solve(RO, stated)
        pool = g.ro_constraint_pool()
        opener, question = g.pick(OPENERS["ro_check"]), g.pick(QUESTIONS["ro_check"])
        family = "ro_check"
    else:
        stated, facts = swro_base(g)
        pt = g.solve(SWRO, stated)
        pool = [("costing.specific_energy_kWh_m3", "<="), ("costing.LCOW_usd_m3", "<="),
                ("performance.product_flow_m3_s", ">="), ("performance.system_recovery_pct", ">=")]
        opener, question = g.pick(OPENERS["swro_check"]), g.pick(QUESTIONS["swro_check"])
        family, extra = "swro_check", []
    cons = r.sample(pool, r.choice([2, 3, 3, 4] if tool == RO else [2, 3, 3]))
    want_pass = g.u < 0.5
    rendered = g.fit_thresholds(cons, [pt], lambda m: all(m[0]) == want_pass)
    decision = "pass" if want_pass else "fail"
    vals = g.pick_values(tool, rendered)
    prompt = g.assemble(opener, facts, g.constraint_sentences(rendered), extra, question,
                        g.values_request(vals, r.choice(["", " at this point"])))
    pt["key"], pt["label"] = "p0", "point"
    return {"family": family, "tool": tool, "prompt": prompt, "points": [pt], "constraints": rendered,
            "inputs": g.inputs, "decision_kind": "passfail",
            "gold": {"decision": decision, "violations": g.violations(rendered, pt),
                     "values": {p: pt["metrics"][p] for p in vals}},
            "proof": ["p0"]}


def swro_base(g: Gen, skip: tuple[str, ...] = ()) -> tuple[dict, list[str]]:
    R, r = g.R["swro"], g.rng
    stated, facts = {}, []
    stated["feed_flow_m3_s"], t, _ = g.quantity("swro_flow", *R["flow"], key="feed_flow_m3_s")
    facts.append(g.fact("feed_flow_m3_s", t))
    stated["feed_tds_g_L"], t, _ = g.quantity("tds", *R["tds"], key="feed_tds_g_L")
    facts.append(g.fact("feed_tds_g_L", t))
    if r.random() < 0.8:
        stated["feed_temperature_c"], t, _ = g.quantity("temp", *R["temp"], key="feed_temperature_c")
        facts.append(g.pick(FACTS["swro_temp"]).format(v=t))
    if "p1_pressure_bar" not in skip:
        stated["p1_pressure_bar"], t, _ = g.quantity("pressure", *R["p"], key="p1_pressure_bar")
        facts.append(g.fact("p1_pressure_bar", t))
    if "p1_efficiency" not in skip and r.random() < 0.6:
        stated["p1_efficiency"], t, _ = g.quantity("eff", *R["p1_eff"], key="p1_efficiency")
        facts.append(g.fact("p1_efficiency", t))
    if "erd" not in skip and r.random() < 0.7:
        args, text = erd_config(g, r.random() < 0.7)
        stated.update(args)
        facts.append(text)
    return stated, facts


def erd_config(g: Gen, px: bool) -> tuple[dict, str]:
    R = g.R["swro"]
    if px:
        eff, t, _ = g.quantity("eff", *R["pxr"], key="pxr_efficiency")
        return {"erd_type": "pressure_exchanger", "pxr_efficiency": eff}, g.pick(FACTS["erd_px"]).format(v=t)
    eff, t, _ = g.quantity("eff", *R["pat"], key="erd_efficiency")
    return {"erd_type": "pump_as_turbine", "erd_efficiency": eff}, g.pick(FACTS["erd_pat"]).format(v=t)


def scan_family(g: Gen, param: str) -> dict:
    """ro_scan_pressure / ro_scan_area: an ordered set of settings, lowest or highest that passes."""
    r = g.rng
    stated, texts = g.ro_base(skip=(param,))
    q = "pressure" if param == "feed_pressure_bar" else "area"
    u = g.unit(INPUT_UNITS[q])
    step = r.choice(SCAN_STEPS[(q, u["name"])])
    k = r.choice([3, 4, 4, 5])
    lo, hi = (g.R["ro"]["p"] if q == "pressure" else g.R["ro"]["area"])
    span_tool = to_tool(u, to_user(u, lo) + (k - 1) * step) - lo
    start_tool = r.uniform(lo, max(lo, hi - span_tool))
    start = round_grid(to_user(u, start_tool), step)
    users = [round(start + j * step, 10) for j in range(k)]
    points = []
    for j, uv in enumerate(users):
        pt = g.solve(RO, {**stated, param: to_tool(u, uv)})
        pt["key"], pt["label"], pt["user_value"] = f"p{j}", u["fmt"].format(uv), uv
        pt["inputs"] = {param: {"q": q, "user_value": uv, "unit": u["name"]}}
        points.append(pt)
    objective = r.choice(["lowest", "highest"])
    # Constraints must be monotone along the scan; classify each as failing at the low or high end.
    monotone = []
    for path, op in g.ro_constraint_pool():
        vs = [p["metrics"][path] for p in points]
        diffs = [b - a for a, b in zip(vs, vs[1:])]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            inc = diffs[0] > 0
            monotone.append((path, op, "low" if (op == ">=") == inc else "high"))
    need = "low" if objective == "lowest" else "high"
    if not any(side == need for *_, side in monotone):
        raise Retry("no constraint binds on the needed side")
    binding = r.choice([c for c in monotone if c[2] == need])
    others = [c for c in monotone if c != binding]
    chosen = [binding] + r.sample(others, min(len(others), r.choice([1, 1, 2])))
    sides = {path: side for path, _, side in chosen}
    target = g.target(k, 0.2)

    def decide(mask):
        passing = [j for j, row in enumerate(mask) if all(row)]
        if not passing:
            return None
        return passing[0] if objective == "lowest" else passing[-1]

    cons = [(p, op) for p, op, _ in chosen]
    rendered = g.fit_thresholds(cons, points, lambda m: decide(m) == target)
    status = []
    for pt in points:
        failed = {sides[c["metric"]] for c in rendered if not g.passes(c, pt["metrics"][c["metric"]])}
        status.append(failed)
    decision, proof = scan_proof(status, objective)
    assert decision == target
    extra = [g.pick(CANDIDATES[(param, "range")]).format(a=u["fmt"].format(users[0]), b=u["fmt"].format(users[-1]),
                                                         s=u["fmt"].format(step))
             if r.random() < 0.5 else
             g.pick(CANDIDATES[(param, "list")]).format(list=_or_list([p["label"] for p in points]))]
    family = "ro_scan_pressure" if q == "pressure" else "ro_scan_area"
    vals = g.pick_values(RO, rendered)  # asked for either way; graded only when a setting works
    prompt = g.assemble(g.pick(OPENERS[family]), g.ro_fact_sentences(texts), g.constraint_sentences(rendered),
                        extra, g.pick(QUESTIONS[(param, objective)]),
                        g.values_request(vals, r.choice([" at that setting", " for the one you pick", ""])))
    values = {} if decision is None else {p: points[decision]["metrics"][p] for p in vals}
    return {"family": family, "tool": RO, "prompt": prompt, "inputs": g.inputs,
            "points": points, "constraints": rendered, "decision_kind": "number", "objective": objective,
            "scan": {"param": param, "unit": u["name"], "users": users},
            "gold": {"decision": None if decision is None else users[decision],
                     "decision_tool": None if decision is None else to_tool(u, users[decision]),
                     "violations": [], "values": values},
            "proof": [f"p{j}" for j in proof]}


def scan_proof(status: list[set], objective: str) -> tuple[int | None, list[int]]:
    """The answer, and the fewest points whose simulation proves it, given monotone metrics."""
    k = len(status)
    lows = [j for j, s in enumerate(status) if "low" in s]
    highs = [j for j, s in enumerate(status) if "high" in s]
    if lows and lows != list(range(len(lows))):
        raise Retry("low-side failures are not a prefix")
    if highs and highs != list(range(k - len(highs), k)):
        raise Retry("high-side failures are not a suffix")
    passing = [j for j, s in enumerate(status) if not s]
    if passing:
        if objective == "lowest":
            i = passing[0]
            return i, ([i - 1, i] if i > 0 else [i])
        i = passing[-1]
        return i, ([i, i + 1] if i < k - 1 else [i])
    both = [j for j, s in enumerate(status) if s == {"low", "high"}]
    if both:
        return None, [both[0]]
    if not highs:
        return None, [k - 1]
    if not lows:
        return None, [0]
    return None, [lows[-1], highs[0]]


def _or_list(items: list[str]) -> str:
    return ", ".join(items[:-1]) + " or " + items[-1]


def membrane_family(g: Gen) -> dict:
    """ro_membrane: candidate (A, B) pairs in price order; the cheapest that passes, or none."""
    r = g.rng
    stated, texts = g.ro_base(skip=("A_comp",))
    k = r.choice([2, 3, 3, 4])
    ua, ub = g.unit(INPUT_UNITS["A"]), g.unit(INPUT_UNITS["B"])
    points, items, seen = [], [], set()
    for j in range(k):
        a, a_txt, _ = g.quantity("A", 2.8e-12, 6.0e-12, ua)
        a_in = g.last_input
        b, b_txt, _ = g.quantity("B", 1.5e-8, 6.0e-8, ub)
        b_in = g.last_input
        if (a_txt, b_txt) in seen:
            raise Retry("duplicate membrane")
        seen.add((a_txt, b_txt))
        pt = g.solve(RO, {**stated, "A_comp": a, "B_comp": b})
        label = "ABCD"[j]
        pt["key"], pt["label"] = f"p{j}", label
        pt["inputs"] = {"A_comp": a_in, "B_comp": b_in}
        points.append(pt)
        items.append(f"Membrane {label} (A = {a_txt}, B = {b_txt})")
    pool = g.ro_constraint_pool()
    cons = [("permeate.flow_kg_s", ">="), ("permeate.nacl_ppm", "<=")] + r.sample(pool[2:], r.choice([0, 0, 1]))
    target = g.target(k, 0.2)

    def decide(mask):
        return next((j for j, row in enumerate(mask) if all(row)), None)

    rendered = g.fit_thresholds(cons, points, lambda m: decide(m) == target)
    decision = target
    proof = list(range(k)) if decision is None else list(range(decision + 1))
    vals = g.pick_values(RO, rendered)
    prompt = g.assemble(g.pick(OPENERS["ro_membrane"]), g.ro_fact_sentences(texts), g.constraint_sentences(rendered),
                        [g.pick(MEMBRANE_LIST).format(items="; ".join(items))], g.pick(QUESTIONS["ro_membrane"]),
                        g.values_request(vals, r.choice([" for the one you pick", ""])))
    return {"family": "ro_membrane", "tool": RO, "prompt": prompt, "points": points, "constraints": rendered,
            "inputs": g.inputs, "decision_kind": "label",
            "gold": {"decision": None if decision is None else "ABCD"[decision], "violations": [],
                     "values": {} if decision is None else {p: points[decision]["metrics"][p] for p in vals}},
            "proof": [f"p{j}" for j in proof]}


def option_family(g: Gen) -> dict:
    """swro_option: plant configurations; the lowest SEC or LCOW among those meeting the targets."""
    r = g.rng
    dims = r.sample(["p1_pressure_bar", "erd", "p1_efficiency"], r.choice([1, 2, 2]))
    stated, facts = swro_base(g, skip=tuple(dims))
    k = r.choice([2, 3, 3])
    R = g.R["swro"]
    u_p = g.unit(INPUT_UNITS["pressure"]) if "p1_pressure_bar" in dims else None
    u_e = g.unit(INPUT_UNITS["eff"])
    points, items = [], []
    for j in range(k):
        args, pieces, inputs = dict(stated), [], {}
        if "p1_pressure_bar" in dims:
            args["p1_pressure_bar"], t, _ = g.quantity("pressure", *R["p"], u_p)
            inputs["p1_pressure_bar"] = g.last_input
            pieces.append(f"the pump at {t}")
        if "p1_efficiency" in dims:
            args["p1_efficiency"], t, _ = g.quantity("eff", *R["p1_eff"], u_e)
            inputs["p1_efficiency"] = g.last_input
            pieces.append(f"a pump efficiency of {t}")
        if "erd" in dims:
            px = r.random() < 0.6
            eff, t, _ = g.quantity("eff", *(R["pxr"] if px else R["pat"]), u_e)
            if px:
                args.update(erd_type="pressure_exchanger", pxr_efficiency=eff)
                inputs["pxr_efficiency"] = g.last_input
                pieces.append(f"a pressure exchanger at {t} efficiency")
            else:
                args.update(erd_type="pump_as_turbine", erd_efficiency=eff)
                inputs["erd_efficiency"] = g.last_input
                pieces.append(f"a pump-as-turbine at {t} efficiency")
        label = "ABC"[j]
        items.append(f"Option {label}: " + ", ".join(pieces) + ".")
        pt = g.solve(SWRO, args)
        pt["key"], pt["label"] = f"p{j}", label
        pt["inputs"] = inputs
        points.append(pt)
    if len({tuple(sorted((k2, str(v)) for k2, v in p["args"].items())) for p in points}) < k:
        raise Retry("duplicate options")
    objective = r.choice(["costing.specific_energy_kWh_m3", "costing.LCOW_usd_m3"])
    pool = [("performance.product_flow_m3_s", ">="), ("costing.LCOW_usd_m3", "<="),
            ("costing.specific_energy_kWh_m3", "<="), ("performance.system_recovery_pct", ">=")]
    cons = [("performance.product_flow_m3_s", ">=")] + r.sample(
        [c for c in pool[1:] if c[0] != objective], r.choice([0, 1]))
    target = g.target(k, 0.15)
    obj = [p["metrics"][objective] for p in points]

    def decide(mask):
        passing = sorted((obj[j], j) for j, row in enumerate(mask) if all(row))
        if not passing:
            return None
        if len(passing) > 1 and passing[1][0] < passing[0][0] * 1.02:
            return "tie"
        return passing[0][1]

    rendered = g.fit_thresholds(cons, points, lambda m: decide(m) == target)
    decision = target
    vals = g.pick_values(SWRO, rendered)
    prompt = g.assemble(g.pick(OPENERS["swro_option"]), facts, g.constraint_sentences(rendered),
                        [" ".join(items)], g.pick(QUESTIONS[("swro_option", objective)]),
                        g.values_request(vals, r.choice([" for the option you pick", ""])))
    return {"family": "swro_option", "tool": SWRO, "prompt": prompt, "points": points, "constraints": rendered,
            "inputs": g.inputs, "decision_kind": "label", "objective": objective,
            "gold": {"decision": None if decision is None else "ABC"[decision], "violations": [],
                     "values": {} if decision is None else {p: points[decision]["metrics"][p] for p in vals}},
            "proof": [f"p{j}" for j in range(k)]}


FAMILY_FNS = {
    "ro_check": lambda g: check_family(g, RO),
    "ro_scan_pressure": lambda g: scan_family(g, "feed_pressure_bar"),
    "ro_scan_area": lambda g: scan_family(g, "membrane_area_m2"),
    "ro_membrane": membrane_family,
    "swro_check": lambda g: check_family(g, SWRO),
    "swro_option": option_family,
}

_HOST: ToolHost | None = None


def _host() -> ToolHost:
    global _HOST
    if _HOST is None:
        _HOST = ToolHost()
    return _HOST


def make_case(job: tuple[str, int, int]) -> dict:
    split, seed, idx = job
    family = FAMILIES[idx % len(FAMILIES)]
    rng = random.Random(f"{seed}:{idx}")
    level = rng.choices((0, 1, 2), LEVEL_MIX)[0]
    u = random.Random(f"{seed}:{idx}:target").random()
    solves, reasons = 0, Counter()
    for attempt in range(200):
        g = Gen(rng, split, level, _host(), u)
        try:
            case = FAMILY_FNS[family](g)
        except Retry as exc:
            solves += g.n_solves
            reasons[str(exc).split(":")[0]] += 1
            continue
        solves += g.n_solves
        case.update(id=f"{family}-{split}-{idx:05d}", split=split, level=max(g.levels, default=0),
                    optimal_calls=len(case["proof"]), attempts=attempt + 1, solves=solves)
        return case
    raise RuntimeError(f"{split}/{idx} ({family}): no case after 200 attempts: {dict(reasons)}")


# ---------------------------------------------------------------------------------------------
# Reward. Pure Python: runs in the training venv without WaterTAP.
# ---------------------------------------------------------------------------------------------

WEIGHTS = {"calls": 0.25, "decision": 0.35, "violations": 0.10, "values": 0.20, "format": 0.05,
           "efficiency": 0.05}
ARG_TOL = 0.005      # relative tolerance on each tool argument
VALUE_TOL = 0.015    # relative tolerance on each reported number (arguments off by ARG_TOL move outputs <= 1.43%)
UNGATED = 0.3        # share of the decision/violations credit paid without the simulations that prove it
NONE_WORDS = {"none", "null", "n/a", "no option", "not feasible", "infeasible", "nothing", "no"}


def parse_number(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        m = re.search(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?", x)
        if m:
            try:
                return float(m.group().replace(",", ""))
            except ValueError:
                return None
    return None


def _same_arg(a: Any, b: Any) -> bool:
    if b is None or isinstance(b, str):
        return (str(a).strip().lower() if a is not None else None) == (b.lower() if b else None)
    x = parse_number(a)
    return x is not None and abs(x - b) <= ARG_TOL * abs(b)


def call_fidelity(case: dict, call: dict) -> tuple[float, str | None]:
    """How faithfully one call reproduces a point of the scenario: (score in [0,1], covered point).

    Scored against the closest point: the share of the scenario's stated arguments the call got
    right (after unit conversion, within ARG_TOL), minus 0.25 for each argument it set that the user
    never gave (unless it equals the tool default) and for each argument the tool does not have.
    A call covers a point when it gets every argument right, changes nothing else and ran cleanly.
    """
    name, args = call.get("name"), call.get("arguments")
    if name != case["tool"] or not isinstance(args, dict):
        return 0.0, None
    defaults = DEFAULTS[name]
    known = {k: v for k, v in args.items() if k in defaults}
    unknown = len(args) - len(known)
    ran = not result_is_error(call.get("result"))
    best, covered = 0.0, None
    for pt in case["points"]:
        exp = pt["args"]
        right = sum(_same_arg(known.get(k, defaults[k]), v) for k, v in exp.items())
        stray = sum(1 for k, v in known.items() if k not in exp and not _same_arg(v, defaults[k]))
        score = max(0.0, right / len(exp) - 0.25 * (stray + unknown))
        if right == len(exp) and stray == 0 and ran:
            covered = pt["key"]
        best = max(best, score)
    return best, covered


def json_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every top-level {...} in text, string-aware."""
    spans, depth, start, in_str, esc = [], 0, 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = depth > 0
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def extract_answer(text: str) -> tuple[dict | None, tuple[int, int] | None, bool]:
    """The first JSON object with a "decision" key: (object, span, nothing-after-it)."""
    for s, e in json_spans(text or ""):
        try:
            obj = json.loads(text[s:e])
        except ValueError:
            continue
        if isinstance(obj, dict) and "decision" in obj:
            return obj, (s, e), not text[e:].strip()
    return None, None, False


def _none_like(x: Any) -> bool:
    return x is None or (isinstance(x, str) and x.strip().lower().rstrip(".") in NONE_WORDS)


def decision_correct(case: dict, pred: Any) -> bool:
    gold, kind = case["gold"]["decision"], case["decision_kind"]
    if _none_like(pred):
        return gold is None
    if gold is None:
        return False
    if kind == "passfail":
        return isinstance(pred, str) and pred.strip().lower().rstrip(".") in (gold, gold + "s", gold + "ed")
    if kind == "number":
        x = parse_number(pred)
        if x is None:
            return False
        return (abs(x - gold) <= 0.002 * abs(gold)
                or abs(x - case["gold"]["decision_tool"]) <= ARG_TOL * abs(case["gold"]["decision_tool"]))
    m = re.match(r"\s*(?:option|membrane|configuration|config)?\s*([A-Da-d])\b", str(pred), re.I)
    return bool(m) and m.group(1).upper() == gold


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flatten(v, key + "."))
            else:
                out[key] = v
    return out


def score(case: dict, calls: list[dict], final_text: str) -> dict:
    """Reward for one episode, with the credits a trainer places on tokens.

    calls: [{"name", "arguments" (dict, or None if unparseable), "result" (tool text)}] in order.
    final_text: the last assistant turn. Credits: ("call", i) at that call's end; ("decision",),
    ("violations",) and ("value", path) at the answer field; ("close",) at the answer's closing
    brace (format + efficiency).
    """
    comps: dict[str, float] = {}
    is_sim = [c.get("name") != CONVERT for c in calls]
    n = sum(is_sim)  # simulations; convert_units calls earn and cost nothing
    fid = [call_fidelity(case, c) if sim else (0.0, None) for c, sim in zip(calls, is_sim)]
    covered = {cov for _, cov in fid if cov}
    evidence = set(case["proof"]) <= covered
    comps["calls"] = sum(f for (f, _), sim in zip(fid, is_sim) if sim) / n if n else 0.0
    ans, _, clean_end = extract_answer(final_text)
    ans = ans or {}
    correct = bool(ans) and decision_correct(case, ans.get("decision"))
    gate = 1.0 if evidence else UNGATED
    comps["decision"] = gate * correct
    if case["decision_kind"] == "passfail":
        pred = ans.get("violations")
        pred_set = {str(x).strip() for x in pred} if isinstance(pred, list) else None
        gold_set = set(case["gold"]["violations"])
        if pred_set is None:
            v = 0.0
        elif not gold_set:
            v = float(not pred_set)
        else:
            v = 2 * len(pred_set & gold_set) / (len(pred_set) + len(gold_set))
        comps["violations"] = gate * v
    gold_vals = case["gold"]["values"]
    value_hits = {}
    if gold_vals:
        pred_vals = flatten(ans.get("values"))
        for path, gv in gold_vals.items():
            x = parse_number(pred_vals.get(path))
            value_hits[path] = x is not None and abs(x - gv) <= VALUE_TOL * abs(gv)
        comps["values"] = sum(value_hits.values()) / len(gold_vals)
    comps["format"] = float(bool(ans) and clean_end and isinstance(ans.get("violations"), list)
                            and isinstance(ans.get("values"), dict))
    comps["efficiency"] = min(1.0, case["optimal_calls"] / n) if (correct and evidence and n) else 0.0
    z = sum(WEIGHTS[k] for k in comps)
    total = sum(WEIGHTS[k] * v for k, v in comps.items()) / z
    credits = [(("call", i), WEIGHTS["calls"] * f / n / z if sim else 0.0) for i, ((f, _), sim) in enumerate(zip(fid, is_sim))]
    credits.append((("decision",), WEIGHTS["decision"] * comps["decision"] / z))
    if "violations" in comps:
        credits.append((("violations",), WEIGHTS["violations"] * comps["violations"] / z))
    for path, hit in value_hits.items():
        credits.append((("value", path), WEIGHTS["values"] * hit / len(gold_vals) / z))
    credits.append((("close",), (WEIGHTS["format"] * comps["format"] + WEIGHTS["efficiency"] * comps["efficiency"]) / z))
    return {"total": total, "components": comps, "evidence": evidence, "covered": sorted(covered),
            "n_calls": n, "n_converts": len(calls) - n, "decision_correct": correct, "credits": credits}


# ---------------------------------------------------------------------------------------------
# Reference episodes and null baselines.
# ---------------------------------------------------------------------------------------------

def gold_answer(case: dict) -> dict:
    g = case["gold"]
    return {"decision": g["decision"], "violations": g["violations"],
            "values": {k: float(f"{v:.4g}") for k, v in g["values"].items()}}


def oracle_calls(case: dict) -> list[dict]:
    by_key = {p["key"]: p for p in case["points"]}
    return [{"name": case["tool"], "arguments": {k: (float(f"{v:.6g}") if isinstance(v, float) else v)
                                                 for k, v in by_key[key]["args"].items()},
             "result": by_key[key]["result"]} for key in case["proof"]]


def baselines(case: dict) -> dict[str, float]:
    """Totals for the reference episode and for strategies that should not earn much."""
    gold = gold_answer(case)
    calls = oracle_calls(case)
    out = {"oracle": score(case, calls, json.dumps(gold))["total"],
           "oracle_plus_2_calls": score(case, calls + calls[:1] * 2, json.dumps(gold))["total"],
           "gold_answer_no_calls": score(case, [], json.dumps(gold))["total"]}
    wrong = dict(gold)
    kind = case["decision_kind"]
    if kind == "passfail":
        wrong["decision"] = "fail" if gold["decision"] == "pass" else "pass"
    else:
        wrong["decision"] = "Z" if kind == "label" else -1
    out["proof_calls_wrong_decision"] = score(case, calls, json.dumps(wrong))["total"]
    # No tools: the best fixed guess a policy could learn per family.
    if kind == "passfail":
        guesses = {"no_calls_say_pass": "pass", "no_calls_say_fail": "fail"}
    elif kind == "number":
        users = case["scan"]["users"]
        guesses = {"no_calls_first_option": users[0], "no_calls_last_option": users[-1], "no_calls_none": None}
    else:
        guesses = {"no_calls_first_option": "A", "no_calls_last_option": case["points"][-1]["label"],
                   "no_calls_none": None}
    for name, dec in guesses.items():
        out[name] = score(case, [], json.dumps({"decision": dec, "violations": [], "values": {}}))["total"]
    return out


# ---------------------------------------------------------------------------------------------
# Benchmark decontamination.
# ---------------------------------------------------------------------------------------------

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def read_xlsx_text(path: Path, first_sheet_only: bool = False) -> str:
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    parts = []
    sheets = list(ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", _NS))
    for sheet in sheets[:1] if first_sheet_only else sheets:
        target = rels[sheet.get(f"{{{_NS['r']}}}id")].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        for c in ET.fromstring(z.read(target)).iter(f"{{{_NS['m']}}}c"):
            v = c.find("m:v", _NS)
            if c.get("t") == "s" and v is not None:
                parts.append(shared[int(v.text)])
            elif c.get("t") == "inlineStr":
                parts.append("".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t")))
            elif v is not None:
                parts.append(v.text)
    return "\n".join(parts)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())


def _numbers(text: str, sig: int = 4) -> set[str]:
    out = set()
    for m in re.findall(r"\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text):
        try:
            out.add(f"{float(m.replace(',', '')):.{sig}g}")
        except ValueError:
            continue
    return out


#: The inputs that define a scenario; a case "reuses" a benchmark scenario if all of them appear in
#: one benchmark workbook (compared at 4 significant figures, in the tool's units).
CORE_INPUTS = {RO: ("feed_flow_mass_kg_s", "feed_nacl_mass_frac", "membrane_area_m2", "feed_pressure_bar"),
               SWRO: ("feed_flow_m3_s", "feed_tds_g_L", "p1_pressure_bar")}


def decontamination(cases: list[dict], n: int = 8) -> dict:
    """Does any generated prompt share text, or a whole scenario, with the benchmark?"""
    files = sorted(BENCH_DIR.glob("D*/*.xlsx"))
    bench = {f.name: read_xlsx_text(f) for f in files}
    grams = set()
    for text in bench.values():
        w = _words(text)
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    bench_nums = {name: _numbers(text) for name, text in bench.items()}
    hits, examples, reused = [], [], []
    for c in cases:
        w = _words(c["prompt"])
        h = sum(tuple(w[i:i + n]) in grams for i in range(len(w) - n + 1))
        hits.append(h)
        if h and len(examples) < 5:
            examples.append((c["id"], [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)
                                       if tuple(w[i:i + n]) in grams][:3]))
        for p in c["points"]:
            core = {f"{p['args'][k]:.4g}" for k in CORE_INPUTS[c["tool"]] if k in p["args"]}
            match = [name for name, nums in bench_nums.items() if core <= nums]
            if match:
                reused.append((c["id"], p["key"], match[:3]))
                break
    sizes = sorted(len(v) for v in bench_nums.values())
    return {"benchmark_files": len(files), "benchmark_8grams": len(grams),
            "prompts_with_any_8gram": sum(h > 0 for h in hits), "max_8gram_hits": max(hits, default=0),
            "examples": examples, "cases_reusing_a_benchmark_scenario": len(reused),
            "reuse_examples": reused[:10],
            "numbers_per_workbook_median": sizes[len(sizes) // 2] if sizes else 0}


# ---------------------------------------------------------------------------------------------
# Rollouts: the policy in the loop with the real simulators.
#
# The training venv has torch but not WaterTAP, so the simulators run in worker processes under
# .venv-watertap (the ``worker`` subcommand: ToolHost over JSON lines), with a result cache -- the
# simulators are deterministic, so an identical call is answered from memory.
#
# The turn protocol is 03's: only <|im_end|> ends a turn (<|endoftext|> is suppressed); a turn with
# <tool_call> blocks is answered with the results in one user turn and the policy writes again; a
# turn without calls is the answer and ends the episode. Episodes are built by token concatenation,
# which equals the chat template's own rendering (``check-harness`` verifies it on reference episodes).
# ---------------------------------------------------------------------------------------------

WATERTAP_PY = REPO / ".venv-watertap" / "bin" / "python"
MODEL = "Qwen/Qwen3.5-0.8B"
TEMPLATE_KWARGS = {"enable_thinking": False}
RUNS = HERE / f"{PREFIX}_runs"
#: Simulations per episode; calls beyond this are answered with an error instead of run.
MAX_CALLS = 8
#: convert_units calls per episode (a reference episode needs at most 12).
MAX_CONVERTS = 16
#: Tool rounds per episode; a turn that still calls after the last round ends the episode unanswered.
MAX_ROUNDS = 6
CALL_BUDGET_ERROR = "Error: call budget exhausted (8 simulations per question); answer with what you have."
CONVERT_BUDGET_ERROR = "Error: conversion budget exhausted (16 per question); answer with what you have."
MALFORMED_CALL_ERROR = "Error: malformed tool call"


class ToolPool:
    """Worker processes serving ToolHost, and a cache of every call already answered."""

    def __init__(self, n: int = 4) -> None:
        import os
        import subprocess
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        self.procs = [subprocess.Popen([str(WATERTAP_PY), str(Path(__file__).resolve()), "worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
                      for _ in range(n)]
        self.cache: dict[str, str] = {}
        self.solved = 0

    @staticmethod
    def key(name: str, arguments: Any) -> str:
        return json.dumps([name, arguments], sort_keys=True, default=str)

    def _run(self, proc, jobs: list[tuple[str, str, Any]]) -> list[tuple[str, str]]:
        out = []
        for key, name, arguments in jobs:
            proc.stdin.write(json.dumps({"name": name, "arguments": arguments}) + "\n")
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("tool worker died")
            out.append((key, json.loads(line)["result"]))
        return out

    def call_many(self, calls: list[tuple[str, Any]]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        todo, seen = [], set()
        for name, arguments in calls:
            k = self.key(name, arguments)
            if k not in self.cache and k not in seen:
                seen.add(k)
                todo.append((k, name, arguments))
        if todo:
            chunks = [todo[i::len(self.procs)] for i in range(len(self.procs))]
            with ThreadPoolExecutor(len(self.procs)) as ex:
                for part in ex.map(self._run, self.procs, chunks):
                    self.cache.update(part)
            self.solved += len(todo)
        return [self.cache[self.key(n, a)] for n, a in calls]

    def close(self) -> None:
        for p in self.procs:
            p.stdin.close()
            p.wait(timeout=30)


def _schema_types(name: str, param: str) -> set[str]:
    decl = next((d["function"] for d in _tool_decls() if d["function"]["name"] == name), None)
    prop = (decl or {}).get("parameters", {}).get("properties", {}).get(param, {})
    return {t.get("type") for t in prop.get("anyOf", [prop])}


_DECLS: list[dict] | None = None


def _tool_decls() -> list[dict]:
    global _DECLS
    if _DECLS is None:
        _DECLS = json.loads(TOOLS_JSON.read_text())
    return _DECLS


def convert_param(name: str, param: str, raw: str) -> Any:
    """A <parameter> value as the deployed tool-call parser types it: by the tool's schema."""
    value = raw.strip()
    if value.lower() in ("null", "none"):
        return None
    types = _schema_types(name, param)
    if "number" in types or "integer" in types:
        try:
            return float(value)
        except ValueError:
            return value  # left as text; the server's validation rejects it
    if "string" in types:
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_calls(turn: str) -> list[dict]:
    """Every <tool_call> block in one assistant turn, in Qwen3.5's XML form."""
    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", turn, flags=re.S):
        fn = re.fullmatch(r"\s*<function=([^>\s]+)>(.*?)</function>\s*", body, flags=re.S)
        if fn is None:
            calls.append({"name": None, "arguments": None, "malformed": True})
            continue
        name = fn.group(1)
        params = re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", fn.group(2), flags=re.S)
        calls.append({"name": name, "arguments": {k: convert_param(name, k, v) for k, v in params}})
    return calls


def call_block(name: str, arguments: dict) -> str:
    """One call exactly as the chat template writes it in an assistant turn."""
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in arguments.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def render_prompt(tokenizer, case: dict) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": case["prompt"]}]
    return tokenizer.apply_chat_template(msgs, tools=_tool_decls(), tokenize=False, add_generation_prompt=True,
                                         **TEMPLATE_KWARGS)


def generation_prefix(tokenizer) -> str:
    """What add_generation_prompt appends: the assistant header and, thinking off, the empty think block."""
    probe = [{"role": "user", "content": "x"}]
    on = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS)
    off = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False, **TEMPLATE_KWARGS)
    return on[len(off):]


def tool_response_text(results: list[str], prefix: str) -> str:
    """Everything between the policy's <|im_end|> and its next turn."""
    body = "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
    return "\n<|im_start|>user" + body + "<|im_end|>\n" + prefix


class Episode:
    """One multi-turn episode. ``ids`` is everything after the prompt; ``policy`` marks the
    policy's own tokens (False = written by the harness: tool responses and turn headers)."""

    def __init__(self, case: dict, prompt_ids: list[int]):
        self.case, self.prompt_ids = case, prompt_ids
        self.ids: list[int] = []
        self.policy: list[bool] = []
        self.turns: list[str] = []
        self.turn_starts: list[int] = []  # index into ids of each turn's first token
        self.calls: list[dict] = []       # {"name", "arguments", "result", "turn"}
        self.final_text = ""
        self.truncated = False            # a turn hit the token budget before <|im_end|>
        self.unanswered = 0               # calls left when the rounds ran out
        self.answer_closed = False

    @property
    def policy_tokens(self) -> int:
        return sum(self.policy)

    def to_json(self, tokenizer=None) -> dict:
        return {"id": self.case["id"], "family": self.case["family"], "turns": self.turns,
                "calls": [{k: c[k] for k in ("name", "arguments", "turn")} | {"error": result_is_error(c["result"])}
                          for c in self.calls],
                "final_text": self.final_text, "truncated": self.truncated, "unanswered": self.unanswered,
                "answer_closed": self.answer_closed, "policy_tokens": self.policy_tokens,
                "total_tokens": len(self.prompt_ids) + len(self.ids)}


#: Largest prefill (rows x padded length) one generate call may take. A late round of 16 episodes
#: at ~7k tokens each asked for 3.8 GiB more than the 13,000 MiB cap allows; rows are split by size.
MAX_PREFILL_TOKENS = 40_000


def _generate_round(policy, inputs: list[list[int]], budgets: list[int], **kw) -> list[list[int]]:
    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
    out: dict[int, list[int]] = {}
    chunk: list[int] = []
    for i in order + [None]:
        if i is not None and (len(chunk) + 1) * len(inputs[i]) <= MAX_PREFILL_TOKENS:
            chunk.append(i)
            continue
        if chunk:
            rows = _generate_chunk(policy, [inputs[j] for j in chunk], [budgets[j] for j in chunk], **kw)
            out.update(zip(chunk, rows))
        chunk = [i] if i is not None else []
    return [out[i] for i in range(len(inputs))]


def _generate_chunk(policy, inputs: list[list[int]], budgets: list[int], *, pad: int, temperature: float,
                    im_end: int, suppress: int, stop_at_json: bool, tokenizer, cache: dict) -> list[list[int]]:
    import torch
    from transformers import StoppingCriteriaList
    device = next(policy.parameters()).device
    width = max(len(x) for x in inputs)
    input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in inputs], device=device)
    attention = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in inputs], device=device)
    sample = temperature > 0
    stopping = StoppingCriteriaList([_StopAtAnswer(tokenizer, width, len(inputs), cache)]) if stop_at_json else None
    out = policy.generate(input_ids=input_ids, attention_mask=attention, do_sample=sample,
                          temperature=temperature if sample else None, top_p=1.0 if sample else None,
                          top_k=0 if sample else None, max_new_tokens=max(budgets), pad_token_id=pad,
                          eos_token_id=im_end, suppress_tokens=[suppress], stopping_criteria=stopping)
    return [row[:b] for row, b in zip(out[:, width:].tolist(), budgets)]


def _tok_text(tokenizer, t: int, cache: dict) -> str:
    s = cache.get(t)
    if s is None:
        s = cache[t] = tokenizer.decode([t])
    return s


def answer_close_index(tokens: list[int], tokenizer, cache: dict) -> int | None:
    """Index of the token closing the first top-level JSON object of a turn with no <tool_call>."""
    text, ends = "", []
    for t in tokens:
        text += _tok_text(tokenizer, t, cache)
        ends.append(len(text))
    if "<tool_call>" in text:
        return None
    spans = json_spans(text)
    if not spans:
        return None
    close = spans[0][1]
    return next(k for k, e in enumerate(ends) if e >= close)


class _StopAtAnswer:
    """generate stopping criterion: a row is done once a call-free turn's first JSON object closes."""

    def __init__(self, tokenizer, width: int, rows: int, cache: dict):
        self.tokenizer, self.width, self.cache = tokenizer, width, cache
        self.texts = [""] * rows
        self.done = [False] * rows
        self.seen = width

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        new = input_ids[:, self.seen:].tolist()
        self.seen = input_ids.shape[1]
        for i, toks in enumerate(new):
            if self.done[i]:
                continue
            self.texts[i] += "".join(_tok_text(self.tokenizer, t, self.cache) for t in toks)
            if "<tool_call>" not in self.texts[i] and json_spans(self.texts[i]):
                self.done[i] = True
        return torch.tensor(self.done, device=input_ids.device)


def rollout(policy, tokenizer, cases: list[dict], pool: ToolPool, *, temperature: float, max_new_tokens: int,
            stop_at_json: bool = False) -> list[Episode]:
    """Run a batch of episodes with the simulators in the loop.

    ``max_new_tokens`` caps the policy's own tokens summed over its turns. Each round, every live
    episode writes one turn; its calls (up to MAX_CALLS per episode) run on the pool and come back as
    one user turn. A turn without calls is the answer. With ``stop_at_json`` that turn ends where its
    first JSON object closes.
    """
    import torch
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    suppress = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    prefix = generation_prefix(tokenizer)
    cache: dict[int, str] = {}
    eps = [Episode(c, tokenizer(render_prompt(tokenizer, c), add_special_tokens=False).input_ids) for c in cases]
    live = list(range(len(eps)))
    with torch.no_grad():
        for rnd in range(MAX_ROUNDS + 1):
            budgets = {i: max_new_tokens - eps[i].policy_tokens for i in live}
            for i in live:
                if budgets[i] <= 0:
                    eps[i].truncated = True  # the token budget ran out between turns: no answer
            live = [i for i in live if budgets[i] > 0]
            if not live:
                break
            rows = _generate_round(policy, [eps[i].prompt_ids + eps[i].ids for i in live], [budgets[i] for i in live],
                                   pad=tokenizer.pad_token_id, temperature=temperature, im_end=im_end,
                                   suppress=suppress, stop_at_json=stop_at_json, tokenizer=tokenizer, cache=cache)
            pending = []
            for row, i in zip(rows, live):
                ep = eps[i]
                cut = next((k for k, t in enumerate(row) if t == im_end), None)
                closed = answer_close_index(row if cut is None else row[:cut], tokenizer, cache) if stop_at_json else None
                if closed is not None:
                    take, ep.answer_closed = row[: closed + 1], True
                else:
                    take = row if cut is None else row[: cut + 1]
                ep.turn_starts.append(len(ep.ids))
                ep.ids += take
                ep.policy += [True] * len(take)
                text = tokenizer.decode(take, skip_special_tokens=True)
                ep.turns.append(text)
                if closed is not None:
                    ep.final_text = text
                    continue
                if cut is None:
                    ep.truncated = True
                    ep.final_text = text
                    continue
                calls = parse_calls(text)
                if not calls:
                    ep.final_text = text
                    continue
                if rnd == MAX_ROUNDS:
                    ep.unanswered = len(calls)
                    continue
                pending.append((i, calls))
            # Run every pending call of the round on the pool at once.
            jobs, where = [], []
            for i, calls in pending:
                n_sims = sum(x.get("name") != CONVERT for x in eps[i].calls)
                n_conv = len(eps[i].calls) - n_sims
                for c in calls:
                    if c.get("malformed"):
                        c["result"] = MALFORMED_CALL_ERROR
                    elif c.get("name") == CONVERT:
                        n_conv += 1
                        if n_conv > MAX_CONVERTS:
                            c["result"] = CONVERT_BUDGET_ERROR
                        else:
                            jobs.append((c["name"], c["arguments"]))
                            where.append(c)
                    else:
                        n_sims += 1
                        if n_sims > MAX_CALLS:
                            c["result"] = CALL_BUDGET_ERROR
                        else:
                            jobs.append((c["name"], c["arguments"]))
                            where.append(c)
            for c, result in zip(where, pool.call_many(jobs)):
                c["result"] = result
            live = []
            for i, calls in pending:
                ep = eps[i]
                for c in calls:
                    c["turn"] = len(ep.turns) - 1
                    ep.calls.append(c)
                env = tokenizer(tool_response_text([c["result"] for c in calls], prefix), add_special_tokens=False).input_ids
                ep.ids += env
                ep.policy += [False] * len(env)
                live.append(i)
    return eps


def score_episode(ep: Episode) -> dict:
    calls = [{"name": c["name"], "arguments": c["arguments"], "result": c["result"]} for c in ep.calls]
    return score(ep.case, calls, ep.final_text)


def load_policy(adapter: str | Path | None = None, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).to(device)
    model.eval()
    return model, tokenizer


def summarise(episodes: list[Episode], scores: list[dict]) -> dict:
    """What the policy did and what it earned, overall and per family."""
    def block(pairs):
        n = len(pairs)
        eps_, scs = [p[0] for p in pairs], [p[1] for p in pairs]
        mean = lambda xs: sum(xs) / n  # noqa: E731
        all_calls = [c for e in eps_ for c in e.calls if c.get("name") != CONVERT]
        converts = [c for e in eps_ for c in e.calls if c.get("name") == CONVERT]
        comps = defaultdict(list)
        for s in scs:
            for k, v in s["components"].items():
                comps[k].append(v)
        return {
            "n": n, "reward": mean([s["total"] for s in scs]),
            "components": {k: sum(v) / len(v) for k, v in comps.items()},
            "called_any": mean([bool(e.calls) for e in eps_]),
            "first_call_right_tool": mean([next((c["name"] for c in e.calls if c.get("name") != CONVERT), None)
                                           == e.case["tool"] for e in eps_]),
            "calls_per_episode": mean([sum(c.get("name") != CONVERT for c in e.calls) for e in eps_]),
            "converts_per_episode": mean([sum(c.get("name") == CONVERT for c in e.calls) for e in eps_]),
            "converts_errored": sum(result_is_error(c["result"]) for c in converts) / max(1, len(converts)),
            "calls_malformed": sum(bool(c.get("malformed")) for c in all_calls) / max(1, len(all_calls)),
            "calls_errored": sum(result_is_error(c["result"]) for c in all_calls) / max(1, len(all_calls)),
            "evidence": mean([s["evidence"] for s in scs]),
            "answer_found": mean([extract_answer(e.final_text)[0] is not None for e in eps_]),
            "decision_correct": mean([s["decision_correct"] for s in scs]),
            "truncated": mean([e.truncated for e in eps_]),
            "unanswered": mean([e.unanswered > 0 for e in eps_]),
            "policy_tokens": mean([e.policy_tokens for e in eps_]),
            "total_tokens": mean([len(e.prompt_ids) + len(e.ids) for e in eps_]),
        }
    pairs = list(zip(episodes, scores))
    fams = defaultdict(list)
    for p in pairs:
        fams[p[0].case["family"]].append(p)
    return {"overall": block(pairs), "families": {f: block(v) for f, v in sorted(fams.items())}}


# ---------------------------------------------------------------------------------------------
# Reference episodes for a seed: a strategy a policy could run without knowing the answer.
#
# The proving points depend on where the answer is (a scan's answer and its failing neighbour), so
# an episode that simulates only them teaches the model to guess the boundary. The reference
# strategy here is blind: simulate every candidate in one turn, then answer. It needs 1-5 calls,
# costs only efficiency (0.963-1.000 on train), and leaves pruning for PPO to find.
#
# Arguments are written the way a careful person would type them -- the tool-unit value to four
# significant figures (at most 0.05% off; the argument tolerance is 0.5%) -- and the answer is
# built from what the simulator returned for those arguments.
# ---------------------------------------------------------------------------------------------

def _typed(v: Any) -> Any:
    return float(f"{v:.4g}") if isinstance(v, float) else v


def convert_plan(case: dict) -> list[dict]:
    """Every conversion a blind episode needs, in a fixed order, each once: the shared inputs, then
    each candidate's own inputs, then the limits. ``targets`` says where the converted number goes."""
    plan: dict[tuple, dict] = {}

    def add(user_value: float, u: dict, from_name: str, to_sym: str, target: tuple) -> None:
        if not needs_conversion(u):
            return
        key = (float(user_value), from_name, to_sym)  # a float: the chat template renders parsed numbers so
        plan.setdefault(key, {"value": float(user_value), "from_unit": UNIT_SYMBOL[from_name], "to_unit": to_sym,
                              "targets": []})["targets"].append(target)

    for k, inp in case["inputs"].items():
        add(inp["user_value"], unit_spec(inp["q"], inp["unit"]), inp["unit"], TOOL_UNIT[inp["q"]], ("arg", k, None))
    for p in case["points"]:
        for k, inp in p.get("inputs", {}).items():
            add(inp["user_value"], unit_spec(inp["q"], inp["unit"]), inp["unit"], TOOL_UNIT[inp["q"]], ("arg", k, p["key"]))
    for c in case["constraints"]:
        add(c["user_value"], metric_unit_spec(c["metric"], c["user_unit"]), c["user_unit"],
            METRIC_TOOL_UNIT[c["metric"]], ("limit", c["metric"], None))
    return list(plan.values())


def blind_calls(case: dict, converted: dict[tuple, float] | None = None) -> list[dict]:
    """Every candidate simulated: each argument as the user stated it, or as convert_units returned
    it (``converted``: ("arg", key, point key or None) -> value)."""
    converted = converted or {}
    calls = []
    for p in case["points"]:
        args = {}
        for k, v in p["args"].items():
            c = converted.get(("arg", k, p["key"]), converted.get(("arg", k, None)))
            args[k] = _typed(v) if c is None else c
        calls.append({"name": case["tool"], "arguments": args})
    return calls


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def aggregation_line(case: dict, verdicts: list[bool], results: list[str]) -> str | None:
    """The line after the candidates: which passed, and which one the question's objective picks.
    A copy of the verdicts just written and of the objective the question states; none for pass/fail."""
    if case["decision_kind"] == "passfail":
        return None
    labels = [p["label"] for p in case["points"]]
    passing = [i for i, ok in enumerate(verdicts) if ok]
    if case["decision_kind"] == "number":
        word = case["objective"]
        chosen = labels[passing[0] if word == "lowest" else passing[-1]] if passing else "none"
        shown = ", ".join(labels[i] for i in passing) or "none"
    elif case["family"] == "ro_membrane":  # the offers are listed cheapest first
        word = "cheapest"
        chosen = labels[passing[0]] if passing else "none"
        shown = ", ".join(labels[i] for i in passing) or "none"
    else:  # swro_option: the lowest objective among the passing options
        word = "lowest"
        obj = [metric(json.loads(r), case["objective"]) for r in results]
        chosen = labels[min(passing, key=lambda i: obj[i])] if passing else "none"
        shown = ", ".join(f"{labels[i]} {_fmt(obj[i])}" for i in passing) or "none"
    return f"passing: {shown} -> {word}: {chosen}"


def reasoning_text(case: dict, results: list[str], limits: dict[str, float] | None = None) -> str:
    """One line per simulated point: each limited field against its limit (in the tool's units, as
    convert_units returned it when the user stated it in another unit); then the aggregation line."""
    limits = limits or {}
    lines, verdicts = [], []
    for p, r in zip(case["points"], results):
        m = json.loads(r)
        parts, ok = [], True
        for c in case["constraints"]:
            v = metric(m, c["metric"])
            passed = v >= c["threshold"] if c["op"] == ">=" else v <= c["threshold"]
            ok &= passed
            lim = limits.get(c["metric"], c["threshold"])
            parts.append(f"{c['metric']} {_fmt(v)} {'meets' if passed else 'breaks'} {c['op']} {lim:g}")
        if case["family"] == "swro_option":
            parts.append(f"{case['objective']} {_fmt(metric(m, case['objective']))}")
        label = p["label"] if case["decision_kind"] != "passfail" else "Result"
        lines.append(f"{label}: " + "; ".join(parts) + f" -> {'pass' if ok else 'fail'}")
        verdicts.append(ok)
    agg = aggregation_line(case, verdicts, results)
    return "\n".join(lines + ([agg] if agg else []))


def reference_answer(case: dict, results: list[str], reasoning: bool, limits: dict[str, float] | None = None) -> str:
    g = case["gold"]
    decision = g["decision"]
    if case["decision_kind"] == "number" and decision is not None:
        # The setting as the user named it ("2,100 ft²"): valid JSON, and what the contract asks for.
        decision = next(p["label"] for p in case["points"] if p["user_value"] == g["decision"])
    values = {}
    if g["values"]:
        at = next(i for i, p in enumerate(case["points"])
                  if case["decision_kind"] == "passfail" or p.get("user_value", p["label"]) == g["decision"])
        m = json.loads(results[at])
        values = {k: float(_fmt(metric(m, k))) for k in g["values"]}
    obj = json.dumps({"decision": decision, "violations": g["violations"], "values": values})
    return (reasoning_text(case, results, limits) + "\n\n" + obj) if reasoning else obj


def reference_episode(case: dict, pool: ToolPool) -> dict:
    """The blind episode: a convert turn (if anything needs converting), a simulate turn, the answer.
    Verified: the converted numbers are within tolerance, no rounding flips a pass/fail, and the
    reward pays the episode in full."""
    plan = convert_plan(case)
    turns = []
    converted: dict[tuple, float] = {}
    limits: dict[str, float] = {}
    if plan:
        calls = [{"name": CONVERT, "arguments": {"value": p["value"], "from_unit": p["from_unit"], "to_unit": p["to_unit"]}}
                 for p in plan]
        results = pool.call_many([(c["name"], c["arguments"]) for c in calls])
        for p, r in zip(plan, results):
            got = json.loads(r)
            assert "value" in got, (case["id"], p, r)
            for t in p["targets"]:
                if t[0] == "arg":
                    converted[t] = got["value"]
                else:
                    limits[t[1]] = got["value"]
        turns.append({"calls": calls, "results": results})
    sims = blind_calls(case, converted)
    results = pool.call_many([(c["name"], c["arguments"]) for c in sims])
    turns.append({"calls": sims, "results": results})
    by_key = {p["key"]: p for p in case["points"]}
    for c, p, r in zip(sims, case["points"], results):
        for k, v in p["args"].items():  # every argument within tolerance of the scenario's
            assert _same_arg(c["arguments"][k], v), (case["id"], p["key"], k, c["arguments"][k], v)
        seen, gold = json.loads(r), p["metrics"]
        for con in case["constraints"]:  # rounding the arguments must not flip a pass/fail
            ok = lambda x: x >= con["threshold"] if con["op"] == ">=" else x <= con["threshold"]  # noqa: E731
            assert ok(metric(seen, con["metric"])) == ok(gold[con["metric"]]), (case["id"], p["key"], con["metric"])
    for con in case["constraints"]:  # a converted limit is the threshold, to 6 significant figures
        if con["metric"] in limits:
            assert abs(limits[con["metric"]] - con["threshold"]) <= 1e-5 * abs(con["threshold"]), (case["id"], con)
    all_calls = [c | {"result": r} for t in turns for c, r in zip(t["calls"], t["results"])]
    for reasoning in (False, True):
        answer = reference_answer(case, results, reasoning, limits)
        sc = score(case, all_calls, answer)
        assert sc["decision_correct"] and sc["evidence"], (case["id"], reasoning)
    reasoned = reference_answer(case, results, True, limits)
    agg = reasoned.split("\n\n")[0].splitlines()[-1]
    if case["decision_kind"] != "passfail":  # the aggregation line names the gold decision
        chosen = agg.rsplit(": ", 1)[1]
        gold = case["gold"]["decision"]
        want = "none" if gold is None else (next(p["label"] for p in case["points"] if p.get("user_value", p["label"]) == gold))
        assert chosen == want, (case["id"], agg, gold)
    return {"id": case["id"], "turns": turns, "answer": reference_answer(case, results, False, limits),
            "answer_reasoned": reasoned, "reward": sc["total"], "converts": len(plan)}


def build_references(split: str, pool: ToolPool) -> list[dict]:
    """Blind reference episodes for a split, each verified against the reward."""
    return [reference_episode(case, pool) for case in load_split(split)]


def _content_spans(answer: str, passfail: bool, wide: bool) -> list[tuple[int, int]]:
    """Character spans of the answer's content, as opposed to its scaffolding.

    ``wide=False`` (what the seed leaves unlabelled): the text inside a string decision, the literal
    of a null/number decision, the text of each violation, each number -- JSON syntax (quotes,
    brackets, commas) is always scaffolding. `seed-verdict` left the decision's quotes and the
    pass/fail violations list whole, and wrote `"decision": C` and violations as objects and prose.

    ``wide=True`` (what PPO may move): the whole decision literal, quotes included, and in a
    pass/fail question the whole violations list -- null versus "A", or [] versus a list, is itself
    the decision.
    """
    obj, span, _ = extract_answer(answer)
    s0 = span[0]
    text = answer[s0:span[1]]
    spans = []
    m = re.search(r'"decision"\s*:\s*', text)
    p = m.end()
    if text.startswith('"', p):
        e = p + 1
        while text[e] != '"':
            e += 2 if text[e] == "\\" else 1
        spans.append((s0 + p, s0 + e + 1) if wide else (s0 + p + 1, s0 + e))
    else:
        lit = re.match(r'[^,}\]]+', text[p:])
        if lit:
            spans.append((s0 + p, s0 + p + lit.end()))
    # A sampled answer need not have a violations list (`ppo-s0` crashed at step 34 on one without).
    vm = re.search(r'"violations"\s*:\s*\[', text)
    if vm and "]" in text[vm.end():]:
        vs, ve = vm.end(), text.index("]", vm.end())
        if wide and passfail:
            spans.append((s0 + vm.end() - 1, s0 + ve + 1))
        else:
            for im in re.finditer(r'"((?:[^"\\]|\\.)*)"', text[vs:ve]):
                spans.append((s0 + vs + im.start(1), s0 + vs + im.end(1)))
    if '"values"' not in text:
        return spans
    base = text.index('"values"')
    for km in re.finditer(r'": (-?[\d.eE+-]+)', text[base:]):
        spans.append((s0 + base + km.start(1), s0 + base + km.end(1)))
    return spans


_NUM = r"-?[\d.]+(?:e[-+]?\d+)?"


def _reasoning_content_spans(text: str, verdicts: bool) -> list[tuple[int, int]]:
    """Content of the comparison lines: every simulated value, and with ``verdicts`` also every limit,
    each verdict (meets/breaks, pass/fail) and the aggregation line's set and pick. The candidate
    label, the field names and the operators are scaffolding.

    The seed labels the verdicts (``verdicts=False`` here): `seed-scaffold` left them out, wrote "->"
    where "meets"/"breaks" goes on 719 of 720 greedy dev answers, and looped metric, value and limit
    to the token budget on up to 22% of them. PPO keeps them trainable (``verdicts=True``): they are
    the judgement it has to learn, not scaffolding to hold still."""
    spans = []
    for m in re.finditer(rf"(\S+) ({_NUM}) (meets|breaks) (<=|>=) ({_NUM})", text):
        # 05: the seed labels the limit too -- it is a copy of the prompt's number or of what
        # convert_units returned, and 04's policy copied the user's number unconverted on 91 of 102
        # limits that needed converting. Only the simulated value is left to PPO by the seed.
        spans += [m.span(2)] + ([m.span(5), m.span(3)] if verdicts else [])
    for m in re.finditer(rf"(costing\.\S+) ({_NUM})(?=\s*->)", text):
        spans.append(m.span(2))
    if verdicts:
        for m in re.finditer(r"-> (pass|fail)", text):
            spans.append(m.span(1))
        # The aggregation line: the passing set and the pick are content; "passing:" and the
        # objective word are scaffolding. The seed labels the whole line (it is a copy of the
        # verdicts above it and of the objective the question states).
        for m in re.finditer(r"^passing: (.+?) -> \w+: (.+)$", text, flags=re.M):
            spans += [m.span(1), m.span(2)]
    return spans


def decision_class(case: dict) -> str | None:
    """The answer class a label seed balances over; None for the pass/fail families (both answers
    are sampled by the scaffold seed already)."""
    if case["decision_kind"] == "passfail":
        return None
    if case["gold"]["decision"] is None:
        return "none"
    return "setting" if case["decision_kind"] == "number" else str(case["gold"]["decision"])


def pick_label_cases(k: int, seed: int) -> list[str]:
    """k train records per (family, answer class), for a label seed."""
    groups = defaultdict(list)
    for c in load_split("train"):
        cls = decision_class(c)
        if cls is not None:
            groups[(c["family"], cls)].append(c["id"])
    rng = random.Random(seed)
    return sorted(i for key in sorted(groups) for i in rng.sample(groups[key], min(k, len(groups[key]))))


def seed_example(case: dict, ref: dict, tokenizer, *, label: str, reasoning: bool, label_decision: bool = False):
    """(ids, labels) for one reference episode. Prompt and tool responses are context. The calls
    turn and its <|im_end|> are labelled. In the answer turn, ``label="full"`` labels everything;
    ``label="scaffold"`` labels only the scaffolding and leaves the decision, the violations and the
    numbers -- the task's content -- to PPO (the reasoning lines, if any, are content too)."""
    import torch
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    prefix = generation_prefix(tokenizer)
    answer = ref["answer_reasoned"] if reasoning else ref["answer"]
    prompt = tokenizer(render_prompt(tokenizer, case), add_special_tokens=False).input_ids
    ids, lab = list(prompt), [False] * len(prompt)
    for turn in ref["turns"]:  # each calls turn and its <|im_end|> labelled; the tool responses context
        calls_text = "\n".join(call_block(c["name"], c["arguments"]) for c in turn["calls"])
        call_ids = tokenizer(calls_text, add_special_tokens=False).input_ids + [im_end]
        env = tokenizer(tool_response_text(turn["results"], prefix), add_special_tokens=False).input_ids
        ids += call_ids + env
        lab += [True] * len(call_ids) + [False] * len(env)
    enc = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ans_ids = enc.input_ids + [im_end]
    if label == "full":
        ans_lab = [True] * len(ans_ids)
    else:
        content = _content_spans(answer, case["decision_kind"] == "passfail", wide=False)
        if label_decision:  # a label-seed record: the decision itself is labelled too
            content = content[1:]
        if reasoning:
            content += _reasoning_content_spans(answer[: extract_answer(answer)[1][0]], verdicts=False)
        ans_lab = [not any(a < e and b > s for s, e in content) for a, b in enc.offset_mapping] + [True]
    ids += ans_ids
    lab += ans_lab
    ids_t = torch.tensor([ids])
    return ids_t, torch.where(torch.tensor([lab]), ids_t, torch.full_like(ids_t, -100))


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                "out_proj")


def sft_loss(policy, ids, labels):
    """Mean next-token cross-entropy, the 248k-wide head applied only where a label is (03's)."""
    import torch
    model = policy.get_base_model()
    hidden = model.model(input_ids=ids).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    where = targets != -100
    return torch.nn.functional.cross_entropy(model.lm_head(hidden[where]).float(), targets[where])


def train_seed(out_dir: Path, *, label: str, reasoning: bool, epochs: int, lr: float, accum: int, lora_r: int,
               seed: int, limit: int, label_per_class: int = 0, progress=print) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    policy = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=list(LORA_TARGETS),
                                             task_type="CAUSAL_LM")).to("cuda")
    cases = {c["id"]: c for c in load_split("train")}
    refs = [json.loads(line) for line in (DATA / "train_reference.jsonl").read_text().splitlines()]
    random.Random(seed).shuffle(refs)
    refs = refs[:limit] if limit else refs
    label_ids = set(pick_label_cases(label_per_class, seed)) if label_per_class else set()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"model": MODEL, "label": label, "reasoning": reasoning, "epochs": epochs, "lr": lr, "accum": accum,
         "lora_r": lora_r, "seed": seed, "examples": len(refs), "lora_targets": list(LORA_TARGETS),
         "strategy": "blind: convert what needs converting, simulate every candidate, then the answer",
         "label_per_class": label_per_class, "label_case_ids": sorted(label_ids),
         "label_class_counts": dict(Counter(f"{cases[i]['family']}:{decision_class(cases[i])}" for i in label_ids))},
        indent=2) + "\n")
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    with (out_dir / "losses.jsonl").open("w") as fh:
        for epoch in range(epochs):
            losses = []
            for i, ref in enumerate(refs):
                ids, labels = (x.to("cuda") for x in seed_example(cases[ref["id"]], ref, tok, label=label,
                                                                   reasoning=reasoning,
                                                                   label_decision=ref["id"] in label_ids))
                loss = sft_loss(policy, ids, labels)
                (loss / accum).backward()
                if (i + 1) % accum == 0 or i + 1 == len(refs):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                fh.write(json.dumps({"epoch": epoch, "i": i, "loss": losses[-1]}) + "\n")
            policy.save_pretrained(out_dir / f"epoch{epoch + 1}")
            k = max(1, len(losses) // 10)
            progress(f"  epoch {epoch + 1}: loss first-10% {sum(losses[:k]) / k:.4f} -> last-10% {sum(losses[-k:]) / k:.4f}")


# ---------------------------------------------------------------------------------------------
# PPO. 03's actor-critic, carried over to multi-turn episodes with the simulators in the loop.
#
# What changes from 03: an episode has any number of call turns before its answer, the tool's
# responses are ~900 tokens each (context: no log-prob, reward or value), and the credits come from
# score()'s anchors -- each call's end, the decision, the violations, each reported number, the
# answer's closing brace. Log-probabilities are computed only at the policy's own positions: an
# episode is up to ~7k tokens and the vocabulary 248k, so full-span logits would be ~5 GB each.
# ---------------------------------------------------------------------------------------------

SUPPRESSED = -1e9


def _bounds(tokenizer, ids: list[int], cache: dict) -> list[int]:
    """End character offset (in the turn's decoded text) of each token."""
    out, n = [], 0
    for t in ids:
        n += len(_tok_text(tokenizer, t, cache)) if t not in tokenizer.all_special_ids else 0
        out.append(n)
    return out


def _at(bounds: list[int], offset: int) -> int:
    return next((i for i, e in enumerate(bounds) if e >= offset), len(bounds) - 1)


def answer_anchor_offsets(text: str, case: dict) -> dict:
    """Character offsets, in the answer turn, where each answer credit lands."""
    obj, span, _ = extract_answer(text)
    if span is None:
        return {}
    s0, region = span[0], text[span[0]: span[1]]
    out = {"close": span[1]}
    m = re.search(r'"decision"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}\]]+)', region)
    if m:
        out["decision"] = s0 + m.end(1)
    m = re.search(r'"violations"\s*:\s*\[[^\]]*\]', region)
    if m:
        out["violations"] = s0 + m.end()
    for path in case["gold"]["values"]:
        m = re.search(rf'"{re.escape(path)}"\s*:\s*"?({_NUM})', region)
        if m is None and "." in path:
            sec, leaf = path.split(".", 1)
            ms = re.search(rf'"{re.escape(sec)}"\s*:\s*\{{', region)
            if ms:
                m2 = re.search(rf'"{re.escape(leaf)}"\s*:\s*"?({_NUM})', region[ms.end():])
                if m2:
                    out[("value", path)] = s0 + ms.end() + m2.end(1)
                    continue
        if m:
            out[("value", path)] = s0 + m.end(1)
    return out


def answer_content_spans(text: str, case: dict) -> list[tuple[int, int]]:
    """Content of an answer turn (what the scaffold seed left to PPO); if the JSON does not parse,
    every number and verdict word in the turn."""
    obj, span, _ = extract_answer(text)
    spans = []
    if obj is not None:
        try:
            spans += _content_spans(text[: span[1]], case["decision_kind"] == "passfail", wide=True)
        except Exception:  # noqa: BLE001 -- a malformed sampled answer falls back to the regex below
            obj = None
        spans += _reasoning_content_spans(text[: span[0]], verdicts=True)
    if obj is None:
        spans += [m.span() for m in re.finditer(rf"{_NUM}|\b(?:meets|breaks|pass|fail|null)\b", text)]
    return spans


def build_batch(episodes: list[Episode], tokenizer, device) -> dict:
    """Left-padded prompts, the episodes, right padding; the policy's tokens compacted; per-token
    rewards from score()'s credits; the masks the actor's loss uses."""
    import torch
    pad = tokenizer.pad_token_id
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    cache: dict[int, str] = {}
    width = max(len(ep.prompt_ids) for ep in episodes)
    span = max(1, max(len(ep.ids) for ep in episodes))
    n = max(1, max(ep.policy_tokens for ep in episodes))
    rows = {k: [] for k in ("seqs", "attn", "index", "ids", "mask", "answer", "turn_end", "content", "rewards")}
    scores = []
    for ep in episodes:
        left, right = width - len(ep.prompt_ids), span - len(ep.ids)
        rows["seqs"].append([pad] * left + ep.prompt_ids + ep.ids + [pad] * right)
        rows["attn"].append([0] * left + [1] * (len(ep.prompt_ids) + len(ep.ids)) + [0] * right)
        where = [k for k, mine in enumerate(ep.policy) if mine]
        fill = n - len(where)
        mine = [ep.ids[k] for k in where]
        rows["index"].append(where + [0] * fill)
        rows["ids"].append(mine + [pad] * fill)
        rows["mask"].append([1.0] * len(where) + [0.0] * fill)
        # Each turn's policy tokens, in compacted coordinates.
        starts = [sum(ep.policy[:s]) for s in ep.turn_starts] + [len(where)]
        turns = [(starts[t], starts[t + 1]) for t in range(len(ep.turns))]
        answer, turn_end, content, reward = ([0.0] * n for _ in range(4))
        for a, b in turns:
            if b > a and mine[b - 1] == im_end:
                turn_end[b - 1] = 1.0
        sc = score_episode(ep)
        scores.append(sc)
        answered = bool(ep.turns) and ep.final_text == ep.turns[-1] and not parse_calls(ep.final_text)
        if answered:
            a, b = turns[-1]
            bounds = _bounds(tokenizer, mine[a:b], cache)
            for k in range(a, b):
                answer[k] = 1.0
            starts_c = [0] + bounds[:-1]
            for s, e in answer_content_spans(ep.final_text, ep.case):
                for k in range(a, b):
                    if starts_c[k - a] < e and bounds[k - a] > s:
                        content[k] = 1.0
            offsets = answer_anchor_offsets(ep.final_text, ep.case)
        # Credits onto tokens.
        call_rank = Counter()
        for (anchor, amount) in sc["credits"]:
            if anchor[0] == "call":
                c = ep.calls[anchor[1]]
                t = c["turn"]
                k = call_rank[t]
                call_rank[t] += 1
                a, b = turns[t]
                ends = [m.end() for m in re.finditer(r"</tool_call>", ep.turns[t])]
                pos = b - 1 if k >= len(ends) else a + _at(_bounds(tokenizer, mine[a:b], cache), ends[k])
            elif answered:
                a, b = turns[-1]
                key = anchor[0] if anchor[0] != "value" else ("value", anchor[1])
                off = offsets.get(key, offsets.get("close"))
                pos = b - 1 if off is None else a + _at(_bounds(tokenizer, mine[a:b], cache), off)
            else:
                pos = len(where) - 1
            reward[max(0, pos)] += amount
        rows["answer"].append(answer)
        rows["turn_end"].append(turn_end)
        rows["content"].append(content)
        rows["rewards"].append(reward)
    t = lambda x: torch.tensor(x, device=device)  # noqa: E731
    out = {k: t(v) for k, v in rows.items()}
    out.update(span=span, scores=scores)
    return out


def padded_position_ids(attention_mask):
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def _take(x, index):
    if x.dim() == 3:
        import torch
        return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    import torch
    return torch.gather(x, 1, index)


def forward_policy(model, seqs, attn, span: int, index, *, suppress: int, hidden_layer: int | None = None):
    """Log-probs at the policy's tokens (``suppress`` removed from the vocabulary), the critic's
    features at the same positions, and the probability the model puts on ``suppress`` there."""
    import torch
    base = model.get_base_model()
    out = base.model(input_ids=seqs, attention_mask=attn, position_ids=padded_position_ids(attn),
                     output_hidden_states=hidden_layer is not None, use_cache=False)
    offset = seqs.shape[1] - span
    pos = offset + index - 1  # the position whose output predicts each policy token
    logits = base.lm_head(_take(out.last_hidden_state, pos)).float()
    mass = None
    if hidden_layer is not None:
        # Its own small tensor: indexing a softmax keeps the whole (n x 248k) result alive as a view,
        # ~0.6 GB per episode, and `gate40-s0`'s first attempt ran out of memory at step 14 on it.
        mass = torch.exp(logits[..., suppress] - torch.logsumexp(logits, dim=-1)).detach()
    logits[..., suppress] = SUPPRESSED
    targets = _take(seqs[:, offset:], index)
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    hidden = _take(out.hidden_states[hidden_layer], pos).detach() if hidden_layer is not None else None
    return logprobs, hidden, mass


def gae(rewards, values, mask, *, gamma: float = 1.0, lam: float = 1.0):
    import torch
    batch, tokens = mask.shape
    advantages = torch.zeros_like(values)
    running = torch.zeros(batch, dtype=values.dtype, device=values.device)
    for t in range(tokens - 1, -1, -1):
        next_value = values[:, t + 1] * mask[:, t + 1] if t + 1 < tokens else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = (delta + gamma * lam * running) * mask[:, t]
        advantages[:, t] = running
    return advantages * mask, (advantages + values) * mask


def position_clock(returns, mask, bins: int = 20):
    import torch
    sel = mask > 0
    out = torch.zeros_like(returns)
    index = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    binned = (index / lengths * bins).floor().clamp(max=bins - 1).long()
    for b in range(bins):
        hit = (binned == b) & sel
        if hit.any():
            out[hit] = returns[hit].mean()
    return out


def critic_fit(values, returns, mask) -> dict:
    sel = mask > 0
    v, g = values[sel], returns[sel]
    g_var = float(g.var(unbiased=False)) if g.numel() > 1 else 0.0
    if g_var < 1e-6:
        return {"value_ev": None, "value_ev_position": None}
    clock = position_clock(returns, mask)[sel]
    return {"value_ev": round(1.0 - float((g - v).var(unbiased=False)) / g_var, 6),
            "value_ev_position": round(1.0 - float((g - clock).var(unbiased=False)) / g_var, 6)}


def ppo_actor_loss(logprobs, old_logprobs, advantages, mask, *, clip_eps: float):
    import torch
    active = mask.sum().clamp(min=1.0)
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -(torch.min(unclipped, clipped) * mask).sum() / active
    with torch.no_grad():
        stats = {"actor_loss": float(loss), "ratio_mean": float((ratio * mask).sum() / active),
                 "clip_frac": float(((unclipped > clipped) & (mask > 0)).float().sum() / active)}
    return loss, stats


def value_loss(values, old_values, returns, *, clip_eps: float | None):
    import torch
    plain = (values - returns) ** 2
    moved = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps) if clip_eps else values
    loss = torch.max(plain, (moved - returns) ** 2).mean()
    return loss, {"value_loss": float(loss.detach()), "value_mean": float(values.mean().detach()),
                  "return_mean": float(returns.mean())}


def _value_head(hidden_size: int, init_bias: float):
    import torch.nn as nn
    head = nn.Linear(hidden_size, 1, dtype=__import__("torch").float32)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, init_bias)
    return head


def _values(head, hidden):
    return head(hidden.float()).squeeze(-1)


def load_trainable(adapter: Path, device: str = "cuda"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    policy = PeftModel.from_pretrained(base, str(adapter), is_trainable=True).to(device)
    return policy, tok


PPO_DEFAULTS = {
    "start_from": "", "steps": 2000, "prompts_per_step": 8, "max_new_tokens": 2048, "temperature": 1.0,
    "lr": 1e-5, "gamma": 1.0, "lam": 0.95, "clip_eps": 0.2, "value_clip_eps": 0.2, "inner_epochs": 4,
    "value_layer": 23, "value_lr": 9.18987e-4, "value_init_bias": -1.0, "critic_epochs": 8,
    "critic_window": 25, "critic_warmup": 25, "critic_gate_from": 40, "seed": 0, "split": "train",
    "eval_every": 25, "eval_split": "dev", "eval_cases": 120, "eval_batch": 16, "patience": 8,
    "regress_margin": 0.05, "keep_eval_checkpoints": 1, "sample_every": 5,
    # 03's stabilisers, each from a measured drift on this model (see 03): the turn-ending
    # <|im_end|> and the answer's scaffolding out of the actor's loss, Adam eps 1e-6, stop at the
    # answer's closing brace, KL 0.01 to the seed in the reward.
    "freeze_turn_ends": 1, "freeze_answer_structure": 1, "adam_eps": 1e-6, "stop_at_json": 1, "kl_coef": 0.01,
    # Tool gate (this notebook's arithmetic gate): stop once the sampled call fidelity, averaged
    # over 10 steps, falls below this -- the seed taught the calls, and PPO must not lose them.
    "tool_stop": 0.0,
}


def train_ppo(cfg: dict, run_dir: Path, progress=print, resume: bool = False) -> dict:
    import faulthandler
    import signal
    from collections import deque
    import torch
    device = "cuda"
    cases = load_split(cfg["split"])
    eval_cases = random.Random(0).sample(load_split(cfg["eval_split"]), cfg["eval_cases"])
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    policy, tok = load_trainable(run_dir / "last" / "adapter" if resume else HERE / cfg["start_from"])
    reference = None
    if cfg["kl_coef"]:
        reference, _ = load_trainable(HERE / cfg["start_from"])
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
    suppress = tok.convert_tokens_to_ids("<|endoftext|>")
    hidden_size = policy.get_base_model().config.get_text_config().hidden_size
    head = _value_head(hidden_size, cfg["value_init_bias"]).to(device)
    actor_params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(actor_params, lr=cfg["lr"], weight_decay=0.0, eps=cfg["adam_eps"])
    critic_opt = torch.optim.AdamW(head.parameters(), lr=cfg["value_lr"], weight_decay=0.0)
    window: deque = deque(maxlen=cfg["critic_window"])
    metrics_path, eval_path = run_dir / "metrics.jsonl", run_dir / "eval.jsonl"
    history: list[dict] = []
    gate = {"best": None, "best_step": None, "since_best": 0, "regress": 0, "critic_fail": 0, "step0": None}
    start = 0
    if resume:
        # Everything a continuation needs, saved at the last evaluation (`last/`): a restart from the
        # adapter alone is not a continuation (`smoke_test/14`: Adam's moments and the critic's window
        # matter), so the run picks up exactly where it was.
        state = torch.load(run_dir / "last" / "state.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["opt"])
        critic_opt.load_state_dict(state["critic_opt"])
        window.extend(state["window"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        gate, start = state["gate"], state["step"]
        history = [r for r in map(json.loads, metrics_path.read_text().splitlines()) if r["step"] < start]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in history))
        evals = [r for r in map(json.loads, eval_path.read_text().splitlines()) if r["step"] <= start]
        eval_path.write_text("".join(json.dumps(r) + "\n" for r in evals))
        samples = run_dir / "samples.jsonl"
        if samples.exists():
            kept = [r for r in map(json.loads, samples.read_text().splitlines()) if r["step"] < start]
            samples.write_text("".join(json.dumps(r) + "\n" for r in kept))
        progress(f"resumed {run_dir.name} at step {start}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        metrics_path.write_text("")
        eval_path.write_text("")
    pool = ToolPool(6)

    def save_state(step: int) -> None:
        last = run_dir / "last"
        last.mkdir(exist_ok=True)
        policy.save_pretrained(last / "adapter")
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                    "window": list(window), "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "gate": dict(gate), "step": step}, last / "state.pt")

    def critic_window_stats(n=25):
        rows = [r for r in history[-n:] if r.get("value_ev") is not None and r.get("value_ev_position") is not None]
        if not rows:
            return {"critic_residual_median": None, "critic_ahead_frac": None}
        res = sorted(r["value_ev"] - r["value_ev_position"] for r in rows)
        mid = len(res) // 2
        med = res[mid] if len(res) % 2 else (res[mid - 1] + res[mid]) / 2
        return {"critic_residual_median": round(med, 6), "critic_ahead_frac": round(sum(x > 0 for x in res) / len(res), 4)}

    def run_eval(step: int) -> str | None:
        policy.eval()
        state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        eps, scs = [], []
        for b in range(0, len(eval_cases), cfg["eval_batch"]):
            got = rollout(policy, tok, eval_cases[b: b + cfg["eval_batch"]], pool, temperature=0.0,
                          max_new_tokens=cfg["max_new_tokens"], stop_at_json=bool(cfg["stop_at_json"]))
            eps += got
            scs += [score_episode(e) for e in got]
        torch.set_rng_state(state[0])
        torch.cuda.set_rng_state_all(state[1])
        o = summarise(eps, scs)["overall"]
        rec = {"step": step, **{k: o[k] for k in ("reward", "decision_correct", "evidence", "answer_found",
                                                  "calls_per_episode", "calls_errored", "truncated", "policy_tokens")},
               "components": o["components"], **critic_window_stats()}
        reason = None
        if step == 0:
            gate["step0"] = rec["reward"]
        if step and cfg["keep_eval_checkpoints"]:
            policy.save_pretrained(run_dir / f"adapter-step{step}")
        if gate["best"] is None or rec["reward"] > gate["best"]:
            gate.update(best=rec["reward"], best_step=step, since_best=0)
            policy.save_pretrained(run_dir / "adapter-best")
            (run_dir / "best.json").write_text(json.dumps(rec, indent=2) + "\n")
        else:
            gate["since_best"] += 1
            if gate["since_best"] >= cfg["patience"]:
                reason = f"converged: {cfg['patience']} evaluations without beating {gate['best']:.4f} (step {gate['best_step']})"
        gate["regress"] = gate["regress"] + 1 if rec["reward"] < gate["step0"] - cfg["regress_margin"] else 0
        if gate["regress"] >= 2:
            reason = f"no-regression gate: held-out reward {rec['reward']:.4f} below step 0 twice"
        if step >= cfg["critic_gate_from"] and rec["critic_residual_median"] is not None:
            gate["critic_fail"] = gate["critic_fail"] + 1 if rec["critic_residual_median"] <= 0 else 0
            if gate["critic_fail"] >= 2:
                reason = "critic gate: value_ev not above the position clock at two evaluations"
        rec["gate"] = dict(gate)
        with eval_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if step:
            save_state(step)
        crit = rec["critic_residual_median"]
        progress(f"    eval @{step:<5} reward {rec['reward']:.4f} decision {rec['decision_correct']:.3f} "
                 f"evidence {rec['evidence']:.3f} answer {rec['answer_found']:.3f} calls {rec['calls_per_episode']:.2f} "
                 f"critic {'-' if crit is None else f'{crit:+.4f}'} | best {gate['best']:.4f}@{gate['best_step']}")
        return reason

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    progress(f"{MODEL} from {cfg['start_from']} | value layer {cfg['value_layer']} lr {cfg['value_lr']:.3g} "
             f"bias {cfg['value_init_bias']:.4f} | kl {cfg['kl_coef']} | warm-up {cfg['critic_warmup']}")
    stop = run_eval(0) if (cfg["eval_every"] and not resume) else None
    step = start
    try:
        while stop is None and step < cfg["steps"]:
            faulthandler.dump_traceback_later(1800, exit=True)
            t0 = time.perf_counter()
            batch_cases = [rng.choice(cases) for _ in range(cfg["prompts_per_step"])]
            policy.eval()
            episodes = rollout(policy, tok, batch_cases, pool, temperature=cfg["temperature"],
                               max_new_tokens=cfg["max_new_tokens"], stop_at_json=bool(cfg["stop_at_json"]))
            gen_s = time.perf_counter() - t0
            B = build_batch(episodes, tok, device)
            seqs, attn, span, index, mask, rewards = B["seqs"], B["attn"], B["span"], B["index"], B["mask"], B["rewards"]
            with torch.no_grad():
                old_lp, hid, mass = [], [], []
                for i in range(len(seqs)):
                    lp, h, m = forward_policy(policy, seqs[i:i + 1], attn[i:i + 1], span, index[i:i + 1],
                                              suppress=suppress, hidden_layer=cfg["value_layer"])
                    old_lp.append(lp), hid.append(h), mass.append(m)
                old_lp, hidden, mass = torch.cat(old_lp), torch.cat(hid), torch.cat(mass)
                kl = torch.zeros_like(old_lp)
                if reference is not None:
                    ref_lp = torch.cat([forward_policy(reference, seqs[i:i + 1], attn[i:i + 1], span, index[i:i + 1],
                                                       suppress=suppress)[0] for i in range(len(seqs))])
                    kl = (old_lp - ref_lp) * mask
                    rewards = rewards - cfg["kl_coef"] * kl
                old_values = _values(head, hidden) * mask
                adv, returns = gae(rewards, old_values, mask, gamma=cfg["gamma"], lam=cfg["lam"])
            sel = mask > 0
            window.append((hidden[sel].cpu(), old_values[sel].cpu(), returns[sel].cpu()))
            fh_, fv_, fr_ = (torch.cat([w[k] for w in window]).to(device) for k in range(3))
            cstats = {}
            for _ in range(cfg["critic_epochs"]):
                critic_opt.zero_grad(set_to_none=True)
                vl, cstats = value_loss(_values(head, fh_), fv_, fr_, clip_eps=cfg["value_clip_eps"])
                vl.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                critic_opt.step()
            del fh_, fv_, fr_
            actor_mask = mask * (1 - B["turn_end"]) if cfg["freeze_turn_ends"] else mask
            if cfg["freeze_answer_structure"]:
                actor_mask = actor_mask * (1 - B["answer"] + B["content"])
            warming = step < cfg["critic_warmup"]
            astats, grad_norm = {}, torch.zeros(())
            policy.train()
            for _ in range(0 if warming else cfg["inner_epochs"]):
                opt.zero_grad(set_to_none=True)
                ep_stats = defaultdict(float)
                for i in range(len(seqs)):
                    lp, _, _ = forward_policy(policy, seqs[i:i + 1], attn[i:i + 1], span, index[i:i + 1], suppress=suppress)
                    loss, st_ = ppo_actor_loss(lp, old_lp[i:i + 1], adv[i:i + 1], actor_mask[i:i + 1],
                                               clip_eps=cfg["clip_eps"])
                    (loss / len(seqs)).backward()
                    for k, v in st_.items():
                        ep_stats[k] += v / len(seqs)
                grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
                opt.step()
                astats = dict(ep_stats)
            scs = B["scores"]
            active = mask.sum().clamp(min=1)
            mean_adv = float((adv * mask).sum() / active)
            ans = B["answer"] * mask
            rec = {
                "step": step, "critic_warmup": warming,
                "reward_mean": sum(s["total"] for s in scs) / len(scs),
                "calls_fidelity": sum(s["components"]["calls"] for s in scs) / len(scs),
                "decision_correct": sum(s["decision_correct"] for s in scs) / len(scs),
                "evidence": sum(s["evidence"] for s in scs) / len(scs),
                "answered": float(ans.sum(dim=-1).gt(0).float().mean()),
                "calls_mean": sum(len(e.calls) for e in episodes) / len(episodes),
                "truncated": sum(e.truncated for e in episodes) / len(episodes),
                "policy_tokens_mean": float(mask.sum(dim=-1).float().mean()),
                "episode_tokens_mean": sum(len(e.prompt_ids) + len(e.ids) for e in episodes) / len(episodes),
                "kl_mean": round(float(kl.sum() / active), 6),
                "kl_answer": round(float((kl * ans).sum() / ans.sum().clamp(min=1)), 6),
                "suppressed_mass_max": float((mass * mask).max()),
                "adv_mean": round(mean_adv, 6),
                "adv_std": round(float((((adv - mean_adv) * mask) ** 2).sum().div(active).sqrt()), 6),
                **critic_fit(old_values, returns, mask),
                **{k: round(v, 6) for k, v in astats.items()}, **{k: round(v, 6) for k, v in cstats.items()},
                "grad_norm": float(grad_norm), "gen_seconds": round(gen_s, 1),
                "step_seconds": round(time.perf_counter() - t0, 1), "solves": pool.solved,
            }
            history.append(rec)
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if step % cfg["sample_every"] == 0:
                with (run_dir / "samples.jsonl").open("a") as fh:
                    for ep, sc in zip(episodes, scs):
                        fh.write(json.dumps({"step": step, **ep.to_json(), "reward": sc["total"]}) + "\n")
            ev, clock = rec["value_ev"], rec["value_ev_position"]
            progress(f"  step {step:>5} reward {rec['reward_mean']:.4f} calls {rec['calls_fidelity']:.3f} "
                     f"dec {rec['decision_correct']:.2f} evid {rec['evidence']:.2f} V {rec.get('value_mean', 0):.3f} "
                     f"ev {'undef' if ev is None else f'{ev:+.3f}'} clock {'undef' if clock is None else f'{clock:+.3f}'} "
                     f"adv {rec['adv_mean']:+.4f} kl {rec['kl_mean']:.4f} tok {rec['policy_tokens_mean']:.0f}/"
                     f"{rec['episode_tokens_mean']:.0f} {rec['step_seconds']:.0f}s")
            step += 1
            if cfg["eval_every"] and step % cfg["eval_every"] == 0:
                stop = run_eval(step)
            if stop is None and cfg["tool_stop"] and len(history) >= 10:
                recent = sum(r["calls_fidelity"] for r in history[-10:]) / 10
                if recent < cfg["tool_stop"]:
                    stop = f"tool gate: sampled call fidelity over the last 10 steps {recent:.3f} < {cfg['tool_stop']}"
    finally:
        faulthandler.cancel_dump_traceback_later()
        pool.close()
    policy.save_pretrained(run_dir / "adapter")
    torch.save(head.state_dict(), run_dir / "value_head.pt")
    ending = {"stopped_at": step, "reason": stop or f"step budget {cfg['steps']} reached",
              "best_step": gate["best_step"], "best_reward": gate["best"]}
    (run_dir / "ending.json").write_text(json.dumps(ending, indent=2) + "\n")
    progress(f"END {ending}")
    return ending



def episode_report(path: Path, split: str) -> dict:
    """What a probe's episodes did, by family and by unit level, and which answers they said."""
    cases = {c["id"]: c for c in load_split(split)}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        c = cases[r["id"]]
        by[("family", c["family"])].append((r, c))
        by[("level", c["level"])].append((r, c))
        by[("all", "")].append((r, c))
    out = {}
    for key, pairs in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        n = len(pairs)
        calls = [x for r, _ in pairs for x in r["calls"]]
        said = Counter()
        for r, c in pairs:
            obj = extract_answer(r["final_text"])[0]
            if obj is None:
                said["(no answer)"] += 1
            else:
                d = obj.get("decision")
                if _none_like(d):
                    said["none"] += 1
                elif c["decision_kind"] == "number":
                    x = parse_number(d)
                    users = c["scan"]["users"]
                    hit = [i for i, u in enumerate(users) if x is not None and abs(x - u) <= 0.002 * abs(u)]
                    said[f"{hit[0] + 1}/{len(users)}" if hit else "other"] += 1
                else:
                    said[str(d)[:12]] += 1
        out[f"{key[0]}:{key[1]}"] = {
            "n": n, "reward": sum(r["score"]["total"] for r, _ in pairs) / n,
            "answer_found": sum(extract_answer(r["final_text"])[0] is not None for r, _ in pairs) / n,
            "format": sum(r["score"]["components"]["format"] for r, _ in pairs) / n,
            "first_call_right_tool": sum(next((x["name"] for x in r["calls"] if x["name"] != CONVERT), None) == c["tool"]
                                         for r, c in pairs) / n,
            "calls_errored": sum(x["error"] for x in calls) / max(1, len(calls)),
            "calls_fidelity": sum(r["score"]["components"]["calls"] for r, _ in pairs) / n,
            "evidence": sum(r["score"]["evidence"] for r, _ in pairs) / n,
            "decision_correct": sum(r["score"]["decision_correct"] for r, _ in pairs) / n,
            "values": sum(r["score"]["components"].get("values", 0) for r, _ in pairs) / n,
            "truncated": sum(r["truncated"] for r, _ in pairs) / n,
            "unanswered": sum(r["unanswered"] > 0 for r, _ in pairs) / n,
            "said": dict(said.most_common()),
        }
    return out

# ---------------------------------------------------------------------------------------------
# Subcommands.
# ---------------------------------------------------------------------------------------------

def cmd_dump_tools(_args) -> None:
    import asyncio
    host = _host()
    listed = asyncio.run(host.server.mcp.list_tools())
    decls = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                               "parameters": t.inputSchema}}
             for t in listed if t.name in TOOLS] + [CONVERT_DECL]
    TOOLS_JSON.write_text(json.dumps(decls, indent=1) + "\n")
    print(f"wrote {TOOLS_JSON.name}: {[d['function']['name'] for d in decls]}")


def _assert_defaults() -> None:
    sys.path.insert(0, str(MCP_DIR))
    import ro_model
    import swro_model
    assert ro_model.DEFAULTS == DEFAULTS[RO], "ro_model.DEFAULTS changed"
    assert swro_model.DEFAULTS == DEFAULTS[SWRO], "swro_model.DEFAULTS changed"


def cmd_generate(args) -> None:
    import multiprocessing as mp
    _assert_defaults()
    DATA.mkdir(exist_ok=True)
    splits = list(SPLITS) if args.split == "all" else [args.split]
    for split in splits:
        n, seed = SPLITS[split]
        n = args.limit or n
        t0 = time.time()
        with mp.Pool(args.workers) as pool:
            cases = list(pool.imap(make_case, [(split, seed, i) for i in range(n)], chunksize=2))
        path = DATA / f"{split}.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in cases))
        solves = sum(c["solves"] for c in cases)
        print(f"{split}: {len(cases)} cases, {solves} solves, {time.time() - t0:.0f}s -> {path.name}", flush=True)


def decision_label(case: dict) -> str:
    """The gold decision as a class: pass/fail, a label, or a scan position like "2/4"."""
    d = case["gold"]["decision"]
    if d is None:
        return "none"
    if case["decision_kind"] == "number":
        users = case["scan"]["users"]
        return f"{users.index(d) + 1}/{len(users)}"
    return str(d)


def load_split(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]


def cmd_check(args) -> None:
    report: dict[str, Any] = {}
    core = AGENT_CORE.read_text()
    m = re.search(r"SYSTEM_MESSAGE = \((.*?)\n\)", core, re.S)
    deployed = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))) if m else None
    report["system_message_matches_agent"] = deployed == DEPLOYED_SYSTEM_MESSAGE
    splits = {s: load_split(s) for s in SPLITS if (DATA / f"{s}.jsonl").exists()}
    prompts = Counter(c["prompt"] for cs in splits.values() for c in cs)
    report["duplicate_prompts"] = sum(v - 1 for v in prompts.values() if v > 1)
    # Scenario overlap between splits (same tool arguments for the first point).
    keyset = {s: {json.dumps(c["points"][0]["args"], sort_keys=True) for c in cs} for s, cs in splits.items()}
    report["cross_split_scenarios"] = {f"{a}&{b}": len(keyset[a] & keyset[b])
                                       for a in keyset for b in keyset if a < b}
    for split, cases in splits.items():
        fam = defaultdict(list)
        for c in cases:
            fam[c["family"]].append(c)
        rows = {}
        for f, cs in sorted(fam.items()):
            b = [baselines(c) for c in cs]
            dec = Counter(decision_label(c) for c in cs)
            rows[f] = {"n": len(cs), "levels": dict(Counter(c["level"] for c in cs)),
                       "optimal_calls": dict(Counter(c["optimal_calls"] for c in cs)),
                       "decisions": dict(dec.most_common()),
                       "baselines": {k: round(sum(x[k] for x in b) / len(b), 4) for k in b[0]}}
        report[split] = rows
    if DATA_04.exists():  # which questions are 04's, byte for byte (only the LMH/bar cases should differ)
        same = {}
        for split, cases in splits.items():
            old = {c["id"]: c for c in map(json.loads, (DATA_04 / f"{split}.jsonl").read_text().splitlines())}
            same[split] = {"identical_prompt": sum(c["prompt"] == old.get(c["id"], {}).get("prompt") for c in cases),
                           "n": len(cases),
                           "lmh_bar_cases": sum("LMH/bar" in c["prompt"] for c in cases)}
        report["identical_to_04"] = same
    if args.decontam:
        report["decontamination"] = decontamination([c for cs in splits.values() for c in cs])
    out = HERE / f"{PREFIX}_checks.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def cmd_elasticity(args) -> None:
    """Worst relative output change when one stated input is off by +-ARG_TOL (justifies MARGIN)."""
    host = _host()
    worst: dict[str, dict[str, dict[str, float]]] = {RO: defaultdict(dict), SWRO: defaultdict(dict)}
    scenarios = Counter()
    rng = random.Random(7)
    while min(scenarios[RO], scenarios[SWRO]) < args.n:
        tool = RO if scenarios[RO] <= scenarios[SWRO] else SWRO
        g = Gen(rng, "dev", 0, host)
        stated = g.ro_base()[0] if tool == RO else swro_base(g)[0]
        try:
            base = g.solve(tool, stated)["metrics"]
        except Retry:
            continue
        scenarios[tool] += 1
        for key, v in stated.items():
            if not isinstance(v, float):
                continue
            for sign in (1, -1):
                try:
                    m = g.solve(tool, {**stated, key: v * (1 + sign * ARG_TOL)})["metrics"]
                except Retry:
                    continue
                for path, b in base.items():
                    d = abs(m[path] / b - 1)
                    worst[tool][key][path] = max(worst[tool][key].get(path, 0.0), d)
    out = {"arg_tol": ARG_TOL, "scenarios": dict(scenarios),
           "worst_by_input": {t: {k: dict(v) for k, v in w.items()} for t, w in worst.items()},
           "worst_by_metric": {t: {p: max(v.get(p, 0.0) for v in w.values()) for p in METRICS[t]}
                               for t, w in worst.items()},
           "margin": {p: MARGIN[p] for t in TOOLS for p in METRICS[t]}}
    (HERE / f"{PREFIX}_elasticity.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["worst_by_metric"], indent=1))



def cmd_check_harness(args) -> None:
    """CPU checks of the rollout plumbing on the blind reference episodes of a split: the calls parse
    back to what was written, the pool reproduces the gold results, and the episode built by token
    concatenation (convert turn, simulate turn, answer) is the chat template's own rendering."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    prefix = generation_prefix(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cases = load_split(args.split)[: args.n]
    pool = ToolPool(args.workers)
    same_tokens, roundtrip, totals, identical, n_turns, n_conv = 0, 0, [], 0, Counter(), 0
    try:
        for c in cases:
            ref = reference_episode(c, pool)
            n_turns[len(ref["turns"])] += 1
            n_conv += ref["converts"]
            answer = ref["answer_reasoned"]
            by_key = {p["key"]: p for p in c["points"]}
            exact = pool.call_many([(c["tool"], by_key[k]["args"]) for k in c["proof"]])
            identical += sum(r == by_key[k]["result"] for r, k in zip(exact, c["proof"]))
            ids = tok(render_prompt(tok, c), add_special_tokens=False).input_ids
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": c["prompt"]}]
            ok = True
            for turn in ref["turns"]:
                calls_text = "\n".join(call_block(x["name"], x["arguments"]) for x in turn["calls"])
                parsed = parse_calls(calls_text)
                ok &= [(p["name"], p["arguments"]) for p in parsed] == [(x["name"], x["arguments"]) for x in turn["calls"]]
                ids += tok(calls_text, add_special_tokens=False).input_ids + [im_end]
                ids += tok(tool_response_text(turn["results"], prefix), add_special_tokens=False).input_ids
                msgs.append({"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": p["name"], "arguments": p["arguments"]}} for p in parsed]})
                msgs += [{"role": "tool", "content": r} for r in turn["results"]]
            roundtrip += ok
            ids += tok(answer, add_special_tokens=False).input_ids + [im_end]
            msgs.append({"role": "assistant", "content": answer})
            text = tok.apply_chat_template(msgs, tools=_tool_decls(), tokenize=False, **TEMPLATE_KWARGS)
            templ = tok(text.rstrip("\n"), add_special_tokens=False).input_ids
            same_tokens += ids == templ
            totals.append(ref["reward"])
    finally:
        pool.close()
    n = len(cases)
    out = {"split": args.split, "n": n, "calls_parse_roundtrip": roundtrip, "token_ids_equal_template": same_tokens,
           "pool_results_identical_to_gold_exact_args": identical, "pool_calls": sum(len(c["proof"]) for c in cases),
           "episodes_by_call_turns": dict(n_turns), "convert_calls": n_conv,
           "reference_score_through_pool_min": min(totals), "reference_score_through_pool_mean": sum(totals) / n}
    (HERE / f"{PREFIX}_harness_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def cmd_probe(args) -> None:
    """Run a policy on a split with the simulators in the loop; write summary + episodes."""
    import torch
    if args.max_vram_mib and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_mib * 2**20 / total))
    torch.manual_seed(args.seed)
    policy, tok = load_policy(args.adapter)
    cases = load_split(args.split)
    if args.limit:
        rng = random.Random(0)
        cases = rng.sample(cases, args.limit)
    cases = [c for c in cases for _ in range(args.samples)]
    out_dir = RUNS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.split}-T{args.temperature:g}" + (f"-x{args.samples}" if args.samples > 1 else "")
           + ("-stop" if args.stop_at_json else ""))
    pool = ToolPool(args.workers)
    episodes, scores, t0 = [], [], time.time()
    try:
        with (out_dir / f"{tag}.episodes.jsonl").open("w") as fh:
            for b in range(0, len(cases), args.batch):
                eps = rollout(policy, tok, cases[b: b + args.batch], pool, temperature=args.temperature,
                              max_new_tokens=args.max_new_tokens, stop_at_json=bool(args.stop_at_json))
                for ep in eps:
                    sc = score_episode(ep)
                    episodes.append(ep)
                    scores.append(sc)
                    fh.write(json.dumps(ep.to_json() | {"score": {k: sc[k] for k in ("total", "components", "evidence", "decision_correct")}}) + "\n")
                done = b + len(eps)
                print(f"{done}/{len(cases)}  reward so far {sum(s['total'] for s in scores) / done:.3f}  "
                      f"{time.time() - t0:.0f}s  solves {pool.solved}", flush=True)
    finally:
        pool.close()
    summary = summarise(episodes, scores)
    summary["meta"] = {"adapter": args.adapter, "split": args.split, "temperature": args.temperature,
                       "samples": args.samples, "max_new_tokens": args.max_new_tokens,
                       "stop_at_json": bool(args.stop_at_json), "seconds": time.time() - t0, "solves": pool.solved,
                       "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None}
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["overall"], indent=1))



def cmd_reference(args) -> None:
    """Blind reference episodes for a split (simulated through the pool), verified by the reward."""
    pool = ToolPool(args.workers)
    try:
        refs = build_references(args.split, pool)
    finally:
        pool.close()
    path = DATA / f"{args.split}_reference.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in refs))
    rewards = [r["reward"] for r in refs]
    conv = [r["converts"] for r in refs]
    print(f"{path.name}: {len(refs)} episodes, {pool.solved} solves, reward min {min(rewards):.4f} "
          f"mean {sum(rewards) / len(rewards):.4f}; convert calls mean {sum(conv) / len(conv):.2f} max {max(conv)}, "
          f"episodes without any {sum(c == 0 for c in conv)}")


def cmd_seed(args) -> None:
    import torch
    if args.max_vram_mib and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_mib * 2**20 / total))
    train_seed(RUNS / args.out, label=args.label, reasoning=bool(args.reasoning), epochs=args.epochs, lr=args.lr,
               accum=args.accum, lora_r=args.lora_r, seed=args.seed, limit=args.limit,
               label_per_class=args.label_per_class)



def cmd_ppo(args) -> None:
    import torch
    if args.max_vram_mib and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_mib * 2**20 / total))
    cfg = dict(PPO_DEFAULTS)
    for k, v in vars(args).items():
        if k in cfg and v is not None:
            cfg[k] = type(PPO_DEFAULTS[k])(v) if PPO_DEFAULTS[k] is not None else v
    if cfg["value_init_bias"] < 0 or not cfg["start_from"]:
        raise SystemExit("pass --start-from and --value-init-bias (the seed's own sampled mean reward)")
    train_ppo(cfg, RUNS / args.out, progress=lambda m: print(m, flush=True), resume=bool(args.resume))


def cmd_worker(_args) -> None:
    host = _host()
    for line in sys.stdin:
        req = json.loads(line)
        sys.stdout.write(json.dumps({"id": req.get("id"), "result": host.call(req["name"], req.get("arguments"))}) + "\n")
        sys.stdout.flush()


def cmd_xcheck(args) -> None:
    """Re-solve a sample of gold points through the deployed MCP (respects its 30/min limit)."""
    import asyncio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    env = dict(line.split("=", 1) for line in (REPO / ".env").read_text().splitlines()
               if "=" in line and not line.startswith("#"))
    url, token = env["RO_MCP_URL"].strip(), env["MCP_BEARER_TOKEN"].strip()
    cases = load_split(args.split)
    rng = random.Random(0)
    sample = [(c, rng.choice(c["points"])) for c in rng.sample(cases, args.n)]

    async def run():
        rows = []
        async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                for c, p in sample:
                    res = await s.call_tool(c["tool"], p["args"])
                    rows.append((c["id"], p["key"], res.content[0].text == p["result"]))
                    await asyncio.sleep(2.5)
        return rows

    rows = asyncio.run(run())
    for row in rows:
        print(*row)
    print(f"identical result text: {sum(r[2] for r in rows)}/{len(rows)}")
    out = {"endpoint": url, "split": args.split, "identical": sum(r[2] for r in rows), "n": len(rows),
           "rows": [{"id": i, "point": k, "identical": same} for i, k, same in rows]}
    (HERE / f"{PREFIX}_xcheck.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump-tools").set_defaults(fn=cmd_dump_tools)
    p = sub.add_parser("generate")
    p.add_argument("--split", default="all", choices=["all", *SPLITS])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cases per split (0 = the full size)")
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("check")
    p.add_argument("--decontam", type=int, default=1)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("elasticity")
    p.add_argument("--n", type=int, default=12, help="scenarios per tool")
    p.set_defaults(fn=cmd_elasticity)
    p = sub.add_parser("check-harness")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_check_harness)
    p = sub.add_parser("probe")
    p.add_argument("--split", default="dev")
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", default="raw")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--stop-at-json", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-vram-mib", type=int, default=13000)
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("reference")
    p.add_argument("--split", default="train")
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(fn=cmd_reference)
    p = sub.add_parser("seed")
    p.add_argument("--out", required=True)
    p.add_argument("--label", choices=["full", "scaffold"], required=True)
    p.add_argument("--reasoning", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--label-per-class", type=int, default=0,
                   help="label the decision on this many train records per (family, answer class)")
    p.add_argument("--max-vram-mib", type=int, default=13000)
    p.set_defaults(fn=cmd_seed)
    p = sub.add_parser("ppo")
    p.add_argument("--out", required=True)
    p.add_argument("--max-vram-mib", type=int, default=13000)
    p.add_argument("--resume", type=int, default=0, help="1: continue from <out>/last/")
    for k, v in PPO_DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v), default=None)
    p.set_defaults(fn=cmd_ppo)
    sub.add_parser("worker").set_defaults(fn=cmd_worker)
    p = sub.add_parser("xcheck")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=12)
    p.set_defaults(fn=cmd_xcheck)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
