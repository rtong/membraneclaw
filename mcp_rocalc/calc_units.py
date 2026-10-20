"""convert_units: one number between the units an RO question states and the units the simulators take.

Exact factors, no third-party unit library: every unit the RO prompts use and the usual spellings of each.
Copied verbatim from the smoke_test35 notebooks (05_units_and_scans.py introduced it; the training harness has
served it to the policy since), so the deployed tool and the trained one are the same code.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

PSI_PER_BAR = 1e5 / 6894.757293168
FT2_PER_M2 = 1 / 0.09290304
MGD_PER_M3S = 86400 / 3785.411784
GFD_PER_LMH = 0.09290304 * 24 / 3.785411784


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
