"""Check convert_units and check_limits against hand-checked values, and the server's declarations.

The expected outputs are the smoke_test35 notebook implementation's (the code here is a verbatim copy of it; a
20,000-case comparison against the notebook module found no difference when it was moved), and the conversions
among them are the textbook factors: 1 bar = 14.5038 psi, 25 C = 77 F, 1 MGD = 0.0438126 m3/s, 1 $/m3 =
3.78541 $/kgal, 8 gfd = 13.5819 LMH.

Run: <python with mcp installed> mcp_rocalc/test_rocalc.py   (e.g. .venv-watertap/bin/python)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from calc_limits import check_limits  # noqa: E402
from calc_units import convert_units  # noqa: E402

CONVERSIONS = [
    ((1, "bar", "psi"), {"value": 14.5038, "unit": "psi"}),
    ((35, "g/kg", "mass fraction"), {"value": 0.035, "unit": "mass fraction"}),
    ((25, "degC", "F"), {"value": 77.0, "unit": "F"}),
    ((0.5, "LMH/bar", "m/s/Pa"), {"value": 1.38889e-12, "unit": "m/s/Pa"}),
    ((10, "m³/h", "kg/s"), {"value": 2.77778, "unit": "kg/s"}),   # water at 1000 kg/m3
    ((1, "MGD", "m3/s"), {"value": 0.0438126, "unit": "m3/s"}),
    ((1, "$/m3", "$/kgal"), {"value": 3.78541, "unit": "$/kgal"}),
    ((1000, "ppm", "g/kg"), {"value": 1.0, "unit": "g/kg"}),      # ppm as a salinity
    ((500, "ppm", "mg/L"), {"value": 500.0, "unit": "mg/L"}),     # ppm as a concentration
    ((3.2, "bar", "bar"), {"value": 3.2, "unit": "bar"}),
    ((55, "%", "fraction"), {"value": 0.55, "unit": "fraction"}),
    (("12.5", "kPa", "bar"), {"value": 0.125, "unit": "bar"}),
]
CONVERSION_ERRORS = [
    ((100, "kg/s", "bar"), "cannot convert 'kg/s' to 'bar'"),
    (("abc", "bar", "psi"), "value must be a number"),
]
LIMITS = [
    ({"permeate.nacl_ppm": "<= 500 mg/L"},
     {"Option A": {"permeate.nacl_ppm": 320}, "Option B": {"permeate.nacl_ppm": 612}},
     ["Option A: permeate.nacl_ppm 320 meets <= 500 -> pass", "Option B: permeate.nacl_ppm 612 breaks <= 500 -> fail"]),
    ({"permeate.nacl_ppm": "<= 0.5 g/L", "flux.water_LMH": ">= 15 LMH"},
     {"Result": {"permeate.nacl_ppm": 480, "flux.water_LMH": 14.2}},
     ["Result: permeate.nacl_ppm 480 meets <= 500; flux.water_LMH 14.2 breaks >= 15 -> fail"]),
    ({"flux.water_LMH": ">= 8 gfd"}, {"Result": {"flux.water_LMH": 14}},
     ["Result: flux.water_LMH 14 meets >= 13.5819 -> pass"]),
    ({"costing.LCOW_usd_m3": "<= 0.9 $/m3"}, {"Result": {"permeate.nacl_ppm": 200}},
     ["Result: costing.LCOW_usd_m3 ? (no value); permeate.nacl_ppm 200 -> unchecked"]),
    ({"unknown.field": "<= 3"}, {"Result": {"unknown.field": 2}},
     ["Result: unknown.field 2 (unknown field) -> unchecked"]),
    ({"permeate.nacl_ppm": "about 500"}, {"Result": {"permeate.nacl_ppm": 400}},
     ["Result: permeate.nacl_ppm 400 (limit 'about 500' not read) -> unchecked"]),
    ({"permeate.nacl_ppm": "<= 500 bar"}, {"Result": {"permeate.nacl_ppm": 400}},
     ["Result: permeate.nacl_ppm 400 (limit '<= 500 bar' not in mg/L) -> unchecked"]),
]
LIMIT_ERRORS = [
    ({"permeate.nacl_ppm": "<= 500"}, ["not", "an", "object"], "values must be an object"),
    ({"permeate.nacl_ppm": "<= 500"}, {"Result": 400}, "the values of 'Result' must be an object"),
]

# What the notebooks declare to the policy (CONVERT_DECL / COMPARE_DECL parameters): the server must declare the
# same, so a model trained there sees the same tools here.
DECLARED_SCHEMAS = {
    "convert_units": {"properties": {"value": {"title": "Value", "type": "number"},
                                     "from_unit": {"title": "From Unit", "type": "string"},
                                     "to_unit": {"title": "To Unit", "type": "string"}},
                      "required": ["value", "from_unit", "to_unit"], "title": "convert_unitsArguments",
                      "type": "object"},
    "check_limits": {"properties": {"limits": {"title": "Limits", "type": "object",
                                               "additionalProperties": {"type": "string"}},
                                    "values": {"title": "Values", "type": "object",
                                               "additionalProperties": {"type": "object",
                                                                        "additionalProperties": {"type": "number"}}}},
                     "required": ["limits", "values"], "title": "check_limitsArguments", "type": "object"},
}


def test_conversions() -> int:
    failures = 0
    for args, expected in CONVERSIONS:
        got = json.loads(convert_units(*args))
        ok = got == expected
        failures += not ok
        print(f"  convert_units{args!r:42} -> {got}{'' if ok else f'  EXPECTED {expected}'}")
    for args, needle in CONVERSION_ERRORS:
        got = json.loads(convert_units(*args))
        ok = needle in got.get("error", "")
        failures += not ok
        print(f"  convert_units{args!r:42} -> error{'' if ok else f'  EXPECTED an error with {needle!r}, got {got}'}")
    return failures


def test_limits() -> int:
    failures = 0
    for limits, values, expected in LIMITS:
        got = json.loads(check_limits(limits, values))
        ok = got == expected
        failures += not ok
        print(f"  check_limits {json.dumps(limits)} -> {got}{'' if ok else f'  EXPECTED {expected}'}")
    for limits, values, needle in LIMIT_ERRORS:
        got = json.loads(check_limits(limits, values))
        ok = isinstance(got, dict) and needle in got.get("error", "")
        failures += not ok
        print(f"  check_limits {json.dumps(values)} -> error{'' if ok else f'  EXPECTED an error with {needle!r}, got {got}'}")
    return failures


def test_server() -> int:
    """The server over stdio, as a client starts it: the declared tools, and one call of each."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    import server  # the descriptions it declares

    async def run() -> int:
        failures = 0
        params = StdioServerParameters(command=sys.executable, args=[str(HERE / "server.py")])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                ok = sorted(tools) == ["check_limits", "convert_units"]
                failures += not ok
                print(f"  tools {sorted(tools)}{'' if ok else '  EXPECTED check_limits, convert_units'}")
                for name, desc in (("convert_units", server.CONVERT_DESCRIPTION),
                                   ("check_limits", server.CHECK_LIMITS_DESCRIPTION)):
                    ok_s = tools[name].inputSchema == DECLARED_SCHEMAS[name]
                    ok_d = tools[name].description == desc
                    failures += (not ok_s) + (not ok_d)
                    print(f"  {name}: schema {'as declared' if ok_s else 'DIFFERS: ' + json.dumps(tools[name].inputSchema)}, "
                          f"description {'as declared' if ok_d else 'DIFFERS'}")
                got = json.loads((await session.call_tool("convert_units", {"value": 35, "from_unit": "g/kg",
                                                                            "to_unit": "mass fraction"})).content[0].text)
                ok = got == {"value": 0.035, "unit": "mass fraction"}
                failures += not ok
                print(f"  call convert_units -> {got}")
                got = json.loads((await session.call_tool("check_limits", {
                    "limits": {"permeate.nacl_ppm": "<= 500 mg/L"},
                    "values": {"Result": {"permeate.nacl_ppm": 320}}})).content[0].text)
                ok = got == ["Result: permeate.nacl_ppm 320 meets <= 500 -> pass"]
                failures += not ok
                print(f"  call check_limits -> {got}")
        return failures

    return asyncio.run(run())


def main() -> int:
    failures = 0
    for name, test in (("conversions", test_conversions), ("limits", test_limits), ("server (stdio)", test_server)):
        print(name)
        failures += test()
    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
