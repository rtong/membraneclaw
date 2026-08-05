"""Validate the full-plant wrapper against WaterTAP's own flowsheet.

Like test_ro_model.py, the reference numbers are upstream's rather than ours: they
are what `seawater_RO_desalination.main()` produces for each ERD configuration, so
this checks that the wrapper reproduces the flowsheet instead of merely running.
They were captured on WaterTAP 1.7.0 and matched to zero delta.

Most checks build the flowsheet but never solve it, which keeps the default run to
a few seconds. The full solve is behind SWRO_SLOW_TEST=1.

Run:      ~/reaktoro-mcp/env/bin/python mcp_watertap/test_swro_model.py
Full run: SWRO_SLOW_TEST=1 ~/reaktoro-mcp/env/bin/python mcp_watertap/test_swro_model.py
"""
from __future__ import annotations

import os
import sys

from pyomo.environ import value
from idaes.core.util.model_statistics import degrees_of_freedom

import swro_model as sm
from swro_model import AREA_PER_FLOW, DEFAULTS, SWROSimulationError, simulate_swro

# From seawater_RO_desalination.main(), WaterTAP 1.7.0.
REFERENCE = {
    "pressure_exchanger": {"LCOW_usd_m3": 0.8291891, "specific_energy_kWh_m3": 3.0628963},
    "pump_as_turbine": {"LCOW_usd_m3": 1.0997000, "specific_energy_kWh_m3": 5.9579820},
}
RTOL = 1e-4


def check(label: str, ok: bool) -> bool:
    print(f"  {label:58s} {'ok' if ok else 'MISMATCH'}")
    return ok


def raises(fn, label: str) -> bool:
    try:
        fn()
    except SWROSimulationError:
        return check(label, True)
    except Exception as exc:
        return check(f"{label} (wrong type: {type(exc).__name__})", False)
    return check(f"{label} (did not raise)", False)


def test_validation() -> int:
    print("1. Bad inputs are rejected, not silently coerced")
    failures = 0
    cases = {
        "unknown parameter": lambda: simulate_swro(nonsense=1),
        "bad erd_type": lambda: simulate_swro(erd_type="turbo"),
        "efficiency above 1": lambda: simulate_swro(p1_efficiency=1.5),
        "efficiency of zero": lambda: simulate_swro(p1_efficiency=0),
        "negative feed flow": lambda: simulate_swro(feed_flow_m3_s=-1),
        "zero TDS": lambda: simulate_swro(feed_tds_g_L=0),
        "negative area": lambda: simulate_swro(ro_area_m2=-5),
        "sub-atmospheric pump pressure": lambda: simulate_swro(p1_pressure_bar=0.5),
    }
    for label, fn in cases.items():
        failures += 0 if raises(fn, label) else 1
    return failures


def test_erd_mismatch() -> int:
    print("\n2. Efficiencies for the other ERD type are refused, not ignored")
    failures = 0
    # Silently ignoring these would hand back a result that looks like it honoured
    # the argument while actually using the default.
    failures += 0 if raises(
        lambda: simulate_swro(erd_type="pump_as_turbine", pxr_efficiency=0.9),
        "pxr_efficiency with pump_as_turbine",
    ) else 1
    failures += 0 if raises(
        lambda: simulate_swro(erd_type="pump_as_turbine", p2_efficiency=0.9),
        "p2_efficiency with pump_as_turbine",
    ) else 1
    failures += 0 if raises(
        lambda: simulate_swro(erd_type="pressure_exchanger", erd_efficiency=0.9),
        "erd_efficiency with pressure_exchanger",
    ) else 1
    return failures


def test_area_tracks_flow() -> int:
    print("\n3. Membrane area follows feed flow unless given explicitly")
    failures = 0
    # The flowsheet sizes area as flow_vol * 4.5e4. Overriding flow without
    # rescaling area would quietly simulate a plant sized for a different feed.
    default_area = DEFAULTS["feed_flow_m3_s"] * AREA_PER_FLOW
    failures += 0 if check(
        f"default area is the documented 13914 m2 (got {default_area:.0f})",
        abs(default_area - 13914) < 1,
    ) else 1

    p = {**DEFAULTS, "feed_flow_m3_s": 0.6184, "ro_area_m2": None}
    derived = p["feed_flow_m3_s"] * AREA_PER_FLOW
    failures += 0 if check(
        "doubling the feed doubles the derived area", abs(derived - 2 * default_area) < 1
    ) else 1
    return failures


def test_bounded_solves_restores() -> int:
    print("\n4. Solver limits are injected and always restored")
    failures = 0
    original = sm.swro.get_solver

    with sm._bounded_solves():
        patched = sm.swro.get_solver
        failures += 0 if check("get_solver is patched inside", patched is not original) else 1
        solver = sm.swro.get_solver()
        opts = dict(solver.options)
        failures += 0 if check(
            "max_iter limit applied", opts.get("max_iter") == sm.MAX_SOLVE_ITERATIONS
        ) else 1
        failures += 0 if check(
            "max_cpu_time limit applied", opts.get("max_cpu_time") == sm.MAX_SOLVE_SECONDS
        ) else 1
    failures += 0 if check("restored after the block", sm.swro.get_solver is original) else 1

    # An exception must not leave the module permanently patched.
    try:
        with sm._bounded_solves():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    failures += 0 if check("restored after an exception", sm.swro.get_solver is original) else 1
    return failures


def _built(erd_type, **overrides):
    p = {**DEFAULTS, **overrides, "erd_type": erd_type}
    if p["ro_area_m2"] is None:
        p["ro_area_m2"] = p["feed_flow_m3_s"] * AREA_PER_FLOW
    with sm.stdout_to_stderr(), sm._bounded_solves():
        m = sm.swro.build(erd_type=erd_type)
        sm.swro.set_operating_conditions(m)
        sm._apply_overrides(m, p)
    return m, p


def test_dof_stays_zero() -> int:
    print("\n5. Applying parameters leaves the model fully specified")
    failures = 0
    for erd in ("pressure_exchanger", "pump_as_turbine"):
        m, _ = _built(erd)
        dof = degrees_of_freedom(m)
        failures += 0 if check(f"{erd}: DOF == 0 after overrides (got {dof})", dof == 0) else 1
    return failures


def _num(var):
    """Read a Pyomo var that may be scalar or indexed.

    A_comp, B_comp and efficiency_pump are IndexedVar (by time, and by solute for
    the permeabilities). `.fix(x)` sets every member, so reading any one of them
    confirms the override landed.
    """
    # Ask rather than catch: value() on an IndexedVar logs a Pyomo ERROR before it
    # raises, which would bury the test output in noise for a non-problem.
    if getattr(var, "is_indexed", None) and var.is_indexed():
        return value(next(iter(var.values())))
    return value(var)


def test_overrides_take_effect() -> int:
    print("\n6. Parameters actually reach the model (not silently dropped)")
    failures = 0
    m, _ = _built(
        "pressure_exchanger",
        feed_flow_m3_s=0.4,
        feed_tds_g_L=42.0,
        p1_pressure_bar=65.0,
        p1_efficiency=0.75,
        ro_area_m2=12000.0,
        A_comp=4.0e-12,
    )
    desal = m.fs.desalination
    checks = {
        "feed flow": (_num(m.fs.feed.flow_vol[0]), 0.4),
        "feed TDS": (_num(m.fs.feed.conc_mass_comp[0, "tds"]), 42.0),
        "RO area": (_num(desal.RO.area), 12000.0),
        "A_comp": (_num(desal.RO.A_comp), 4.0e-12),
        "P1 pressure (Pa)": (
            _num(desal.P1.control_volume.properties_out[0].pressure), 65e5,
        ),
        "P1 efficiency": (_num(desal.P1.efficiency_pump), 0.75),
    }
    for label, (got, want) in checks.items():
        failures += 0 if check(
            f"{label} = {want:g}", abs(got - want) <= RTOL * abs(want)
        ) else 1
    return failures


def test_full_solve() -> int:
    print("\n7. Full solve reproduces WaterTAP's own reference results")
    if not os.environ.get("SWRO_SLOW_TEST"):
        print("  skipped (set SWRO_SLOW_TEST=1 to run the real solves)")
        return 0
    failures = 0
    for erd, expected in REFERENCE.items():
        out = simulate_swro(erd_type=erd)
        for key, want in expected.items():
            got = out["costing"][key]
            failures += 0 if check(
                f"{erd}: {key} = {want:.6f} (got {got:.6f})",
                abs(got - want) <= RTOL * abs(want),
            ) else 1
        rec = out["performance"]["ro_recovery_pct"]
        failures += 0 if check(f"{erd}: RO recovery is plausible ({rec:.1f}%)", 20 < rec < 70) else 1
    return failures


def main() -> int:
    failures = 0
    failures += test_validation()
    failures += test_erd_mismatch()
    failures += test_area_tracks_flow()
    failures += test_bounded_solves_restores()
    failures += test_dof_stays_zero()
    failures += test_overrides_take_effect()
    failures += test_full_solve()
    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
