"""Full seawater RO plant simulation, wrapped for tool use.

Where `ro_model.py` solves a single bare ReverseOsmosis0D unit, this wraps
WaterTAP's `seawater_RO_desalination` flowsheet: a whole facility — pretreatment
(intake, ferric chloride, chlorination, static mixer, storage, media and cartridge
filtration), desalination (P1 -> RO -> pressure exchanger + P2, or an
energy-recovery turbine), post-treatment (anti-scalant, lime, UV/AOP, storage),
and municipal/landfill product handling — costed with ZeroOrderCosting.

That costing is the point: only after `cost_process()` do LCOW and specific energy
consumption exist, and those are the two numbers `simulate_ro` structurally cannot
produce.

Reference: https://watertap.readthedocs.io/en/stable/technical_reference/flowsheets/seawater_RO_desalination.html

Two things the reference page gets wrong about the current release, both verified
against the installed package: the module path it cites
(`watertap.examples.flowsheets.case_studies...`) has moved, and the convenience
helpers it documents (`build_flowsheet`, `solve_flowsheet`) do not exist. So this
module replicates the sequence in the flowsheet's own `main()` instead.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stdio_guard import stdout_to_stderr

from pyomo.environ import value
from idaes.core.util.model_statistics import degrees_of_freedom

try:
    from watertap.flowsheets.seawater_RO_desalination import (
        seawater_RO_desalination as swro,
    )
except ImportError:  # pre-move layout, which the docs still describe
    from watertap.examples.flowsheets.case_studies import (  # type: ignore[no-redef]
        seawater_RO_desalination as swro,
    )

BAR = 1e5

# This train is far heavier than the single RO unit in ro_model.py — pretreatment,
# RO, post-treatment and costing, with a solve per block during sequential
# initialization — so it gets its own, larger ceiling rather than sharing
# MCP_MAX_SOLVE_SECONDS.
MAX_SOLVE_SECONDS = float(os.environ.get("MCP_MAX_SWRO_SOLVE_SECONDS", "600"))
MAX_SOLVE_ITERATIONS = int(os.environ.get("MCP_MAX_SWRO_ITERATIONS", "2000"))
_SOLVE_LIMITS = {
    "max_cpu_time": MAX_SOLVE_SECONDS,
    "max_iter": MAX_SOLVE_ITERATIONS,
}

ERD_TYPES = ("pressure_exchanger", "pump_as_turbine")

# Membrane area scales with feed flow in the flowsheet's own specification:
# `desal.RO.area.fix(flow_vol * 4.5e4 s/m)`. Overriding the flow without
# rescaling area would silently simulate a plant sized for a different feed.
AREA_PER_FLOW = 4.5e4

DEFAULTS: dict[str, Any] = {
    "erd_type": "pressure_exchanger",
    "feed_flow_m3_s": 0.3092,          # 7.05 MGD
    "feed_tds_g_L": 35.0,
    "feed_tss_g_L": 0.03,
    "feed_temperature_c": 24.85,       # = 298 K, the flowsheet's own default
    "ro_area_m2": None,                # None -> feed_flow_m3_s * AREA_PER_FLOW
    "A_comp": 4.2e-12,
    "B_comp": 3.5e-8,
    "p1_pressure_bar": 70.0,
    "p1_efficiency": 0.80,
    "pxr_efficiency": 0.95,            # pressure_exchanger builds only
    "p2_efficiency": 0.80,             # pressure_exchanger builds only
    "erd_efficiency": 0.95,            # pump_as_turbine builds only
}

_PXR_ONLY = ("pxr_efficiency", "p2_efficiency")
_TURBINE_ONLY = ("erd_efficiency",)


class SWROSimulationError(RuntimeError):
    """Raised when the flowsheet cannot be built, initialized, or solved."""


@contextlib.contextmanager
def _bounded_solves():
    """Cap every solve this flowsheet performs, not just the ones we call.

    `initialize_system()` and `set_operating_conditions()` both solve internally
    and neither accepts a solver or optarg, so there is no argument to pass limits
    through. They reach the solver via the module-level `get_solver`, so patching
    that name is the only seam that covers all of them.

    This matters more than it looks: OPERATIONS.md records that bounding only the
    final solve achieves nothing, because initialization does most of the work and
    the capped solve then converges immediately.
    """
    original = swro.get_solver

    def bounded(*args, **kwargs):
        solver = original(*args, **kwargs)
        solver.options.update(_SOLVE_LIMITS)
        return solver

    swro.get_solver = bounded
    try:
        yield
    finally:
        swro.get_solver = original


def _validate(p: dict[str, Any]) -> None:
    if p["erd_type"] not in ERD_TYPES:
        raise SWROSimulationError(f"erd_type must be one of {list(ERD_TYPES)}")

    for key in ("p1_efficiency", "p2_efficiency", "pxr_efficiency", "erd_efficiency"):
        eff = p[key]
        if eff is not None and not 0 < eff <= 1:
            raise SWROSimulationError(f"{key} must be in (0, 1]")

    if p["feed_flow_m3_s"] <= 0:
        raise SWROSimulationError("feed_flow_m3_s must be positive")
    if p["feed_tds_g_L"] <= 0:
        raise SWROSimulationError("feed_tds_g_L must be positive")
    if p["feed_tss_g_L"] < 0:
        raise SWROSimulationError("feed_tss_g_L cannot be negative")
    if p["ro_area_m2"] is not None and p["ro_area_m2"] <= 0:
        raise SWROSimulationError("ro_area_m2 must be positive")
    if p["A_comp"] <= 0 or p["B_comp"] <= 0:
        raise SWROSimulationError("A_comp and B_comp must be positive")
    if p["p1_pressure_bar"] <= 1:
        raise SWROSimulationError("p1_pressure_bar must exceed atmospheric")


def _reject_mismatched(overrides: dict[str, Any], erd_type: str) -> None:
    """Refuse efficiencies that belong to the other ERD configuration.

    Silently ignoring them would be worse than failing: the caller would get a
    result that looks like it honoured the argument, at the default value.
    """
    wrong = _TURBINE_ONLY if erd_type == "pressure_exchanger" else _PXR_ONLY
    supplied = [k for k in wrong if overrides.get(k) is not None]
    if supplied:
        raise SWROSimulationError(
            f"{sorted(supplied)} do not apply when erd_type={erd_type!r}"
        )


def _apply_overrides(m, p: dict[str, Any]) -> None:
    """Re-fix the flowsheet's hardcoded specification with our parameters.

    `set_operating_conditions(m)` takes no arguments — every value is fixed inside
    it — so the only way to parameterise the flowsheet is to fix the same variables
    again afterwards. Re-fixing a fixed variable replaces its value and leaves the
    degrees of freedom unchanged, which the caller re-checks.
    """
    desal = m.fs.desalination

    m.fs.feed.flow_vol[0].fix(p["feed_flow_m3_s"])
    m.fs.feed.conc_mass_comp[0, "tds"].fix(p["feed_tds_g_L"])
    m.fs.feed.conc_mass_comp[0, "tss"].fix(p["feed_tss_g_L"])
    # Temperature and pressure are fixed on the translator block between
    # pretreatment and desalination, not on the feed.
    m.fs.tb_prtrt_desal.properties_out[0].temperature.fix(273.15 + p["feed_temperature_c"])

    desal.RO.area.fix(p["ro_area_m2"])
    desal.RO.A_comp.fix(p["A_comp"])
    desal.RO.B_comp.fix(p["B_comp"])

    desal.P1.efficiency_pump.fix(p["p1_efficiency"])
    desal.P1.control_volume.properties_out[0].pressure.fix(p["p1_pressure_bar"] * BAR)

    if p["erd_type"] == "pressure_exchanger":
        desal.PXR.efficiency_pressure_exchanger.fix(p["pxr_efficiency"])
        desal.P2.efficiency_pump.fix(p["p2_efficiency"])
    else:
        desal.ERD.efficiency_pump.fix(p["erd_efficiency"])


def _touch_reported(m) -> None:
    """Build the properties we report before the final solve.

    Same trap as ro_model._touch_reported_properties: a property first accessed
    after the solve is constructed unconstrained at its default, so it returns a
    plausible wrong number rather than an error. Touching them here puts them in
    the system that the post-costing solve closes.
    """
    prod = m.fs.municipal.properties[0]
    prod.flow_vol
    with contextlib.suppress(Exception):
        # Not every ZO property package exposes concentration for every solute;
        # absence is fine, a silently-defaulted value would not be.
        prod.conc_mass_comp


def simulate_swro(**overrides: Any) -> dict[str, Any]:
    """Build, initialize, cost and solve the full seawater RO plant."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise SWROSimulationError(
            f"unknown parameter(s): {sorted(unknown)}; valid: {sorted(DEFAULTS)}"
        )
    supplied = {k: v for k, v in overrides.items() if v is not None}
    p = {**DEFAULTS, **supplied}

    _reject_mismatched(supplied, p["erd_type"])
    _validate(p)
    if p["ro_area_m2"] is None:
        p["ro_area_m2"] = p["feed_flow_m3_s"] * AREA_PER_FLOW

    with stdout_to_stderr(), _bounded_solves():
        try:
            m = swro.build(erd_type=p["erd_type"])
            swro.set_operating_conditions(m)
            _apply_overrides(m, p)
        except Exception as exc:
            raise SWROSimulationError(f"could not build the flowsheet: {exc}") from exc

        dof = degrees_of_freedom(m)
        if dof != 0:
            raise SWROSimulationError(
                f"model has {dof} degrees of freedom (needs 0) after applying the "
                "supplied parameters"
            )

        try:
            swro.initialize_system(m)
            swro.solve(m)
            swro.add_costing(m)
            if hasattr(swro, "initialize_costing"):
                swro.initialize_costing(m)      # 1.7: two packages to initialize
            else:
                m.fs.costing.initialize()       # single-package layout
            _touch_reported(m)
            # The flowsheet is solved twice on purpose, mirroring its own main():
            # costing adds variables and constraints after the first solve, so a
            # second is required to make LCOW and energy consumption consistent.
            swro.solve(m)
        except Exception as exc:
            raise SWROSimulationError(
                f"solve failed: {exc}. The requested operating point may be "
                f"infeasible, or may have hit the {MAX_SOLVE_SECONDS:g}s / "
                f"{MAX_SOLVE_ITERATIONS} iteration ceiling."
            ) from exc

        return _extract(m, p)


def _opt(fn):
    """Read a value that only exists in one ERD configuration."""
    try:
        return fn()
    except Exception:
        return None


def _costing_metrics(m) -> dict[str, Any]:
    """Plant-level cost metrics, across two incompatible flowsheet layouts.

    WaterTAP 1.7 (what is deployed here) runs *two* costing packages —
    `m.fs.zo_costing` for the zero-order units and `m.fs.ro_costing` for the RO
    train — and combines them into Expressions on the **model**: `m.LCOW`, `m.SEC`,
    `m.total_capital_cost`, `m.total_operating_cost`, all in USD_2018. Later
    versions collapse that into a single `m.fs.costing` exposing `.LCOW` and
    `.specific_energy_consumption`.

    Both are handled because the reference docs, the GitHub main branch and the
    installed package disagree with each other — reading the installed source is
    the only reliable answer, and that can change under a version bump.
    """
    if hasattr(m, "LCOW"):
        return {
            "LCOW_usd_m3": value(m.LCOW),
            "specific_energy_kWh_m3": value(m.SEC),
            "total_capital_cost_usd": value(m.total_capital_cost),
            "total_operating_cost_usd_year": value(m.total_operating_cost),
            "currency": "USD_2018",
        }
    costing = m.fs.costing
    return {
        "LCOW_usd_m3": value(costing.LCOW),
        "specific_energy_kWh_m3": value(costing.specific_energy_consumption),
        "total_capital_cost_usd": value(costing.total_capital_cost),
        "total_operating_cost_usd_year": value(costing.total_operating_cost),
        "currency": str(costing.base_currency),
    }


def _extract(m, p: dict[str, Any]) -> dict[str, Any]:
    desal = m.fs.desalination
    prod = m.fs.municipal.properties[0]

    product_flow = value(prod.flow_vol)
    feed_flow = p["feed_flow_m3_s"]

    out: dict[str, Any] = {
        "inputs": p,
        "costing": _costing_metrics(m),
        "performance": {
            "product_flow_m3_s": product_flow,
            "product_flow_MGD": product_flow * 22.824465,  # m3/s -> million gal/day
            "ro_recovery_pct": value(desal.RO.recovery_vol_phase[0, "Liq"]) * 100,
            # Plant recovery, not the RO unit's: pretreatment and post-treatment
            # losses mean the two differ, and this is the one that pays for water.
            "system_recovery_pct": product_flow / feed_flow * 100,
        },
        "desalination": {
            "erd_type": p["erd_type"],
            "p1_outlet_pressure_bar": value(
                desal.P1.control_volume.properties_out[0].pressure
            ) / BAR,
            "p1_power_kW": value(desal.P1.work_mechanical[0]) / 1e3,
        },
    }

    if p["erd_type"] == "pressure_exchanger":
        out["desalination"]["p2_outlet_pressure_bar"] = _opt(
            lambda: value(desal.P2.control_volume.properties_out[0].pressure) / BAR
        )
        out["desalination"]["p2_power_kW"] = _opt(
            lambda: value(desal.P2.work_mechanical[0]) / 1e3
        )
        out["desalination"]["pxr_efficiency"] = p["pxr_efficiency"]
    else:
        # work_mechanical is negative for a turbine; report the recovered
        # magnitude, which is what "energy recovery" means to a reader.
        out["desalination"]["erd_power_recovered_kW"] = _opt(
            lambda: -value(desal.ERD.work_mechanical[0]) / 1e3
        )
        out["desalination"]["erd_efficiency"] = p["erd_efficiency"]

    return out
