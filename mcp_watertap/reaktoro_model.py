"""Reaktoro-PSE equilibrium chemistry, wrapped for tool use.

Each call builds and solves a fresh Pyomo model carrying a single ReaktoroBlock,
so calls are stateless and independent of each other. Composition is taken as
elemental molar flows; results report mineral scaling tendencies alongside bulk
solution properties.

Two choices here are deliberate and load-bearing:

* Activity models are pinned to Pitzer. reaktoro-pse defaults *every* phase to an
  ideal activity model, which for brine returns confident, wrong numbers rather
  than an error — the same failure shape as an unconstrained property variable.
* The solver runs with a CPU-time ceiling. A graybox NLP has no natural bound on
  iteration cost, and this is called from a request handler.

Reference: https://github.com/watertap-org/reaktoro-pse
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stdio_guard import stdout_to_stderr

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Var,
    check_optimal_termination,
    units as pyunits,
    value,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
import idaes.core.util.scaling as iscale

from reaktoro_pse.reaktoro_block import ReaktoroBlock
from reaktoro_pse.core.util_classes.cyipopt_solver import get_cyipopt_watertap_solver

BAR = 1e5

# Shared ceiling with ro_model, for the same reason: this endpoint is public and
# a graybox NLP has no natural bound on iteration cost.
MAX_SOLVE_SECONDS = float(os.environ.get("MCP_MAX_SOLVE_SECONDS", "120"))

# Seawater at ~34.7 g/kg, as elemental molar flows against 55.5 mol/s of water
# (1 kg/s) — roughly ro_model's default feed, but not identical to it.
#
# The two models do not agree on osmotic pressure for "the same" seawater: 28.53
# bar from ro_model, 24.50 here. That is expected, not a defect. ro_model's
# property package treats all dissolved solids as NaCl, and a mole of NaCl
# contributes more osmoles per gram than the Mg/Ca/SO4 salts that a real seawater
# analysis contains, some of which additionally ion-pair. Quote the two numbers
# against their own tool; do not carry one across.
DEFAULT_COMPOSITION: dict[str, float] = {
    "H2O": 55.5,
    "Na": 0.4690,
    "Cl": 0.5460,
    "Mg": 0.0528,
    "SO4": 0.0282,
    "Ca": 0.0103,
    "HCO3": 0.0024,
}

# Calcite and gypsum are the two scales that actually govern RO recovery limits.
DEFAULT_MINERALS = ("Calcite", "Gypsum")

# g/mol, for converting a molar composition to the mass fraction ro_model wants.
# Only what a water analysis actually reports; an unlisted key raises rather than
# being silently dropped, which would understate salinity and overstate recovery.
MOLAR_MASS: dict[str, float] = {
    "H2O": 18.015,
    "Na": 22.990,
    "K": 39.098,
    "Ca": 40.078,
    "Mg": 24.305,
    "Ba": 137.327,
    "Sr": 87.620,
    "Li": 6.941,
    "Cl": 35.453,
    "Br": 79.904,
    "F": 18.998,
    "SO4": 96.060,
    "HCO3": 61.016,
    "CO3": 60.008,
    "NO3": 62.004,
    "SiO2": 60.083,
}


def composition_salinity(composition: dict[str, float]) -> dict[str, float]:
    """Total dissolved solids for a molar composition, as g/kg and mass fraction.

    Lets the RO model and the chemistry model be driven from one description of
    the feed, rather than a salinity and a composition that quietly disagree.
    """
    unknown = sorted(set(composition) - set(MOLAR_MASS))
    if unknown:
        raise ReaktoroSimulationError(
            f"no molar mass known for {unknown}; supported: {sorted(MOLAR_MASS)}"
        )
    water_g = composition["H2O"] * MOLAR_MASS["H2O"]
    solids_g = sum(
        n * MOLAR_MASS[k] for k, n in composition.items() if k != "H2O"
    )
    if water_g <= 0:
        raise ReaktoroSimulationError("composition['H2O'] must be positive")
    return {
        "tds_g_per_kg_water": solids_g / water_g * 1000.0,
        "mass_fraction": solids_g / (solids_g + water_g),
    }

DEFAULTS: dict[str, Any] = {
    "composition": DEFAULT_COMPOSITION,
    "temperature_c": 25.0,
    "pressure_bar": 1.0,
    "ph": 7.0,
    # Redox potential. Left unset by default: fixing it constrains redox
    # speciation, which only matters for waters carrying a redox-active species.
    "pe": None,
    "water_recovery": None,
    "acid_addition_mol_s": 0.0,
    "base_addition_mol_s": 0.0,
    "minerals": list(DEFAULT_MINERALS),
    "max_seconds": 120.0,
}

# Reagents the caller may dose. Restricted deliberately: an arbitrary formula
# reaches Reaktoro as a database lookup, and an unrecognised one fails deep
# inside block construction with a message that does not name the caller's input.
MODIFIERS = {"acid_addition_mol_s": "HCl", "base_addition_mol_s": "NaOH"}


class ReaktoroSimulationError(RuntimeError):
    """Raised when the model cannot be built, initialized, or solved."""


def available_minerals() -> list[str]:
    """Mineral phases the pinned database can precipitate.

    Read from Reaktoro directly rather than hardcoded, so the list cannot drift
    from what the solver will actually accept.
    """
    from reaktoro import AggregateState, PhreeqcDatabase

    db = PhreeqcDatabase("pitzer.dat")
    names = {
        s.name()
        for s in db.species()
        if s.aggregateState() == AggregateState.Solid
    }
    return sorted(names)


def _validate(p: dict[str, Any]) -> None:
    comp = p["composition"]
    if not isinstance(comp, dict) or not comp:
        raise ReaktoroSimulationError("composition must be a non-empty mapping")
    if "H2O" not in comp:
        raise ReaktoroSimulationError("composition must include H2O")
    for k, v in comp.items():
        if not isinstance(v, (int, float)) or v < 0:
            raise ReaktoroSimulationError(
                f"composition['{k}'] must be a non-negative number (mol/s), got {v!r}"
            )
    if comp["H2O"] <= 0:
        raise ReaktoroSimulationError("composition['H2O'] must be positive")
    if not 0 < p["ph"] < 14:
        raise ReaktoroSimulationError("ph must be between 0 and 14")
    if p["pressure_bar"] <= 0:
        raise ReaktoroSimulationError("pressure_bar must be positive")
    if not -50 < p["temperature_c"] < 300:
        raise ReaktoroSimulationError("temperature_c is outside the supported range")
    rec = p["water_recovery"]
    if rec is not None and not 0 < rec <= 0.95:
        raise ReaktoroSimulationError(
            "water_recovery must be greater than 0 and at most 0.95"
        )
    if not p["minerals"]:
        raise ReaktoroSimulationError("at least one mineral must be requested")
    for key in MODIFIERS:
        if p[key] < 0:
            raise ReaktoroSimulationError(f"{key} must be non-negative")
    # Clamped, not validated: max_seconds is a safety ceiling, so a caller
    # raising it past the limit would defeat the point. Silently lowering it is
    # the correct behaviour here.
    p["max_seconds"] = max(1.0, min(float(p["max_seconds"]), MAX_SOLVE_SECONDS))


def _build(p: dict[str, Any]) -> ConcreteModel:
    comp = p["composition"]
    keys = list(comp)

    m = ConcreteModel()
    m.feed_composition = Var(keys, initialize=comp, units=pyunits.mol / pyunits.s)
    for k in keys:
        m.feed_composition[k].fix(comp[k])

    m.temperature = Var(initialize=273.15 + p["temperature_c"], units=pyunits.K)
    m.temperature.fix()
    m.pressure = Var(initialize=p["pressure_bar"] * BAR, units=pyunits.Pa)
    m.pressure.fix()
    m.pH = Var(initialize=p["ph"], bounds=(1, 13), units=pyunits.dimensionless)
    m.pH.fix()

    # Reagent doses are Vars rather than constants so a later optimization can
    # unfix them — that is the whole reason for using a graybox over a one-shot
    # Reaktoro call.
    m.reagents = Var(list(MODIFIERS), initialize=0.0, units=pyunits.mol / pyunits.s)
    modifier: dict[str, Any] = {}
    for key, formula in MODIFIERS.items():
        m.reagents[key].fix(p[key])
        if p[key] > 0:
            modifier[formula] = m.reagents[key]

    # Concentration is applied as removed water rather than by rewriting the
    # composition. Subtracting H2O from the inlet forces the concentrate to keep
    # the feed's fixed pH, which silently pins an output to an input: pH came back
    # exactly equal to the value passed in at every recovery. Dosing
    # H2O_evaporation against the speciated feed lets pH fall as the solution
    # concentrates, which is what actually happens and what PHREEQC reports.
    recovery = p["water_recovery"]
    if recovery is not None:
        m.water_recovery = Var(initialize=recovery, bounds=(0.0, 0.95))
        m.water_recovery.fix(recovery)
        m.water_removal = Var(
            initialize=comp["H2O"] * recovery, units=pyunits.mol / pyunits.s
        )
        m.eq_water_removal = Constraint(
            expr=m.water_recovery * m.feed_composition["H2O"] == m.water_removal
        )
        modifier["H2O_evaporation"] = m.water_removal

    m.scaling_tendency = Var(
        [("scalingTendency", mineral) for mineral in p["minerals"]], initialize=1.0
    )
    m.solution_ph = Var(initialize=p["ph"], units=pyunits.dimensionless)
    m.osmotic_pressure = Var(initialize=1e5, units=pyunits.Pa)

    outputs: dict[Any, Any] = dict(m.scaling_tendency.items())
    outputs[("pH", None)] = m.solution_ph
    outputs[("osmoticPressure", "H2O")] = m.osmotic_pressure

    state = {
        "temperature": m.temperature,
        "pressure": m.pressure,
        "pH": m.pH,
    }
    if p["pe"] is not None:
        m.pE = Var(initialize=p["pe"], units=pyunits.dimensionless)
        m.pE.fix()
        state["pE"] = m.pE

    m.properties = ReaktoroBlock(
        aqueous_phase={
            "composition": m.feed_composition,
            "convert_to_rkt_species": True,
            "activity_model": "ActivityModelPitzer",
        },
        system_state=state,
        outputs=outputs,
        chemistry_modifier=modifier or None,
        dissolve_species_in_reaktoro=True,
        # Any modifier changes the state, so the feed has to be speciated first
        # and the modifier applied to that result to reach the final state.
        build_speciation_block=True,
    )

    _scale(m, keys, comp)
    return m


def _scale(m: ConcreteModel, keys: list[str], comp: dict[str, float]) -> None:
    """Scale on the caller's own magnitudes.

    Composition spans ~5 orders of magnitude between H2O and a trace ion, and
    reaktoro-pse documents iterate divergence as the usual symptom of leaving
    that unscaled.
    """
    for k in keys:
        iscale.set_scaling_factor(
            m.feed_composition[k], 1.0 / comp[k] if comp[k] > 0 else 1.0
        )
    if m.find_component("water_recovery") is not None:
        iscale.set_scaling_factor(m.water_recovery, 1)
        iscale.set_scaling_factor(m.water_removal, 1.0 / max(comp["H2O"], 1e-8))
    for key in MODIFIERS:
        iscale.set_scaling_factor(m.reagents[key], 1e3)


def _initialize(m: ConcreteModel) -> None:
    if m.find_component("eq_water_removal") is not None:
        calculate_variable_from_constraint(m.water_removal, m.eq_water_removal)
    m.properties.initialize()


def equilibrate(**overrides: Any) -> dict[str, Any]:
    """Equilibrate a feed and report scaling tendencies and bulk properties."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ReaktoroSimulationError(
            f"unknown parameter(s): {sorted(unknown)}; valid: {sorted(DEFAULTS)}"
        )
    p = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    _validate(p)

    with stdout_to_stderr():
        try:
            m = _build(p)
        except ReaktoroSimulationError:
            raise
        except Exception as exc:
            # An unknown mineral or species surfaces here, deep in block
            # construction, so name the likely culprit rather than re-raising raw.
            raise ReaktoroSimulationError(
                f"could not build the chemistry model ({type(exc).__name__}: {exc}). "
                f"Check that every mineral in {p['minerals']} and every species in "
                f"{sorted(p['composition'])} exists in the pitzer.dat database"
            ) from exc

        try:
            _initialize(m)
            solver = get_cyipopt_watertap_solver(
                # recalc_y is the documented workaround for ipopt failing to close
                # dual infeasibility against an approximated graybox hessian.
                solver_options={
                    "recalc_y": "yes",
                    "recalc_y_feas_tol": 1e-2,
                    "max_cpu_time": float(p["max_seconds"]),
                },
            )
            results = solver.solve(m)
        except Exception as exc:
            raise ReaktoroSimulationError(f"solve failed: {exc}") from exc

        if not check_optimal_termination(results):
            tc = results.solver.termination_condition
            raise ReaktoroSimulationError(
                f"solver did not converge (termination: {tc}). For a concentrated "
                "brine this usually means the requested state is infeasible — try a "
                "lower water_recovery or a smaller reagent dose"
            )

        return _extract(m, p)


def _extract(m: ConcreteModel, p: dict[str, Any]) -> dict[str, Any]:
    keys = list(p["composition"])
    recovery = p["water_recovery"]

    scaling = {
        mineral: value(m.scaling_tendency[("scalingTendency", mineral)])
        for mineral in p["minerals"]
    }
    osmotic_pa = value(m.osmotic_pressure)

    out: dict[str, Any] = {
        "inputs": {
            **{k: v for k, v in p.items() if k != "composition"},
            "composition_mol_s": p["composition"],
            "activity_model": "ActivityModelPitzer",
            "database": "PhreeqcDatabase/pitzer.dat",
        },
        "scaling_tendency": scaling,
        # Saturation index is log10 of scaling tendency; reported because most
        # membrane literature quotes SI, and ST > 1 <=> SI > 0 is the scaling
        # threshold either way.
        "saturation_index": {
            mineral: (math.log10(st) if st > 0 else float("-inf"))
            for mineral, st in scaling.items()
        },
        "solution": {
            "ph": value(m.solution_ph),
            "osmotic_pressure_bar": osmotic_pa / BAR,
            "osmotic_pressure_Pa": osmotic_pa,
        },
        "feed_composition_mol_s": {k: value(m.feed_composition[k]) for k in keys},
        "at_risk": sorted(mineral for mineral, st in scaling.items() if st >= 1.0),
    }

    if recovery is not None:
        out["concentration"] = {
            "water_recovery": recovery,
            "concentration_factor": 1.0 / (1.0 - recovery),
            "water_removed_mol_s": value(m.water_removal),
        }
    return out
