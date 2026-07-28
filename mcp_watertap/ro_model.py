"""WaterTAP ReverseOsmosis0D simulation, wrapped for tool use.

Each call builds and solves a fresh Pyomo model, so simulations are stateless and
independent. Inputs are taken in practical units (bar, degC, LMH) and converted to
the SI units WaterTAP expects; results are returned in both.

Reference: https://watertap.readthedocs.io/en/stable/technical_reference/unit_models/reverse_osmosis_0D.html
"""
from __future__ import annotations

import io
import sys
import contextlib
import os
from pathlib import Path
from typing import Any

# The IDAES ipopt build links libgfortran.so.5 / liblapack.so.3, which Debian 13
# does not ship. .vendor holds them unpacked from .deb files (see README). ipopt
# runs as a subprocess, so exporting this before Pyomo spawns it is enough.
_VENDOR_LIB = Path(__file__).resolve().parent.parent / ".vendor/root/usr/lib/x86_64-linux-gnu"
if _VENDOR_LIB.is_dir():
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    if str(_VENDOR_LIB) not in _existing.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{_VENDOR_LIB}{os.pathsep}{_existing}" if _existing else str(_VENDOR_LIB)
        )

from pyomo.environ import ConcreteModel, value
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale

import watertap.property_models.NaCl_prop_pack as props
from watertap.core.solvers import get_solver
from watertap.unit_models.reverse_osmosis_0D import ReverseOsmosis0D
from watertap.core.membrane_channel_base import (
    ConcentrationPolarizationType,
    MassTransferCoefficient,
)

BAR = 1e5
LMH_PER_KG_M2_S = 3600.0  # kg/m2/s -> L/m2/h, taking water density as 1000 kg/m3

CP_TYPES = {
    "none": ConcentrationPolarizationType.none,
    "fixed": ConcentrationPolarizationType.fixed,
    "calculated": ConcentrationPolarizationType.calculated,
}
KF_TYPES = {
    "none": MassTransferCoefficient.none,
    "fixed": MassTransferCoefficient.fixed,
    "calculated": MassTransferCoefficient.calculated,
}

DEFAULTS: dict[str, Any] = {
    "feed_flow_mass_kg_s": 1.0,
    "feed_nacl_mass_frac": 0.035,
    "feed_pressure_bar": 50.0,
    "feed_temperature_c": 25.0,
    "membrane_area_m2": 50.0,
    "A_comp": 4.2e-12,
    "B_comp": 3.5e-8,
    "permeate_pressure_bar": 1.01325,
    "pressure_drop_bar": 3.0,
    "channel_height_m": 0.002,
    "spacer_porosity": 0.75,
    "module_length_m": 20.0,
    "cp_modulus": 1.1,
    "mass_transfer_coeff": 2e-5,
    "concentration_polarization": "calculated",
    "mass_transfer_coefficient": "calculated",
}


class ROSimulationError(RuntimeError):
    """Raised when the model cannot be built, initialized, or solved."""


@contextlib.contextmanager
def _stdout_to_stderr():
    """Point OS-level fd 1 at fd 2 for the duration.

    contextlib.redirect_stdout only swaps sys.stdout, which is not enough here:
    IDAES's logging handler binds the original stdout at import time, and ipopt
    writes from C. Under MCP stdio transport a single stray byte on fd 1 corrupts
    the JSON-RPC stream, so the redirect has to happen at the descriptor level.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _build(p: dict[str, Any]) -> ConcreteModel:
    cp = CP_TYPES[p["concentration_polarization"]]
    kf = KF_TYPES[p["mass_transfer_coefficient"]]

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = props.NaClParameterBlock()
    m.fs.unit = ReverseOsmosis0D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        concentration_polarization_type=cp,
        mass_transfer_coefficient=kf,
    )
    u = m.fs.unit

    flow, frac = p["feed_flow_mass_kg_s"], p["feed_nacl_mass_frac"]
    u.inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(flow * frac)
    u.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(flow * (1 - frac))
    u.inlet.pressure[0].fix(p["feed_pressure_bar"] * BAR)
    u.inlet.temperature[0].fix(273.15 + p["feed_temperature_c"])
    u.deltaP.fix(-p["pressure_drop_bar"] * BAR)
    u.area.fix(p["membrane_area_m2"])
    u.A_comp.fix(p["A_comp"])
    u.B_comp.fix(p["B_comp"])
    u.permeate.pressure[0].fix(p["permeate_pressure_bar"] * BAR)

    # Each concentration-polarization / mass-transfer combination closes the
    # degrees of freedom differently.
    if kf is MassTransferCoefficient.calculated:
        u.feed_side.channel_height.fix(p["channel_height_m"])
        u.feed_side.spacer_porosity.fix(p["spacer_porosity"])
        u.length.fix(p["module_length_m"])
    elif kf is MassTransferCoefficient.fixed:
        for x in (0.0, 1.0):
            u.feed_side.K[0, x, "NaCl"].fix(p["mass_transfer_coeff"])
    elif cp is ConcentrationPolarizationType.fixed:
        u.feed_side.cp_modulus.fix(p["cp_modulus"])

    _touch_reported_properties(u, cp)

    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "NaCl"))
    iscale.calculate_scaling_factors(m.fs.unit)
    return m


def _touch_reported_properties(u, cp) -> None:
    """Build every property we report, before the solve.

    WaterTAP property blocks construct variables on demand. A property first
    accessed in _extract is created *after* the solve, so it keeps its default
    initial value and is never constrained — which returns a plausible-looking
    wrong number rather than an error (feed osmotic pressure read 10 bar instead
    of 28.5). Touching them here puts them in the solved system, and before
    calculate_scaling_factors so the new constraints get scaled too.
    """
    for blk in (u.feed_side.properties_in[0], u.feed_side.properties_out[0]):
        blk.conc_mass_phase_comp
        blk.pressure_osm_phase
    u.mixed_permeate[0].mass_frac_phase_comp
    u.mixed_permeate[0].conc_mass_phase_comp
    if cp is not ConcentrationPolarizationType.none:
        for x in (0.0, 1.0):
            u.feed_side.properties_interface[0, x].conc_mass_phase_comp


def _validate(p: dict[str, Any]) -> None:
    if p["concentration_polarization"] not in CP_TYPES:
        raise ROSimulationError(
            f"concentration_polarization must be one of {sorted(CP_TYPES)}"
        )
    if p["mass_transfer_coefficient"] not in KF_TYPES:
        raise ROSimulationError(
            f"mass_transfer_coefficient must be one of {sorted(KF_TYPES)}"
        )
    # WaterTAP requires a mass transfer coefficient to compute polarization, and
    # rejects a coefficient when polarization is switched off.
    cp, kf = p["concentration_polarization"], p["mass_transfer_coefficient"]
    if cp == "calculated" and kf == "none":
        raise ROSimulationError(
            "concentration_polarization='calculated' requires "
            "mass_transfer_coefficient='fixed' or 'calculated'"
        )
    if cp == "none" and kf != "none":
        raise ROSimulationError(
            "mass_transfer_coefficient must be 'none' when "
            "concentration_polarization='none'"
        )
    if cp == "fixed" and kf != "none":
        raise ROSimulationError(
            "mass_transfer_coefficient must be 'none' when "
            "concentration_polarization='fixed' (use cp_modulus instead)"
        )
    if not 0 < p["feed_nacl_mass_frac"] < 1:
        raise ROSimulationError("feed_nacl_mass_frac must be between 0 and 1")
    if p["feed_flow_mass_kg_s"] <= 0 or p["membrane_area_m2"] <= 0:
        raise ROSimulationError("feed flow and membrane area must be positive")
    if p["feed_pressure_bar"] <= p["permeate_pressure_bar"]:
        raise ROSimulationError("feed pressure must exceed permeate pressure")


def simulate(**overrides: Any) -> dict[str, Any]:
    """Build, initialize and solve an RO 0D model. Returns a results dict."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ROSimulationError(
            f"unknown parameter(s): {sorted(unknown)}; valid: {sorted(DEFAULTS)}"
        )
    p = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    _validate(p)

    with _stdout_to_stderr():
        m = _build(p)
        u = m.fs.unit

        dof = degrees_of_freedom(m)
        if dof != 0:
            raise ROSimulationError(
                f"model has {dof} degrees of freedom (needs 0); the chosen "
                "concentration_polarization / mass_transfer_coefficient combination "
                "does not match the supplied parameters"
            )

        try:
            u.initialize(outlvl=0)
            results = get_solver().solve(m)
        except Exception as exc:
            raise ROSimulationError(f"solve failed: {exc}") from exc

        tc = str(results.solver.termination_condition)
        if tc != "optimal":
            raise ROSimulationError(f"solver did not converge (termination: {tc})")

        return _extract(u, p)


def _extract(u, p: dict[str, Any]) -> dict[str, Any]:
    feed = u.feed_side.properties_in[0]
    ret = u.feed_side.properties_out[0]
    perm = u.mixed_permeate[0]

    jw = value(u.flux_mass_phase_comp_avg[0, "Liq", "H2O"])
    js = value(u.flux_mass_phase_comp_avg[0, "Liq", "NaCl"])
    perm_h2o = value(perm.flow_mass_phase_comp["Liq", "H2O"])
    perm_nacl = value(perm.flow_mass_phase_comp["Liq", "NaCl"])

    out: dict[str, Any] = {
        "inputs": p,
        "flux": {
            "water_LMH": jw * LMH_PER_KG_M2_S,
            "water_kg_m2_s": jw,
            "salt_g_m2_h": js * 3600 * 1000,
            "salt_kg_m2_s": js,
        },
        "permeate": {
            "flow_kg_s": perm_h2o + perm_nacl,
            "water_kg_s": perm_h2o,
            "nacl_kg_s": perm_nacl,
            "nacl_ppm": value(perm.mass_frac_phase_comp["Liq", "NaCl"]) * 1e6,
            "pressure_bar": p["permeate_pressure_bar"],
        },
        "retentate": {
            "flow_kg_s": sum(
                value(ret.flow_mass_phase_comp["Liq", j]) for j in ("H2O", "NaCl")
            ),
            "nacl_g_L": value(ret.conc_mass_phase_comp["Liq", "NaCl"]),
            "pressure_bar": value(u.retentate.pressure[0]) / BAR,
            "osmotic_pressure_bar": value(ret.pressure_osm_phase["Liq"]) / BAR,
        },
        "feed": {
            "nacl_g_L": value(feed.conc_mass_phase_comp["Liq", "NaCl"]),
            "osmotic_pressure_bar": value(feed.pressure_osm_phase["Liq"]) / BAR,
        },
        "performance": {
            "water_recovery_pct": value(u.recovery_vol_phase[0, "Liq"]) * 100,
            "salt_rejection_pct": value(u.rejection_phase_comp[0, "Liq", "NaCl"]) * 100,
            "pressure_drop_bar": -value(u.deltaP[0]) / BAR,
            "net_driving_pressure_bar": (
                p["feed_pressure_bar"]
                - p["permeate_pressure_bar"]
                - value(feed.pressure_osm_phase["Liq"]) / BAR
            ),
        },
    }

    # Only produced when polarization is modelled.
    if p["concentration_polarization"] != "none":
        iface_in = u.feed_side.properties_interface[0, 0]
        iface_out = u.feed_side.properties_interface[0, 1]
        out["concentration_polarization"] = {
            "wall_nacl_g_L_inlet": value(iface_in.conc_mass_phase_comp["Liq", "NaCl"]),
            "wall_nacl_g_L_outlet": value(iface_out.conc_mass_phase_comp["Liq", "NaCl"]),
            "cp_modulus_inlet": (
                value(iface_in.conc_mass_phase_comp["Liq", "NaCl"])
                / value(feed.conc_mass_phase_comp["Liq", "NaCl"])
            ),
        }
    return out
