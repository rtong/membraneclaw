"""MCP server exposing the WaterTAP ReverseOsmosis0D unit model as tools.

Transport defaults to stdio. Set MCP_TRANSPORT=streamable-http (with MCP_HOST /
MCP_PORT) to serve over HTTP instead.

Nothing may be written to stdout under stdio transport — it carries the protocol —
so the simulation layer captures IDAES/Pyomo output.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Clients launch this by absolute path with an arbitrary cwd, so make the sibling
# module importable regardless of where it was started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ro_model import DEFAULTS, ROSimulationError, simulate

mcp = FastMCP("watertap-ro")


@mcp.tool()
def describe_ro_parameters() -> str:
    """List every RO 0D simulation parameter with its units, default and meaning.

    Call this first when unsure which arguments simulate_ro accepts or what units
    it expects.
    """
    spec = {
        "feed_flow_mass_kg_s": "Total feed mass flow [kg/s]",
        "feed_nacl_mass_frac": "NaCl mass fraction in feed [-], e.g. 0.035 = 35 g/kg seawater",
        "feed_pressure_bar": "Feed pressure [bar]",
        "feed_temperature_c": "Feed temperature [degC]",
        "membrane_area_m2": "Membrane area [m2]",
        "A_comp": "Water permeability coefficient [m/s/Pa]",
        "B_comp": "Salt permeability coefficient [m/s]",
        "permeate_pressure_bar": "Permeate-side pressure [bar]",
        "pressure_drop_bar": "Feed-channel pressure drop across the module [bar]",
        "channel_height_m": "Feed spacer channel height [m] (mass_transfer_coefficient='calculated')",
        "spacer_porosity": "Feed spacer porosity [-] (mass_transfer_coefficient='calculated')",
        "module_length_m": "Module length [m] (mass_transfer_coefficient='calculated')",
        "cp_modulus": "Concentration polarization modulus [-] (concentration_polarization='fixed')",
        "mass_transfer_coeff": "Mass transfer coefficient [m/s] (mass_transfer_coefficient='fixed')",
        "concentration_polarization": "'none' | 'fixed' | 'calculated'",
        "mass_transfer_coefficient": "'none' | 'fixed' | 'calculated'",
    }
    return json.dumps(
        {
            "parameters": [
                {"name": k, "description": v, "default": DEFAULTS[k]}
                for k, v in spec.items()
            ],
            "valid_combinations": [
                "concentration_polarization='none'      + mass_transfer_coefficient='none'",
                "concentration_polarization='fixed'     + mass_transfer_coefficient='none' (uses cp_modulus)",
                "concentration_polarization='calculated'+ mass_transfer_coefficient='fixed' (uses mass_transfer_coeff)",
                "concentration_polarization='calculated'+ mass_transfer_coefficient='calculated' (uses channel geometry)",
            ],
            "note": "Defaults describe a seawater RO module: 1 kg/s of 35 g/kg feed at 50 bar across 50 m2.",
        },
        indent=2,
    )


@mcp.tool()
def simulate_ro(
    feed_flow_mass_kg_s: Optional[float] = None,
    feed_nacl_mass_frac: Optional[float] = None,
    feed_pressure_bar: Optional[float] = None,
    feed_temperature_c: Optional[float] = None,
    membrane_area_m2: Optional[float] = None,
    A_comp: Optional[float] = None,
    B_comp: Optional[float] = None,
    permeate_pressure_bar: Optional[float] = None,
    pressure_drop_bar: Optional[float] = None,
    channel_height_m: Optional[float] = None,
    spacer_porosity: Optional[float] = None,
    module_length_m: Optional[float] = None,
    cp_modulus: Optional[float] = None,
    mass_transfer_coeff: Optional[float] = None,
    concentration_polarization: Optional[str] = None,
    mass_transfer_coefficient: Optional[str] = None,
) -> str:
    """Simulate a reverse osmosis module with the WaterTAP ReverseOsmosis0D model.

    Solves a steady-state seawater RO unit and reports water/salt flux, permeate
    quality, water recovery, salt rejection and concentration polarization. Any
    argument left unset keeps its default (see describe_ro_parameters).

    Returns JSON. On failure returns {"error": ...} rather than raising.
    """
    try:
        return json.dumps(
            simulate(
                feed_flow_mass_kg_s=feed_flow_mass_kg_s,
                feed_nacl_mass_frac=feed_nacl_mass_frac,
                feed_pressure_bar=feed_pressure_bar,
                feed_temperature_c=feed_temperature_c,
                membrane_area_m2=membrane_area_m2,
                A_comp=A_comp,
                B_comp=B_comp,
                permeate_pressure_bar=permeate_pressure_bar,
                pressure_drop_bar=pressure_drop_bar,
                channel_height_m=channel_height_m,
                spacer_porosity=spacer_porosity,
                module_length_m=module_length_m,
                cp_modulus=cp_modulus,
                mass_transfer_coeff=mass_transfer_coeff,
                concentration_polarization=concentration_polarization,
                mass_transfer_coefficient=mass_transfer_coefficient,
            ),
            indent=2,
        )
    except ROSimulationError as exc:
        return json.dumps({"error": str(exc)}, indent=2)
    except Exception as exc:  # never kill the server on a bad request
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport != "stdio":
        mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8002"))
    print(f"watertap-ro MCP starting ({transport})", file=sys.stderr)
    mcp.run(transport=transport)
