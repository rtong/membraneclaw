"""MCP server for the RO calculator tools: unit conversion and limit checks.

Two tools, both pure Python -- no simulation stack, no conda environment, no rate limit -- so this runs on any
interpreter with `mcp` installed, beside (not inside) the ro-chem server in mcp_watertap/, whose WaterTAP and
Reaktoro engines must share one heavy process. The tools are the ones the smoke_test35 notebooks train the
policy with (convert_units since 05, check_limits since 06), declared to the model exactly as there.

Transport defaults to stdio. Set MCP_TRANSPORT=streamable-http (with MCP_HOST / MCP_PORT / MCP_PATH) to serve
over HTTP instead; MCP_BEARER_TOKEN then gates it and MCP_ALLOWED_HOSTS adds host names beyond localhost.

Nothing may be written to stdout under stdio transport -- it carries the protocol.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Clients launch this by absolute path with an arbitrary cwd, so make the sibling modules importable regardless
# of where it was started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calc_limits import check_limits as _check_limits  # noqa: E402
from calc_units import convert_units as _convert_units  # noqa: E402

# The descriptions the notebooks declare to the policy (CONVERT_DECL / COMPARE_DECL), verbatim: a model trained
# on them reads the same text here.
CONVERT_DESCRIPTION = (
    "Convert one number between units.\n\n"
    "Supported: mass flow (kg/s, t/h, kg/h; m3/h and m3/d of water at 1000 kg/m3), volume flow "
    "(m3/s, L/s, m3/h, m3/d, MGD), salinity as a mass fraction (mass fraction, g/kg, \u2030, %, mg/kg, "
    "ppm), concentration (g/L, mg/L, kg/m3, ppm), temperature (C, F), area (m2, ft2), pressure "
    "(bar, kPa, MPa, psi), water permeability A (m/s/Pa, LMH/bar), salt permeability B (m/s, LMH), "
    "flux (LMH, gfd), cost ($/m3, $/kgal), efficiency (fraction, %).\n\n"
    "Returns JSON {\"value\": ..., \"unit\": ...}. On failure returns {\"error\": ...} rather than raising.\n"
)
CHECK_LIMITS_DESCRIPTION = (
    "Check every candidate's values against the limits and write out each comparison.\n\n"
    "limits is an object with one limit per field, written \"op number unit\": op is \">=\" when the "
    "value must be at least the number, \"<=\" when it must be at most the number; \"<\" and \">\" "
    "are strict. A value equal to the number meets \">=\" and \"<=\". The unit is the one the number "
    "is stated in; when it differs from the field's own unit (the unit in the field's name) the number "
    "is converted, with the units convert_units knows. values is an object: for each candidate, by "
    "name, an object with that candidate's value of every limited field, in the field's own unit; a "
    "field given without a limit is written out as it is.\n\n"
    "Returns a JSON list with one line per candidate: \"candidate: field value meets|breaks op limit; "
    "... -> pass|fail\", the limit in the field's own unit. A limit it cannot read -- an unknown field, a "
    "unit that does not convert to the field's own, no op or number -- is written into every line as "
    "\"field value (...)\" and left out of the verdict, and a candidate without a number for a limited "
    "field gets \"field ? (no value)\"; the verdict is then unchecked unless a read limit breaks. Returns "
    "{\"error\": ...} only when limits or values are not objects of that shape.\n"
)

mcp = FastMCP("ro-calc")


@mcp.tool(description=CONVERT_DESCRIPTION)
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    return _convert_units(value, from_unit, to_unit)


@mcp.tool(description=CHECK_LIMITS_DESCRIPTION)
def check_limits(limits: dict[str, str], values: dict[str, dict[str, float]]) -> str:
    return _check_limits(limits, values)


def _http_app():
    """Streamable-HTTP app with host allow-listing and optional bearer auth -- mcp_watertap/server.py's,
    without OAuth: these tools hold no data and need no per-user identity, so a shared secret is enough."""
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.responses import JSONResponse

    extra = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    host, port = mcp.settings.host, mcp.settings.port
    allowed = [f"{host}:{port}", "127.0.0.1", f"127.0.0.1:{port}", "localhost", f"localhost:{port}"]
    for h in extra:
        allowed += [h, f"{h}:{port}", f"{h}:443", f"{h}:8443"]
    mcp.settings.transport_security = TransportSecuritySettings(allowed_hosts=allowed, allowed_origins=["*"])

    app = mcp.streamable_http_app()
    token = os.environ.get("MCP_BEARER_TOKEN", "")
    if token:
        from starlette.middleware.base import BaseHTTPMiddleware

        async def require_bearer(request, call_next):
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if supplied != token:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=require_bearer)
    else:
        print("warning: MCP_BEARER_TOKEN unset, endpoint is unauthenticated", file=sys.stderr)
    return app


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        print("ro-calc MCP starting (stdio)", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        import uvicorn

        mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8004"))
        mcp.settings.streamable_http_path = os.environ.get("MCP_PATH", "/mcp")
        print(f"ro-calc MCP starting (streamable-http) on {mcp.settings.host}:{mcp.settings.port}"
              f"{mcp.settings.streamable_http_path}", file=sys.stderr)
        uvicorn.run(_http_app(), host=mcp.settings.host, port=mcp.settings.port)
