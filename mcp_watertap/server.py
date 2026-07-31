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

# Module-qualified: reaktoro_model has its own DEFAULTS, and the two must not be
# confused — they describe different models with different units.
import reaktoro_model

def _build_mcp() -> tuple[FastMCP, object]:
    """FastMCP instance, plus the OAuth provider when one is configured.

    OAuth turns on only when MCP_OAUTH_APPROVAL_KEY and MCP_PUBLIC_URL are both
    set — the issuer must be the externally reachable URL, or the discovery
    documents advertise endpoints the client cannot reach.
    """
    approval_key = os.environ.get("MCP_OAUTH_APPROVAL_KEY", "").strip()
    public_url = os.environ.get("MCP_PUBLIC_URL", "").strip().rstrip("/")
    if not (approval_key and public_url):
        return FastMCP("watertap-ro"), None

    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
    from pydantic import AnyHttpUrl

    from oauth import WatertapOAuthProvider

    provider = WatertapOAuthProvider(
        approval_key=approval_key,
        static_token=os.environ.get("MCP_BEARER_TOKEN", ""),
    )
    settings = AuthSettings(
        issuer_url=AnyHttpUrl(public_url),
        resource_server_url=AnyHttpUrl(public_url),
        # Hosted connectors register themselves; there is no console to pre-create
        # a client in, so dynamic registration is required rather than optional.
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=["watertap"], default_scopes=["watertap"]
        ),
        revocation_options=RevocationOptions(enabled=True),
        # Deliberately empty: connectors request no `scope` at /authorize, so
        # gating on one mints tokens that fail the check, 401 the MCP call, and
        # send the client back to re-authorize in a loop.
        required_scopes=[],
    )
    return FastMCP("watertap-ro", auth_server_provider=provider, auth=settings), provider


mcp, _oauth_provider = _build_mcp()


def _mount_consent(app) -> None:
    """Approval-key page that turns a parked authorize request into a code."""
    from starlette.responses import HTMLResponse, RedirectResponse
    from starlette.routing import Route

    from oauth import CONSENT_PAGE

    async def consent(request):
        if request.method == "GET":
            rid = request.query_params.get("rid", "")
            return HTMLResponse(CONSENT_PAGE.format(rid=rid, error=""))
        form = await request.form()
        target = _oauth_provider.complete_consent(
            str(form.get("rid", "")), str(form.get("key", ""))
        )
        if target is None:
            return HTMLResponse(
                CONSENT_PAGE.format(
                    rid=str(form.get("rid", "")),
                    error='<p class="err">Incorrect or expired approval key.</p>',
                ),
                status_code=401,
            )
        # 303, not 302: this redirect follows a POST, and 302 lets the browser
        # repeat it as a POST. The connector callback only accepts GET, so a 302
        # lands as a rejected POST — the browser returns to /authorize and the
        # flow loops without ever reaching /token.
        return RedirectResponse(target, status_code=303)

    app.router.routes.append(Route("/consent", consent, methods=["GET", "POST"]))


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


@mcp.tool()
def describe_reaktoro_options() -> str:
    """List the inputs, reagents and minerals the chemistry tools accept.

    Call this before equilibrate_feed or analyze_ro_scaling when unsure which
    ions or mineral names are valid.
    """
    return json.dumps(
        {
            "pinned": {
                "database": "PhreeqcDatabase/pitzer.dat",
                "activity_model": "ActivityModelPitzer",
                "why": (
                    "Not caller-selectable. reaktoro-pse defaults every phase to an "
                    "ideal activity model, which for brine returns confident but "
                    "wrong numbers instead of failing."
                ),
            },
            "composition": {
                "units": "mol/s per ion, plus H2O",
                "supported_species": sorted(reaktoro_model.MOLAR_MASS),
                "default_seawater": reaktoro_model.DEFAULT_COMPOSITION,
                "note": "H2O is required. Unlisted species are rejected, not ignored.",
            },
            "reagents": {
                "acid_addition_mol_s": "HCl dose [mol/s]",
                "base_addition_mol_s": "NaOH dose [mol/s]",
            },
            "minerals": {
                "default": list(reaktoro_model.DEFAULT_MINERALS),
                "available": reaktoro_model.available_minerals(),
            },
            "interpretation": {
                "scaling_tendency": "ion activity product / solubility product; >= 1 means it can precipitate",
                "saturation_index": "log10(scaling_tendency); >= 0 means it can precipitate",
                "acid_dosing": (
                    "Lowers carbonate scale (Calcite) strongly. Does not help "
                    "sulfate scale (Gypsum, Anhydrite, Barite), whose solubility "
                    "is essentially pH-independent."
                ),
            },
            "osmotic_pressure_caveat": (
                "Osmotic pressure here will not match simulate_ro's for the same "
                "water. simulate_ro treats all dissolved solids as NaCl, which "
                "contributes more osmoles per gram than the Mg/Ca/SO4 salts in a "
                "real analysis. Quote each number against its own tool."
            ),
        },
        indent=2,
    )


@mcp.tool()
def equilibrate_feed(
    composition_mol_s: Optional[dict] = None,
    temperature_c: Optional[float] = None,
    pressure_bar: Optional[float] = None,
    ph: Optional[float] = None,
    water_recovery: Optional[float] = None,
    acid_addition_mol_s: Optional[float] = None,
    base_addition_mol_s: Optional[float] = None,
    minerals: Optional[list] = None,
) -> str:
    """Equilibrate a water and report which minerals can scale.

    Computes mineral scaling tendencies, saturation indices, pH and osmotic
    pressure for a feed, optionally after concentrating it (water_recovery) and
    optionally after dosing acid or base. Use this for a standalone water; use
    analyze_ro_scaling to derive the recovery from RO operating conditions
    instead of supplying it.

    Returns JSON. On failure returns {"error": ...} rather than raising.
    """
    try:
        return json.dumps(
            reaktoro_model.equilibrate(
                composition=composition_mol_s,
                temperature_c=temperature_c,
                pressure_bar=pressure_bar,
                ph=ph,
                water_recovery=water_recovery,
                acid_addition_mol_s=acid_addition_mol_s,
                base_addition_mol_s=base_addition_mol_s,
                minerals=minerals,
            ),
            indent=2,
        )
    except reaktoro_model.ReaktoroSimulationError as exc:
        return json.dumps({"error": str(exc)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)


@mcp.tool()
def analyze_ro_scaling(
    composition_mol_s: Optional[dict] = None,
    feed_pressure_bar: Optional[float] = None,
    membrane_area_m2: Optional[float] = None,
    feed_flow_mass_kg_s: Optional[float] = None,
    feed_temperature_c: Optional[float] = None,
    ph: Optional[float] = None,
    acid_addition_mol_s: Optional[float] = None,
    minerals: Optional[list] = None,
) -> str:
    """Simulate an RO module, then report what scales in its concentrate.

    Chains the two models: solves the RO unit to get water recovery, then
    equilibrates the feed concentrated to exactly that recovery. Answers "at
    these operating conditions, what precipitates on the membrane?" in one call.

    Salinity for the RO model is derived from composition_mol_s, so both halves
    describe the same water.

    Returns JSON. On failure returns {"error": ...} rather than raising.
    """
    try:
        composition = composition_mol_s or reaktoro_model.DEFAULT_COMPOSITION
        salinity = reaktoro_model.composition_salinity(composition)

        ro = simulate(
            feed_nacl_mass_frac=salinity["mass_fraction"],
            feed_pressure_bar=feed_pressure_bar,
            membrane_area_m2=membrane_area_m2,
            feed_flow_mass_kg_s=feed_flow_mass_kg_s,
            feed_temperature_c=feed_temperature_c,
        )
        recovery = ro["performance"]["water_recovery_pct"] / 100.0

        chem = reaktoro_model.equilibrate(
            composition=composition,
            # The concentrate sits at the retentate pressure and the feed
            # temperature, so equilibrate it there rather than at 1 bar.
            temperature_c=ro["inputs"]["feed_temperature_c"],
            pressure_bar=ro["retentate"]["pressure_bar"],
            ph=ph,
            water_recovery=recovery,
            acid_addition_mol_s=acid_addition_mol_s,
            minerals=minerals,
        )

        return json.dumps(
            {
                "feed_salinity": salinity,
                "ro_performance": {
                    "water_recovery_pct": ro["performance"]["water_recovery_pct"],
                    "salt_rejection_pct": ro["performance"]["salt_rejection_pct"],
                    "water_flux_LMH": ro["flux"]["water_LMH"],
                    "retentate_pressure_bar": ro["retentate"]["pressure_bar"],
                },
                "concentrate_chemistry": chem,
                "verdict": (
                    f"at {recovery * 100:.1f}% recovery, at risk of scaling: "
                    + (", ".join(chem["at_risk"]) if chem["at_risk"] else "none")
                ),
            },
            indent=2,
        )
    except (ROSimulationError, reaktoro_model.ReaktoroSimulationError) as exc:
        return json.dumps({"error": str(exc)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)


def _http_app():
    """Streamable-HTTP app with host allow-listing and optional bearer auth.

    Two things the stock FastMCP HTTP transport does not give us:

    * DNS-rebinding protection only trusts localhost, so reaching the server by
      any other name (a tailnet `*.ts.net` host, say) returns 421 Misdirected
      Request. MCP_ALLOWED_HOSTS adds those names.
    * There is no built-in static-token auth, but ChatGPT's config supports
      `bearer_token_env_var`, so a shared secret is the natural fit.
    """
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.responses import JSONResponse

    extra = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    host, port = mcp.settings.host, mcp.settings.port
    allowed = [f"{host}:{port}", "127.0.0.1", f"127.0.0.1:{port}", "localhost", f"localhost:{port}"]
    for h in extra:
        allowed += [h, f"{h}:{port}", f"{h}:443", f"{h}:8443"]
    mcp.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=allowed,
        allowed_origins=["*"],
    )

    app = mcp.streamable_http_app()
    token = os.environ.get("MCP_BEARER_TOKEN", "")
    if _oauth_provider is not None:
        _mount_consent(app)
        return app  # OAuth verifies tokens; the bearer middleware would double-gate
    if token:
        # streamable_http_app() returns a bare Starlette app, which has no
        # @app.middleware decorator — that is FastAPI-only.
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
        print("watertap-ro MCP starting (stdio)", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        import uvicorn

        mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8002"))
        # A connector UI keys its "already exists" check on the exact URL string,
        # so a stale record that can't be removed is worked around by serving the
        # same endpoint at a different path. OAuth is unaffected — issuer and
        # resource metadata key off the base URL, not this path.
        mcp.settings.streamable_http_path = os.environ.get("MCP_PATH", "/mcp")
        print(
            f"watertap-ro MCP starting (streamable-http) on "
            f"{mcp.settings.host}:{mcp.settings.port}{mcp.settings.streamable_http_path}",
            file=sys.stderr,
        )
        uvicorn.run(_http_app(), host=mcp.settings.host, port=mcp.settings.port)
