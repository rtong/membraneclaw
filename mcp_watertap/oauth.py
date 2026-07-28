"""OAuth 2.1 authorization server for the WaterTAP MCP endpoint.

Hosted connector UIs (Claude, ChatGPT web) authenticate MCP servers over OAuth
with dynamic client registration; they have no field for a static bearer token, so
a token-gated endpoint returns 401 to them no matter what is configured.

There is no user directory behind this, so the consent step is a single shared
approval key: the client registers itself, the browser lands on /consent, and a
correct key mints the authorization code. That keeps a real authorization
boundary — anyone can register a client, but only someone holding the key can
approve one.

The static bearer token still works (see verify_static in load_access_token), so
the agent and any config-file client keep authenticating the simple way.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

CODE_TTL = 300           # authorization codes are single-use and short-lived
ACCESS_TTL = 3600
STATE_FILE = Path(__file__).resolve().parent / ".oauth_state.json"


class Store:
    """Small JSON-backed store so a service restart doesn't drop connectors.

    Registered clients and refresh tokens survive; short-lived access codes and
    access tokens are allowed to die with the process.
    """

    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self.clients: dict[str, dict] = {}
        self.refresh: dict[str, dict] = {}
        self.codes: dict[str, dict] = {}
        self.access: dict[str, dict] = {}
        self.pending: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.clients = data.get("clients", {})
            self.refresh = data.get("refresh", {})
        except Exception:
            pass  # a corrupt state file should not stop the server booting

    def save(self) -> None:
        # register_client.py writes this same file while the server is running, so
        # a blind overwrite would drop a client registered since we last loaded.
        # Merge on-disk state first, letting in-memory win on genuine conflicts.
        on_disk: dict[str, Any] = {}
        if self.path.exists():
            try:
                on_disk = json.loads(self.path.read_text())
            except Exception:
                on_disk = {}
        clients = {**on_disk.get("clients", {}), **self.clients}
        refresh = {**on_disk.get("refresh", {}), **self.refresh}
        self.clients, self.refresh = clients, refresh

        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"clients": clients, "refresh": refresh}))
        tmp.replace(self.path)
        os.chmod(self.path, 0o600)

    def reload_clients(self) -> None:
        """Pick up clients registered by another process without a restart."""
        if not self.path.exists():
            return
        try:
            self.clients.update(json.loads(self.path.read_text()).get("clients", {}))
        except Exception:
            pass


class WatertapOAuthProvider(OAuthAuthorizationServerProvider):
    def __init__(self, approval_key: str, static_token: str = "", store: Store | None = None):
        self.approval_key = approval_key
        self.static_token = static_token
        self.store = store or Store()

    # ---- client registration ------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        raw = self.store.clients.get(client_id)
        if raw is None:
            # Could have been pre-registered by register_client.py after we booted.
            self.store.reload_clients()
            raw = self.store.clients.get(client_id)
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.store.clients[client_info.client_id] = client_info.model_dump(mode="json")
        self.store.save()

    # ---- authorization ------------------------------------------------------

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """Park the request and send the browser to the consent page.

        Returning a URL here is what the SDK redirects to; the code is not minted
        until /consent verifies the approval key.
        """
        rid = secrets.token_urlsafe(24)
        self.store.pending[rid] = {
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "state": params.state,
            "code_challenge": params.code_challenge,
            # Connectors omit `scope` at /authorize; fall back to the grant this
            # server issues rather than minting a scope-less token.
            "scopes": params.scopes or ["watertap"],
            "resource": params.resource,
            "expires_at": time.time() + CODE_TTL,
        }
        return f"/consent?rid={rid}"

    def complete_consent(self, rid: str, key: str) -> str | None:
        """Verify the approval key and return the client redirect URL, or None."""
        pending = self.store.pending.get(rid)
        if not pending or pending["expires_at"] < time.time():
            self.store.pending.pop(rid, None)
            return None
        if not secrets.compare_digest(key, self.approval_key):
            return None
        self.store.pending.pop(rid, None)

        code = secrets.token_urlsafe(32)
        self.store.codes[code] = {
            "code": code,
            "client_id": pending["client_id"],
            "redirect_uri": pending["redirect_uri"],
            "redirect_uri_provided_explicitly": pending["redirect_uri_provided_explicitly"],
            "code_challenge": pending["code_challenge"],
            "scopes": pending["scopes"],
            "resource": pending["resource"],
            "expires_at": time.time() + CODE_TTL,
        }
        sep = "&" if "?" in pending["redirect_uri"] else "?"
        url = f"{pending['redirect_uri']}{sep}code={code}"
        if pending["state"]:
            url += f"&state={pending['state']}"
        return url

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        raw = self.store.codes.get(authorization_code)
        if not raw or raw["client_id"] != client.client_id or raw["expires_at"] < time.time():
            return None
        return AuthorizationCode.model_validate(raw)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Single use: burn the code before issuing anything.
        self.store.codes.pop(authorization_code.code, None)
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource)

    # ---- tokens -------------------------------------------------------------

    def _issue(self, client_id: str, scopes: list[str], resource: str | None) -> OAuthToken:
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        now = int(time.time())
        self.store.access[access] = {
            "token": access, "client_id": client_id, "scopes": scopes,
            "expires_at": now + ACCESS_TTL, "resource": resource,
        }
        self.store.refresh[refresh] = {
            "token": refresh, "client_id": client_id, "scopes": scopes, "expires_at": None,
        }
        self.store.save()
        return OAuthToken(
            access_token=access, token_type="Bearer",
            expires_in=ACCESS_TTL, refresh_token=refresh, scope=" ".join(scopes),
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        raw = self.store.refresh.get(refresh_token)
        if not raw or raw["client_id"] != client.client_id:
            return None
        return RefreshToken.model_validate(raw)

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        self.store.refresh.pop(refresh_token.token, None)
        return self._issue(client.client_id, scopes or refresh_token.scopes, None)

    async def load_access_token(self, token: str) -> AccessToken | None:
        # The static bearer token stays valid so non-OAuth clients (the agent,
        # config-file clients) keep working against the same endpoint.
        if self.static_token and secrets.compare_digest(token, self.static_token):
            return AccessToken(token=token, client_id="static", scopes=["watertap"], expires_at=None)
        raw = self.store.access.get(token)
        if not raw:
            return None
        if raw["expires_at"] and raw["expires_at"] < time.time():
            self.store.access.pop(token, None)
            return None
        return AccessToken.model_validate(raw)

    async def revoke_token(self, token: Any) -> None:
        tok = getattr(token, "token", token)
        self.store.access.pop(tok, None)
        self.store.refresh.pop(tok, None)
        self.store.save()


CONSENT_PAGE = """<!doctype html>
<title>Authorize WaterTAP MCP</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:26rem;margin:4rem auto;padding:0 1rem;line-height:1.5}}
 input,button{{font:inherit;padding:.55rem .7rem;width:100%;box-sizing:border-box;margin-top:.4rem}}
 button{{margin-top:1rem;cursor:pointer}} .err{{color:#b00}} code{{background:#f4f4f4;padding:.1rem .3rem}}
</style>
<h2>Authorize WaterTAP MCP</h2>
<p>A client is requesting access to the reverse-osmosis simulation tools.</p>
{error}
<form method="post">
  <input type="hidden" name="rid" value="{rid}">
  <label>Approval key<input type="password" name="key" autofocus autocomplete="off"></label>
  <button type="submit">Approve</button>
</form>
"""
