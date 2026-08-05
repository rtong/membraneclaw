"""OAuth 2.0 client credential for Google's official Workspace MCP servers.

Google hosts the Workspace MCP servers remotely
(https://developers.google.com/workspace/guides/configure-mcp-servers); they are
plain streamable-http endpoints that want a Google access token as a bearer.
qwen-agent's MCP client only supports *static* headers, and access tokens last an
hour while `server.py` keeps an Assistant (and its MCP session) alive for the whole
process, so a header baked in at construction is dead well before the process is.

The refresh therefore lives in an httpx.Auth, which mcp's own streamable_http client
accepts per request — see `agent/compat.py` for how it gets threaded through.

Hand-rolled for the same reason `mcp_watertap/oauth.py` is: the flow is ~100 lines
of stdlib and adding google-auth-oauthlib would pull in a transitive tree for it.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Generator

import httpx

from agent import config

logger = logging.getLogger("membraneclaw")

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

# Union of the scopes Google documents for the Drive and Sheets MCP servers. One
# consent covers both; Sheets needs the Drive pair as well as the sheets pair.
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]

TOKEN_FILE = config.ROOT / ".google_token.json"

# Per-user credentials, one file per verified Open WebUI user id. TOKEN_FILE above
# stays the *owner's* token, used only by the CLI where the operator is the only
# user; server.py never reaches for it, because doing so would hand one person's
# Drive to every logged-in user (see agent/identity.py).
TOKEN_DIR = config.ROOT / ".google_tokens"
CALLBACK_PATH = "/oauth2callback"
# Web-application OAuth clients match the redirect URI *exactly*, port included
# (loopback/desktop clients are the ones where Google ignores the port). Whatever
# is set here must also be registered on the client in the console.
OAUTH_PORT = int(os.environ.get("GOOGLE_OAUTH_PORT", "8765"))
REDIRECT_URI = f"http://localhost:{OAUTH_PORT}{CALLBACK_PATH}"

# Refresh this far ahead of the stated expiry, so a token cannot go stale between
# the check and the request landing at Google.
EXPIRY_SKEW = 60


def client_credentials() -> tuple[str, str]:
    cid = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise RuntimeError(
            "GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET must be set "
            "(see .env.example)"
        )
    return cid, secret


# ---- token store ------------------------------------------------------------


def token_path_for(user_id: str) -> Path:
    """Where this user's Google credentials live.

    The id lands in a filename, so it is validated rather than trusted; identity
    .safe_id already rejects anything that could escape TOKEN_DIR, and this raises
    instead of falling back to a shared path if it somehow gets here unchecked.
    """
    from agent import identity

    if not identity.safe_id(user_id):
        raise ValueError(f"unusable user id for a token path: {user_id!r}")
    TOKEN_DIR.mkdir(mode=0o700, exist_ok=True)
    # An existing directory keeps its old mode, so re-assert it — these are
    # long-lived refresh tokens for other people.
    os.chmod(TOKEN_DIR, 0o700)
    return TOKEN_DIR / f"{user_id}.json"


def load_token(path: Path = TOKEN_FILE) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        # A corrupt token file should read as "not logged in" rather than crash the
        # agent at import; `login` rewrites it.
        logger.warning("could not parse %s, treating as unauthenticated", path)
        return {}


def save_token(data: dict[str, Any], path: Path = TOKEN_FILE) -> None:
    # Same write pattern as mcp_watertap/oauth.py: a refresh token is a long-lived
    # credential, so it lands atomically and never sits world-readable.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)


def have_token(path: Path = TOKEN_FILE) -> bool:
    return bool(load_token(path).get("refresh_token"))


def revoke(path: Path) -> bool:
    """Revoke this user's grant at Google and delete their token file.

    Returns True if Google's revoke call succeeded (or there was nothing to
    revoke), False if it failed — the file is deleted either way, because a stale
    local file that still refreshes is worse than a grant Google thinks is live
    for a few extra minutes until it naturally expires unused.
    """
    data = load_token(path)
    revoked_ok = True
    token = data.get("refresh_token") or data.get("access_token")
    if token:
        resp = httpx.post(REVOKE_ENDPOINT, data={"token": token}, timeout=15)
        # Google returns 200 even if the token was already invalid/expired — a
        # non-200 here means the *request* failed, not that there was nothing to
        # revoke, so it's worth surfacing distinctly from "no token file at all".
        revoked_ok = resp.status_code == 200
        if not revoked_ok:
            logger.warning(
                "Google revoke failed for %s (%s): %s", path.name, resp.status_code, resp.text[:200]
            )
    path.unlink(missing_ok=True)
    return revoked_ok


def _store(payload: dict[str, Any], path: Path, refresh_token: str = "") -> dict[str, Any]:
    """Fold a Google token response into the on-disk record.

    A refresh_token grant response carries no new refresh_token, so the existing one
    is carried forward rather than blanked.
    """
    data = load_token(path)
    data["access_token"] = payload["access_token"]
    data["expires_at"] = time.time() + float(payload.get("expires_in", 3600))
    data["scopes"] = (payload.get("scope") or " ".join(SCOPES)).split()
    new_refresh = payload.get("refresh_token") or refresh_token or data.get("refresh_token")
    if new_refresh:
        data["refresh_token"] = new_refresh
    save_token(data, path)
    return data


class NotAuthorized(Exception):
    """This user has no usable Google credentials yet.

    Raised instead of returning an error string so callers can tell "needs to log
    in" apart from "the call failed", and answer with a login link.
    """


_refresh_lock = threading.Lock()


def access_token(path: Path) -> str:
    """A currently-valid access token for this user, refreshing if needed.

    The sync counterpart to GoogleAuth: qwen-agent executes tools synchronously on
    its own loop thread, so the tools cannot await the async flow.
    """
    data = load_token(path)
    if not data.get("refresh_token"):
        raise NotAuthorized(str(path))

    def _fresh(d: dict) -> bool:
        return bool(d.get("access_token")) and float(d.get("expires_at", 0)) - EXPIRY_SKEW > time.time()

    if _fresh(data):
        return data["access_token"]

    with _refresh_lock:
        # Another thread may have refreshed while this one waited, and another
        # process may have written a good token since we loaded.
        data = load_token(path)
        if _fresh(data):
            return data["access_token"]
        refresh_token = data.get("refresh_token")
        if not refresh_token:
            raise NotAuthorized(str(path))
        client_id, client_secret = client_credentials()
        resp = httpx.post(TOKEN_ENDPOINT, data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }, timeout=30)
        if resp.status_code != 200:
            # A revoked or expired grant is unrecoverable without a fresh consent,
            # so surface it as "log in again" rather than a transient failure.
            if resp.status_code in (400, 401):
                logger.warning("Google refresh rejected for %s: %s", path.name, resp.text[:200])
                raise NotAuthorized(str(path))
            raise RuntimeError(f"Google token refresh failed ({resp.status_code}): {resp.text}")
        data = _store(resp.json(), path, refresh_token=refresh_token)
        logger.info("refreshed Google access token for %s", path.name)
        return data["access_token"]


# ---- interactive login ------------------------------------------------------


def pkce_pair() -> tuple[str, str]:
    """(verifier, challenge) for an authorization-code + PKCE flow."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return verifier, challenge


def authorize_url(redirect_uri: str, state: str, challenge: str) -> str:
    client_id, _ = client_credentials()
    return AUTH_ENDPOINT + "?" + urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # Google only returns a refresh token when both of these are set, and only
        # re-issues one on a *fresh* consent — without prompt=consent a second login
        # silently yields an access token that cannot be renewed.
        "access_type": "offline",
        "prompt": "consent",
    })


def exchange_code(code: str, verifier: str, redirect_uri: str, path: Path) -> dict[str, Any]:
    """Trade an authorization code for tokens and persist them at `path`."""
    client_id, client_secret = client_credentials()
    resp = httpx.post(TOKEN_ENDPOINT, data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": verifier,
    }, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"token exchange failed ({resp.status_code}): {resp.text}")
    payload = resp.json()
    if not payload.get("refresh_token"):
        raise RuntimeError(
            "Google returned no refresh token. Revoke this app's access at "
            "https://myaccount.google.com/permissions and authorize again."
        )
    return _store(payload, path)


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None
    error: str | None = None
    expected_state: str = ""

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_error(404)
            return
        params = urllib.parse.parse_qs(parsed.query)
        state = (params.get("state") or [""])[0]
        if not secrets.compare_digest(state, _CallbackHandler.expected_state):
            _CallbackHandler.error = "state mismatch"
        elif "error" in params:
            _CallbackHandler.error = params["error"][0]
        else:
            _CallbackHandler.code = (params.get("code") or [""])[0]

        body = (
            b"<title>MembraneClaw</title><h2>Authorized.</h2>"
            b"<p>You can close this tab and return to the terminal.</p>"
            if _CallbackHandler.code
            else b"<title>MembraneClaw</title><h2>Authorization failed.</h2>"
            b"<p>Check the terminal.</p>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass  # the handler's default logging writes to stderr and adds nothing


def login(path: Path = TOKEN_FILE) -> dict[str, Any]:
    """Run the authorization-code + PKCE flow and persist the refresh token."""
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(24)
    url = authorize_url(REDIRECT_URI, state, challenge)

    _CallbackHandler.code = None
    _CallbackHandler.error = None
    _CallbackHandler.expected_state = state

    # Bind before printing the URL, so a port clash fails before the user has gone
    # off to a browser.
    server = http.server.HTTPServer(("127.0.0.1", OAUTH_PORT), _CallbackHandler)
    print(f"Open this URL to authorize (redirect URI: {REDIRECT_URI}):\n\n{url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass  # headless is the normal case here; the printed URL is the real path
    print("Waiting for the callback…")
    try:
        server.handle_request()
    finally:
        server.server_close()

    if _CallbackHandler.error or not _CallbackHandler.code:
        raise RuntimeError(f"authorization failed: {_CallbackHandler.error or 'no code'}")

    data = exchange_code(_CallbackHandler.code, verifier, REDIRECT_URI, path)
    print(f"Saved to {path} (0600). Scopes: {' '.join(data['scopes'])}")
    return data


# ---- httpx auth -------------------------------------------------------------


class GoogleAuth(httpx.Auth):
    """Attaches a Google bearer token, refreshing it when it is about to expire.

    async_auth_flow is overridden deliberately. httpx's default runs the *sync*
    auth_flow generator inline on the event loop, so a blocking refresh would stall
    qwen-agent's MCP loop thread — and that thread serves every MCP tool call in the
    process, ro-chem included.
    """

    def __init__(self, path: Path = TOKEN_FILE):
        self.path = path
        self._data = load_token(path)
        self._lock: asyncio.Lock | None = None

    # A lock created at __init__ binds to whichever loop happens to be current;
    # qwen-agent runs MCP on a loop of its own, so bind lazily on first use.
    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _fresh(self) -> bool:
        return bool(self._data.get("access_token")) and (
            float(self._data.get("expires_at", 0)) - EXPIRY_SKEW > time.time()
        )

    async def _refresh(self) -> None:
        async with self._get_lock():
            # Another request may have refreshed while this one waited.
            if self._fresh():
                return
            # Re-read first: `login` in another process, or an earlier run, may have
            # left a usable token on disk.
            self._data = load_token(self.path)
            if self._fresh():
                return
            refresh_token = self._data.get("refresh_token")
            if not refresh_token:
                raise RuntimeError(
                    "no Google refresh token; run `python -m agent.google_oauth login`"
                )
            client_id, client_secret = client_credentials()
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(TOKEN_ENDPOINT, data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                })
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Google token refresh failed ({resp.status_code}): {resp.text}. "
                    "Re-run `python -m agent.google_oauth login` if the grant was revoked."
                )
            self._data = _store(resp.json(), self.path, refresh_token=refresh_token)
            logger.info("refreshed Google access token")

    async def async_auth_flow(self, request: httpx.Request) -> Any:
        if not self._fresh():
            await self._refresh()
        request.headers["Authorization"] = f"Bearer {self._data['access_token']}"
        yield request

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, Any, None]:
        # Only reached if something drives this synchronously; the MCP client does
        # not. Use whatever is on disk rather than blocking on a refresh here.
        if not self._fresh():
            self._data = load_token(self.path)
        if not self._data.get("access_token"):
            raise RuntimeError(
                "no Google access token; run `python -m agent.google_oauth login`"
            )
        request.headers["Authorization"] = f"Bearer {self._data['access_token']}"
        yield request


def _status(path: Path = TOKEN_FILE) -> int:
    data = load_token(path)
    if not data.get("refresh_token"):
        print(f"Not authorized. Run: python -m agent.google_oauth login  ({path})")
        return 1
    remaining = float(data.get("expires_at", 0)) - time.time()
    state = f"valid for {remaining / 60:.0f} min" if remaining > 0 else "expired (will refresh)"
    print(f"Authorized. Token file: {path}")
    print(f"Access token: {state}")
    print("Scopes:\n  " + "\n  ".join(data.get("scopes", [])))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "login":
        login()
    elif cmd == "status":
        sys.exit(_status())
    else:
        print("usage: python -m agent.google_oauth [login|status]")
        sys.exit(2)
