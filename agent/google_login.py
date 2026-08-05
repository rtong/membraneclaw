"""The browser half of per-user Google authorization.

A user chatting through Open WebUI cannot run `agent.google_oauth login` — they are
not on the server, and there is no terminal in a chat window. So the agent hands
them a link instead: the tool reports "not connected", the model relays the URL,
they consent in their own browser, and the callback writes *their* token file.

The link has to carry identity, because by the time the browser arrives it has no
Open WebUI JWT on it — that header only exists on backend-to-backend calls. So the
identity is baked into a signed, short-lived, single-use `state`, minted only in a
request where the JWT already proved who was asking. Anyone replaying an old link
gets a rejected state rather than someone else's account.

The signing key is derived from the Open WebUI shared secret rather than being it,
so a login state can never be mistaken for an identity assertion or vice versa.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from typing import Optional

import jwt

from agent import identity

logger = logging.getLogger("membraneclaw")

# Where Google sends the browser back. Must be registered on the OAuth client
# verbatim, and must be reachable from the user's browser — so for anyone not on
# this machine it is the public URL, not localhost.
PUBLIC_REDIRECT = os.environ.get("GOOGLE_OAUTH_PUBLIC_REDIRECT", "").strip()
STATE_TTL = 600
_PURPOSE = "google-login"


def _key() -> str:
    if not identity.JWT_SECRET:
        return ""
    # Domain separation: same secret, different key, so a token minted for one
    # purpose cannot verify for the other.
    return hashlib.sha256(b"membraneclaw/google-login|" + identity.JWT_SECRET.encode()).hexdigest()


def configured() -> bool:
    return bool(PUBLIC_REDIRECT and _key())


# Verifiers are per-login and short-lived, so memory is the right home: a restart
# invalidates in-flight logins, which is correct — the user just clicks again.
_pending: dict[str, dict] = {}


def _sweep() -> None:
    now = time.time()
    for jti, rec in list(_pending.items()):
        if rec["expires_at"] < now:
            _pending.pop(jti, None)


def mint_state(user_id: str, verifier: str) -> str:
    """Sign a state binding this login attempt to this user and PKCE verifier."""
    _sweep()
    jti = secrets.token_urlsafe(18)
    _pending[jti] = {"user_id": user_id, "verifier": verifier, "expires_at": time.time() + STATE_TTL}
    now = int(time.time())
    return jwt.encode(
        {"sub": user_id, "jti": jti, "purpose": _PURPOSE, "iat": now, "exp": now + STATE_TTL},
        _key(), algorithm="HS256",
    )


def consume_state(state: str) -> Optional[tuple[str, str]]:
    """Verify a returning state and burn it. Returns (user_id, verifier) or None."""
    key = _key()
    if not key or not state:
        return None
    try:
        claims = jwt.decode(
            state, key, algorithms=["HS256"],
            options={"require": ["exp", "sub", "jti"]},
        )
    except jwt.InvalidTokenError as exc:
        logger.warning("rejected google login state: %s", exc)
        return None
    if claims.get("purpose") != _PURPOSE:
        logger.warning("rejected google login state: wrong purpose")
        return None
    # Single use: popping here means a replayed callback finds nothing, even
    # though the signature is still valid until it expires.
    record = _pending.pop(str(claims.get("jti")), None)
    if not record:
        logger.warning("rejected google login state: already used or expired")
        return None
    if record["user_id"] != claims.get("sub"):
        return None
    return record["user_id"], record["verifier"]
