"""Who is making this request?

`server.py` is called by Open WebUI's *backend*, not the browser, over a single
shared `AGENT_API_KEY` — so by default every request looks identical and there is
no way to tell two users apart. That is fine while the only per-user state is the
conversation (which the client passes in), but Google credentials are per person:
without identity, one user's Drive would be readable by everyone.

Open WebUI can forward the signed identity of the person who typed the message.
With `ENABLE_FORWARD_USER_INFO_HEADERS=true` and `FORWARD_USER_INFO_HEADER_JWT_SECRET`
set, it mints a short-lived HS256 JWT per request and sends it as
`X-OpenWebUI-User-Jwt` (open_webui/utils/headers.py::include_user_info_headers).
Because it is signed with a secret only Open WebUI and this process share, a
forged header from anything else on the box fails verification.

Fail closed, always: every path that cannot prove who is asking returns None, and
the caller must then offer *no* Google tools rather than falling back to the
owner's credentials. `token_path_for` in agent/google_oauth.py is the other half —
it maps a verified id to that user's own token file, and there is deliberately no
code path from an unverified request to the legacy single-user token.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

import jwt

logger = logging.getLogger("membraneclaw")

# Names match Open WebUI's defaults (backend/open_webui/env.py:878-896). Overriding
# them there means overriding them here too.
JWT_HEADER = os.environ.get("FORWARD_USER_INFO_HEADER_JWT", "X-OpenWebUI-User-Jwt")
JWT_SECRET = (os.environ.get("FORWARD_USER_INFO_HEADER_JWT_SECRET") or "").strip()
JWT_ISSUER = "open-webui"

# Open WebUI ids are UUIDs, but this value picks a filename in
# google_oauth.token_path_for, so treat it as untrusted input and allow only
# characters that cannot walk out of the directory.
_SAFE_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
_MAX_ID_LEN = 128


@dataclass(frozen=True)
class User:
    id: str
    email: str = ""
    name: str = ""
    role: str = ""


def enabled() -> bool:
    """True when Open WebUI is configured to sign identities we can verify."""
    return bool(JWT_SECRET)


def safe_id(user_id: str) -> bool:
    return (
        bool(user_id)
        and len(user_id) <= _MAX_ID_LEN
        and set(user_id) <= _SAFE_ID
    )


def from_headers(headers: Mapping[str, str]) -> Optional[User]:
    """Verify the forwarded identity JWT, or return None.

    None means "no verified user" for every reason — feature off, header absent,
    signature bad, expired, malformed id. Callers must not distinguish them when
    deciding whether to expose credentials; the difference is only for the log.
    """
    if not JWT_SECRET:
        return None
    token = headers.get(JWT_HEADER) or headers.get(JWT_HEADER.lower())
    if not token:
        return None
    try:
        claims = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],          # never trust the token's own `alg`
            issuer=JWT_ISSUER,
            options={"require": ["exp", "sub", "iss"]},
        )
    except jwt.InvalidTokenError as exc:
        # Expiry is routine if a request is retried late; anything else means the
        # secret is mismatched or someone is guessing. Both are worth seeing.
        logger.warning("rejected forwarded identity: %s", exc)
        return None

    user_id = str(claims.get("sub") or "")
    if not safe_id(user_id):
        logger.warning("rejected forwarded identity: unusable sub %r", user_id)
        return None
    return User(
        id=user_id,
        email=str(claims.get("email") or ""),
        name=str(claims.get("name") or ""),
        role=str(claims.get("role") or ""),
    )
