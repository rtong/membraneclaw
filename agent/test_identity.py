"""Tests for forwarded-identity verification and per-user token paths.

The property that matters most here is negative: a request that cannot prove who
sent it must resolve to *no* user, and therefore to no credentials — never to the
owner's token. Every "returns None" case below is that property, approached from a
different angle (feature off, header missing, wrong secret, expired, wrong issuer,
alg confusion, hostile id).

Run: .venv/bin/python agent/test_identity.py
"""
from __future__ import annotations

import importlib
import os
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt

SECRET = "test-secret-do-not-use"


def check(label: str, ok: bool) -> bool:
    print(f"  {label:62s} {'ok' if ok else 'MISMATCH'}")
    return ok


def _identity_with_secret(secret: str):
    """Reimport agent.identity with a given secret, since it reads env at import."""
    if secret:
        os.environ["FORWARD_USER_INFO_HEADER_JWT_SECRET"] = secret
    else:
        os.environ.pop("FORWARD_USER_INFO_HEADER_JWT_SECRET", None)
    import agent.identity as identity
    return importlib.reload(identity)


def _token(secret=SECRET, sub="user-abc", exp_delta=300, issuer="open-webui", alg="HS256", **extra):
    payload = {
        "sub": sub,
        "email": "someone@example.com",
        "name": "Someone",
        "role": "user",
        "iss": issuer,
        "iat": int(time.time()),
        "exp": int(time.time()) + exp_delta,
    }
    payload.update(extra)
    return jwt.encode(payload, secret, algorithm=alg)


def test_happy_path() -> int:
    print("1. A correctly signed identity verifies")
    identity = _identity_with_secret(SECRET)
    failures = 0

    user = identity.from_headers({"X-OpenWebUI-User-Jwt": _token()})
    failures += 0 if check("returns a user", user is not None) else 1
    if user:
        failures += 0 if check("id from `sub`", user.id == "user-abc") else 1
        failures += 0 if check("email carried through", user.email == "someone@example.com") else 1
        failures += 0 if check("role carried through", user.role == "user") else 1

    lower = identity.from_headers({"x-openwebui-user-jwt": _token()})
    failures += 0 if check("header lookup is case-insensitive", lower is not None) else 1
    failures += 0 if check("enabled() true when a secret is set", identity.enabled()) else 1
    return failures


def test_fails_closed() -> int:
    print("\n2. Anything unverifiable resolves to no user (never the owner)")
    identity = _identity_with_secret(SECRET)
    failures = 0

    cases = {
        "no header at all": {},
        "empty header": {"X-OpenWebUI-User-Jwt": ""},
        "not a JWT": {"X-OpenWebUI-User-Jwt": "garbage"},
        "signed with the wrong secret": {"X-OpenWebUI-User-Jwt": _token(secret="wrong-secret")},
        "expired": {"X-OpenWebUI-User-Jwt": _token(exp_delta=-60)},
        "wrong issuer": {"X-OpenWebUI-User-Jwt": _token(issuer="somebody-else")},
    }
    for label, headers in cases.items():
        failures += 0 if check(label, identity.from_headers(headers) is None) else 1

    # alg=none is the classic JWT bypass: a token with no signature at all must not
    # be accepted just because it parses.
    unsigned = jwt.encode(
        {"sub": "attacker", "iss": "open-webui", "exp": int(time.time()) + 300},
        key="", algorithm="none",
    )
    failures += 0 if check(
        "unsigned alg=none token rejected",
        identity.from_headers({"X-OpenWebUI-User-Jwt": unsigned}) is None,
    ) else 1
    return failures


def test_disabled_means_no_user() -> int:
    print("\n3. With no shared secret configured, identity is off entirely")
    identity = _identity_with_secret("")
    failures = 0

    failures += 0 if check("enabled() is false", not identity.enabled()) else 1
    # Even a well-formed token must not be honoured when we have no secret to
    # check it against — otherwise anyone reaching the port could mint identities.
    failures += 0 if check(
        "a valid-looking token is still refused",
        identity.from_headers({"X-OpenWebUI-User-Jwt": _token()}) is None,
    ) else 1
    return failures


def test_hostile_ids_rejected() -> int:
    print("\n4. User ids that could escape the token directory are rejected")
    identity = _identity_with_secret(SECRET)
    failures = 0

    hostile = [
        "../../etc/passwd",
        "..",
        "a/b",
        "a\\b",
        "",
        "x" * 200,
        "with space",
        "nul\x00byte",
    ]
    for bad in hostile:
        ok = identity.from_headers({"X-OpenWebUI-User-Jwt": _token(sub=bad)}) is None
        failures += 0 if check(f"rejected sub={bad[:24]!r}", ok) else 1

    failures += 0 if check("safe_id accepts a normal UUID", identity.safe_id("0f8b-4c2a-9d11")) else 1
    return failures


def test_token_paths_are_per_user() -> int:
    print("\n5. Each user maps to their own token file, 0700 directory")
    _identity_with_secret(SECRET)
    import agent.google_oauth as google_oauth
    importlib.reload(google_oauth)
    failures = 0

    p1 = google_oauth.token_path_for("user-one")
    p2 = google_oauth.token_path_for("user-two")
    failures += 0 if check("two users get different files", p1 != p2) else 1
    failures += 0 if check("path sits inside TOKEN_DIR", p1.parent == google_oauth.TOKEN_DIR) else 1
    failures += 0 if check(
        "neither is the owner's token file",
        google_oauth.TOKEN_FILE not in (p1, p2),
    ) else 1

    mode = stat.S_IMODE(google_oauth.TOKEN_DIR.stat().st_mode)
    failures += 0 if check(f"TOKEN_DIR is 0700 (got {oct(mode)})", mode == 0o700) else 1

    try:
        google_oauth.token_path_for("../escape")
        failures += 0 if check("traversal id raises", False) else 1
    except ValueError:
        failures += 0 if check("traversal id raises", True) else 1
    return failures


def main() -> int:
    failures = 0
    failures += test_happy_path()
    failures += test_fails_closed()
    failures += test_disabled_means_no_user()
    failures += test_hostile_ids_rejected()
    failures += test_token_paths_are_per_user()
    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
