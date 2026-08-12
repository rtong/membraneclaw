"""Tests for the Drive/Sheets tools and the browser login flow.

No network: every Google call is intercepted at httpx.request. What is being
checked is the wiring around the API — id extraction, range encoding, per-user
token binding, the not-authorized path, and the single-use login state.

Run: .venv/bin/python agent/test_google_tools.py
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SECRET = "test-secret-for-google-login-flow"
os.environ["FORWARD_USER_INFO_HEADER_JWT_SECRET"] = SECRET
os.environ["GOOGLE_OAUTH_PUBLIC_REDIRECT"] = "https://example.test/google/callback"

import httpx

from agent import google_oauth
from agent.tools import google as gtools


def check(label: str, ok: bool) -> bool:
    print(f"  {label:62s} {'ok' if ok else 'MISMATCH'}")
    return ok


class _Recorder:
    """Captures the request the tool would have made and replays a canned reply."""

    def __init__(self, status=200, payload=None, text=None):
        self.status, self.payload, self.text = status, payload, text
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return httpx.Response(
            self.status,
            json=self.payload if self.payload is not None else None,
            text=self.text if self.payload is None else None,
            request=httpx.Request(method, url),
        )


def _tool(cls, tmp: Path, login_url=None):
    path = tmp / "tok.json"
    google_oauth._store(
        {"access_token": "at", "expires_in": 3600, "refresh_token": "rt"}, path,
    )
    return cls(path, login_url), path


def test_id_extraction() -> int:
    print("1. Ids are pulled out of the URLs users actually paste")
    failures = 0
    sid = "1r4Gxqb8I5vDhtGvlVSrw27vdTGTG5q4ZQnc5NYJSV2c"
    cases = {
        f"https://docs.google.com/spreadsheets/d/{sid}/edit?gid=0#gid=0": sid,
        f"https://docs.google.com/document/d/{sid}/edit": sid,
        f"https://drive.google.com/file/d/{sid}/view?usp=sharing": sid,
        sid: sid,
        f"  {sid}  ": sid,
    }
    for raw, want in cases.items():
        failures += 0 if check(f"{raw[:44]:44s}", gtools.extract_id(raw) == want) else 1
    return failures


def test_range_encoding() -> int:
    print("\n2. A1 ranges survive becoming a URL path segment")
    failures = 0
    cases = {
        "Sheet1!A1:D20": "Sheet1%21A1%3AD20",
        "My Sheet!A1": "My%20Sheet%21A1",
        "a/b!A1": "a%2Fb%21A1",
    }
    for raw, want in cases.items():
        failures += 0 if check(f"{raw!r} -> encoded", gtools._quote_range(raw) == want) else 1
    return failures


def test_sheets_get_values(tmp: Path) -> int:
    print("\n3. get_values reads the right URL and returns the cells")
    failures = 0
    tool, _ = _tool(gtools.SheetsGetValues, tmp)
    rec = _Recorder(payload={"values": [["a", "b"], ["1", "2"]]})

    with mock.patch.object(httpx, "request", rec):
        out = tool.call('{"spreadsheet_id": "https://docs.google.com/spreadsheets/d/ABCDEFGHIJKLMNOPQRSTUV/edit", "range": "Sheet1!A1:B2"}')

    failures += 0 if check("values came back", '"a"' in out and '"2"' in out) else 1
    url = rec.calls[0]["url"]
    failures += 0 if check("id taken from the URL", "ABCDEFGHIJKLMNOPQRSTUV" in url) else 1
    failures += 0 if check("range percent-encoded in path", "Sheet1%21A1%3AB2" in url) else 1
    failures += 0 if check(
        "bearer token attached",
        rec.calls[0]["headers"]["Authorization"] == "Bearer at",
    ) else 1
    return failures


def test_missing_range_falls_back_to_first_tab(tmp: Path) -> int:
    print("\n4. With no range, the first tab is read")
    failures = 0
    tool, _ = _tool(gtools.SheetsGetValues, tmp)

    class TwoStep:
        def __init__(self):
            self.calls = []

        def __call__(self, method, url, **kwargs):
            self.calls.append(url)
            body = (
                {"sheets": [{"properties": {"title": "First Tab"}}]}
                if "/values/" not in url else {"values": [["x"]]}
            )
            return httpx.Response(200, json=body, request=httpx.Request(method, url))

    rec = TwoStep()
    with mock.patch.object(httpx, "request", rec):
        out = tool.call('{"spreadsheet_id": "ABCDEFGHIJKLMNOPQRSTUV"}')

    failures += 0 if check("asked for the tab list first", "/values/" not in rec.calls[0]) else 1
    failures += 0 if check("then read that tab", "First%20Tab" in rec.calls[1]) else 1
    failures += 0 if check("returned the cells", '"x"' in out) else 1
    return failures


def test_not_authorized_returns_login_link(tmp: Path) -> int:
    print("\n5. A user with no token gets a login link, not a stack trace")
    failures = 0
    empty = tmp / "absent.json"
    tool = gtools.SheetsGetValues(empty, lambda: "https://example.test/login?state=abc")

    out = tool.call('{"spreadsheet_id": "ABCDEFGHIJKLMNOPQRSTUV"}')
    failures += 0 if check("mentions connecting", "Not connected" in out) else 1
    failures += 0 if check("carries the link", "https://example.test/login?state=abc" in out) else 1

    bare = gtools.SheetsGetValues(empty, None)
    out2 = bare.call('{"spreadsheet_id": "ABCDEFGHIJKLMNOPQRSTUV"}')
    failures += 0 if check("degrades sanely with no link configured", "Not connected" in out2) else 1
    return failures


def test_errors_are_readable(tmp: Path) -> int:
    print("\n6. Google's errors surface as text the model can act on")
    failures = 0
    tool, _ = _tool(gtools.SheetsInfo, tmp)

    for status, expect in ((403, "access denied"), (404, "not found"), (500, "500")):
        rec = _Recorder(status=status, payload={"error": {"message": "boom"}})
        with mock.patch.object(httpx, "request", rec):
            out = tool.call('{"spreadsheet_id": "ABCDEFGHIJKLMNOPQRSTUV"}')
        failures += 0 if check(f"HTTP {status} -> {expect!r}", expect in out.lower()) else 1
    return failures


class _PostRecorder:
    """Like _Recorder, but for the httpx.post(url, ...) call shape used by revoke()."""

    def __init__(self, status=200, payload=None, text=None):
        self.status, self.payload, self.text = status, payload, text
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return httpx.Response(
            self.status,
            json=self.payload if self.payload is not None else None,
            text=self.text if self.payload is None else None,
            request=httpx.Request("POST", url),
        )


def test_revoke(tmp: Path) -> int:
    print("\n7. revoke() calls Google, deletes the file, and degrades safely")
    failures = 0

    print("  with a refresh token: revokes at Google, deletes the file")
    p = tmp / "revoke-ok.json"
    google_oauth._store({"access_token": "a", "expires_in": 3600, "refresh_token": "r"}, p)
    rec = _PostRecorder(status=200, payload={})
    with mock.patch.object(httpx, "post", rec):
        ok = google_oauth.revoke(p)
    failures += 0 if check("posted to REVOKE_ENDPOINT", rec.calls[0]["url"] == google_oauth.REVOKE_ENDPOINT) else 1
    failures += 0 if check("token sent in the request", rec.calls[0]["data"]["token"] == "r") else 1
    failures += 0 if check("returns True on 200", ok is True) else 1
    failures += 0 if check("file deleted", not p.exists()) else 1

    print("  Google rejects the revoke: file is still deleted")
    p2 = tmp / "revoke-fail.json"
    google_oauth._store({"access_token": "a", "expires_in": 3600, "refresh_token": "r"}, p2)
    rec2 = _PostRecorder(status=400, payload={"error": "invalid_token"})
    with mock.patch.object(httpx, "post", rec2):
        ok2 = google_oauth.revoke(p2)
    failures += 0 if check("returns False on non-200", ok2 is False) else 1
    failures += 0 if check("file deleted even so", not p2.exists()) else 1

    print("  no token file at all: no HTTP call, no crash")
    p3 = tmp / "never-existed.json"
    rec3 = _PostRecorder(status=200, payload={})
    with mock.patch.object(httpx, "post", rec3):
        ok3 = google_oauth.revoke(p3)
    failures += 0 if check("no HTTP call made", len(rec3.calls) == 0) else 1
    failures += 0 if check("returns True (nothing to revoke)", ok3 is True) else 1
    return failures


def test_disconnect_tool(tmp: Path) -> int:
    print("\n8. google_disconnect tool")
    failures = 0

    print("  already disconnected: short-circuits, never calls revoke")
    p = tmp / "already-gone.json"
    tool = gtools.GoogleDisconnect(p, None)
    with mock.patch.object(gtools.google_oauth, "revoke") as fake_revoke:
        out = tool.call("{}")
    failures += 0 if check("says already disconnected", "Already disconnected" in out) else 1
    failures += 0 if check("revoke() not called", not fake_revoke.called) else 1

    print("  connected, revoke succeeds")
    p2 = tmp / "connected.json"
    google_oauth._store({"access_token": "a", "expires_in": 3600, "refresh_token": "r"}, p2)
    tool2 = gtools.GoogleDisconnect(p2, None)
    with mock.patch.object(gtools.google_oauth, "revoke", return_value=True):
        out2 = tool2.call("{}")
    failures += 0 if check("reports disconnected", "Disconnected" in out2 and "revoked" in out2) else 1

    print("  connected, Google's revoke call fails")
    p3 = tmp / "connected2.json"
    google_oauth._store({"access_token": "a", "expires_in": 3600, "refresh_token": "r"}, p3)
    tool3 = gtools.GoogleDisconnect(p3, None)
    with mock.patch.object(gtools.google_oauth, "revoke", return_value=False):
        out3 = tool3.call("{}")
    failures += 0 if check("still reports local removal", "Removed local access" in out3) else 1
    return failures


def test_tools_are_bound_per_user(tmp: Path) -> int:
    print("\n9. Two users' tools carry two different tokens")
    failures = 0
    p1, p2 = tmp / "u1.json", tmp / "u2.json"
    google_oauth._store({"access_token": "token-one", "expires_in": 3600, "refresh_token": "r"}, p1)
    google_oauth._store({"access_token": "token-two", "expires_in": 3600, "refresh_token": "r"}, p2)

    seen = []
    def rec(method, url, **kwargs):
        seen.append(kwargs["headers"]["Authorization"])
        return httpx.Response(200, json={"values": [["v"]]}, request=httpx.Request(method, url))

    with mock.patch.object(httpx, "request", rec):
        gtools.SheetsGetValues(p1, None).call('{"spreadsheet_id":"ABCDEFGHIJKLMNOPQRSTUV","range":"A1"}')
        gtools.SheetsGetValues(p2, None).call('{"spreadsheet_id":"ABCDEFGHIJKLMNOPQRSTUV","range":"A1"}')

    failures += 0 if check("first user's token used", seen[0] == "Bearer token-one") else 1
    failures += 0 if check("second user's token used", seen[1] == "Bearer token-two") else 1
    failures += 0 if check("no crossover", seen[0] != seen[1]) else 1
    return failures


def test_login_state_is_single_use_and_bound() -> int:
    print("\n10. Login state is signed, single-use and bound to one user")
    import agent.google_login as gl
    importlib.reload(gl)
    failures = 0

    failures += 0 if check("flow reports configured", gl.configured()) else 1

    state = gl.mint_state("user-one", "verifier-one")
    got = gl.consume_state(state)
    failures += 0 if check("first use resolves to the right user", got == ("user-one", "verifier-one")) else 1
    failures += 0 if check("second use is refused", gl.consume_state(state) is None) else 1

    failures += 0 if check("garbage refused", gl.consume_state("nonsense") is None) else 1
    failures += 0 if check("empty refused", gl.consume_state("") is None) else 1

    # A state signed with the identity secret itself must not verify: the login key
    # is derived, so the two purposes cannot be swapped.
    import jwt
    now = int(time.time())
    forged = jwt.encode(
        {"sub": "attacker", "jti": "x", "purpose": "google-login", "iat": now, "exp": now + 600},
        SECRET, algorithm="HS256",
    )
    failures += 0 if check("state signed with the raw identity secret refused",
                           gl.consume_state(forged) is None) else 1

    # Expiry is enforced by the signature itself, so build a state that is already
    # past its exp and confirm it is refused even though its _pending record exists.
    gl._pending["stale-jti"] = {
        "user_id": "user-two", "verifier": "v", "expires_at": time.time() + 600,
    }
    stale = jwt.encode(
        {"sub": "user-two", "jti": "stale-jti", "purpose": "google-login",
         "iat": now - 1200, "exp": now - 600},
        gl._key(), algorithm="HS256",
    )
    failures += 0 if check("expired state refused", gl.consume_state(stale) is None) else 1
    failures += 0 if check(
        "an expired state does not consume its record",
        "stale-jti" in gl._pending,
    ) else 1

    # And the in-memory record expires independently, so a signature that somehow
    # outlived it still cannot be redeemed.
    gl._pending["swept-jti"] = {"user_id": "u", "verifier": "v", "expires_at": time.time() - 1}
    gl.mint_state("someone", "v")  # mint_state sweeps
    failures += 0 if check("stale pending records are swept", "swept-jti" not in gl._pending) else 1
    return failures


def main() -> int:
    import tempfile

    failures = 0
    failures += test_id_extraction()
    failures += test_range_encoding()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        failures += test_sheets_get_values(tmp)
        failures += test_missing_range_falls_back_to_first_tab(tmp)
        failures += test_not_authorized_returns_login_link(tmp)
        failures += test_errors_are_readable(tmp)
        failures += test_revoke(tmp)
        failures += test_disconnect_tool(tmp)
        failures += test_tools_are_bound_per_user(tmp)
    failures += test_login_state_is_single_use_and_bound()
    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
