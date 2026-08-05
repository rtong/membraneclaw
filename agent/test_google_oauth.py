"""Regression tests for agent/google_oauth.py and the tool allowlist in
agent/core.py, with no network involved.

Matches the hand-rolled style of mcp_watertap/test_oauth_store.py rather than
pytest — pytest isn't a dependency of the main .venv (see requirements.txt).

Run: .venv/bin/python agent/test_google_oauth.py
"""
from __future__ import annotations

import asyncio
import stat
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from agent import google_oauth


def check(label: str, ok: bool) -> bool:
    print(f"  {label:60s} {'ok' if ok else 'MISMATCH'}")
    return ok


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient(...).post(...) during token refresh."""

    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, data=None, **kwargs):
        _FakeAsyncClient.calls += 1
        await asyncio.sleep(0)  # yield control, like a real request would
        return _FakeResponse(200, {
            "access_token": f"tok-{_FakeAsyncClient.calls}",
            "expires_in": 3600,
            "scope": " ".join(google_oauth.SCOPES),
        })


def test_token_store_roundtrip(tmp: Path) -> int:
    print("1. Token store round-trips and lands 0600")
    path = tmp / "token.json"
    failures = 0

    data = google_oauth._store(
        {"access_token": "a1", "expires_in": 3600, "refresh_token": "r1", "scope": "x y"},
        path,
    )
    failures += 0 if check("access_token stored", data["access_token"] == "a1") else 1
    failures += 0 if check("refresh_token stored", data["refresh_token"] == "r1") else 1

    reloaded = google_oauth.load_token(path)
    failures += 0 if check("reloaded matches", reloaded["access_token"] == "a1") else 1
    mode = stat.S_IMODE(path.stat().st_mode)
    failures += 0 if check(f"file mode is 0600 (got {oct(mode)})", mode == 0o600) else 1

    # A refresh-token grant response carries no new refresh_token; the old one
    # must be carried forward rather than blanked (google_oauth.py:107-109).
    google_oauth._store({"access_token": "a2", "expires_in": 3600}, path, refresh_token="r1")
    still_there = google_oauth.load_token(path)
    failures += 0 if check("refresh_token survives a refresh grant", still_there["refresh_token"] == "r1") else 1

    failures += 0 if check("have_token() true with a refresh_token on disk", google_oauth.have_token(path)) else 1
    failures += 0 if check("have_token() false for a missing file", not google_oauth.have_token(tmp / "nope.json")) else 1
    return failures


def test_refresh_gating(tmp: Path) -> int:
    print("\n2. _fresh() gates on expires_at, not just presence of a token")
    path = tmp / "token.json"
    failures = 0

    google_oauth._store({"access_token": "a1", "expires_in": 3600, "refresh_token": "r1"}, path)
    auth = google_oauth.GoogleAuth(path)
    failures += 0 if check("fresh right after storing (well outside skew)", auth._fresh()) else 1

    auth._data["expires_at"] = time.time() + google_oauth.EXPIRY_SKEW - 1
    failures += 0 if check("not fresh once inside the expiry skew", not auth._fresh()) else 1

    auth._data["expires_at"] = time.time() - 10
    failures += 0 if check("not fresh once actually expired", not auth._fresh()) else 1
    return failures


def test_concurrent_refresh_mints_one_token(tmp: Path) -> int:
    print("\n3. Concurrent async_auth_flow calls under one token refresh")
    path = tmp / "token.json"
    failures = 0

    google_oauth._store(
        {"access_token": "stale", "expires_in": -1, "refresh_token": "r1"}, path,
    )
    auth = google_oauth.GoogleAuth(path)
    _FakeAsyncClient.calls = 0

    async def one_call():
        gen = auth.async_auth_flow(httpx.Request("GET", "https://drivemcp.googleapis.com/mcp/v1"))
        req = await gen.__anext__()
        return req.headers["Authorization"]

    async def run_all():
        return await asyncio.gather(*(one_call() for _ in range(5)))

    with mock.patch.object(google_oauth, "client_credentials", return_value=("cid", "secret")), \
         mock.patch.object(httpx, "AsyncClient", _FakeAsyncClient):
        results = asyncio.run(run_all())

    failures += 0 if check("exactly one token refresh for 5 concurrent calls", _FakeAsyncClient.calls == 1) else 1
    failures += 0 if check("all 5 calls got the same bearer token", len(set(results)) == 1) else 1
    return failures


def _filter_tools(tools, allow):
    # Same expression as the `if GOOGLE_TOOL_ALLOW is not None:` branch in
    # agent/core.py:_google_mcp_tools — kept in sync deliberately rather than
    # imported, since importing agent.core would pull in MCPManager's network path.
    if allow is not None:
        tools = [t for t in tools if t.name.split("-", 1)[1] in allow]
    return tools


def test_tool_allowlist_filtering() -> int:
    print("\n4. Tool allowlist keeps named tools and drops the rest")
    failures = 0

    class _FakeTool:
        def __init__(self, name):
            self.name = name

    tools = [_FakeTool(f"drive-{n}") for n in (
        "search_files", "get_file_metadata", "read_file_content", "create_file",
        "copy_file", "get_file_permissions", "list_recent_files", "download_file_content",
    )]
    allow = {"search_files", "get_file_metadata", "read_file_content", "create_file"}

    kept = _filter_tools(tools, allow)
    kept_names = {t.name.split("-", 1)[1] for t in kept}
    failures += 0 if check("allow-listed tools kept", kept_names == allow) else 1
    failures += 0 if check("everything else dropped", len(kept) == len(allow)) else 1

    # allow=None (GOOGLE_MCP_TOOLS=* or "all") must keep everything.
    kept_all = _filter_tools(tools, None)
    failures += 0 if check("allow=None keeps every tool (open-all opt-out)", len(kept_all) == len(tools)) else 1
    return failures


def main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        failures += test_token_store_roundtrip(tmp)
        failures += test_refresh_gating(tmp)
        failures += test_concurrent_refresh_mints_one_token(tmp)
    failures += test_tool_allowlist_filtering()

    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
