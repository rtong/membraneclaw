"""Regression test for Store's refresh-token deletion.

Store.save() used to merge refresh tokens as a pure union of memory and disk,
so a token popped from memory (revoke_token, exchange_refresh_token) came back
on the very next save. Concretely: the OAuth state on temur accumulated 60
non-expiring refresh tokens in two days because rotation never actually removed
the old one. This file exercises the fix directly against the JSON file, with
no network or MCP transport involved.

Run: .venv-watertap/bin/python mcp_watertap/test_oauth_store.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oauth import REFRESH_TTL, Store


def check(label: str, ok: bool) -> bool:
    print(f"  {label:52s} {'ok' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".oauth_state.json"

        print("1. A deleted refresh token stays deleted across save/reload")
        s1 = Store(path)
        s1.clients["claude"] = {"client_id": "claude"}
        s1.refresh["tok-a"] = {"token": "tok-a", "client_id": "claude", "scopes": [], "expires_at": None}
        s1.refresh["tok-b"] = {"token": "tok-b", "client_id": "claude", "scopes": [], "expires_at": None}
        s1.save()

        # Simulate the real failure mode: token rotation pops the old token from
        # memory (exchange_refresh_token / revoke_token), then saves again.
        del s1.refresh["tok-a"]
        s1.save()

        s2 = Store(path)  # fresh process reloading the file
        failures += 0 if check("tok-a absent after reload", "tok-a" not in s2.refresh) else 1
        failures += 0 if check("tok-b still present", "tok-b" in s2.refresh) else 1

        print("\n2. register_client.py can add a client concurrently without losing it")
        s1.clients["watertap"] = {"client_id": "watertap"}  # a second process's write
        raw = path.read_text()
        import json
        on_disk = json.loads(raw)
        on_disk["clients"]["external"] = {"client_id": "external"}
        path.write_text(json.dumps(on_disk))  # simulate register_client.py writing mid-flight

        s1.save()
        s3 = Store(path)
        failures += 0 if check("this process's client survived", "watertap" in s3.clients) else 1
        failures += 0 if check("concurrently-written client survived", "external" in s3.clients) else 1

        print("\n3. REFRESH_TTL is configured (issuance itself is on the provider, not Store)")
        failures += 0 if check(
            f"REFRESH_TTL is positive ({REFRESH_TTL}s)", REFRESH_TTL > 0
        ) else 1

        print("\n4. Expired tokens are dropped on load and on save")
        s4 = Store(path)
        s4.refresh["tok-fresh"] = {
            "token": "tok-fresh", "client_id": "claude", "scopes": [],
            "expires_at": time.time() + 3600,
        }
        s4.refresh["tok-stale"] = {
            "token": "tok-stale", "client_id": "claude", "scopes": [],
            "expires_at": time.time() - 3600,
        }
        s4.save()
        s5 = Store(path)  # _load() applies _drop_expired
        failures += 0 if check("expired token gone after reload", "tok-stale" not in s5.refresh) else 1
        failures += 0 if check("live token survived reload", "tok-fresh" in s5.refresh) else 1

    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
