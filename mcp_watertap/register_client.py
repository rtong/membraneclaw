"""Pre-register an OAuth client with fixed credentials.

Claude's connector form has optional OAuth Client ID / Client Secret fields under
Advanced settings. Leaving them blank makes Claude register itself dynamically —
which this server supports — but when a UI insists on values, they have to exist
on the server beforehand. This mints that pair.

OAuth 2.1 requires *exact* redirect-URI matching, so the client is registered with
the connector callbacks below; a URI that isn't listed is rejected at /authorize.

    .venv/bin/python register_client.py claude
    .venv/bin/python register_client.py claude --show
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oauth import Store

# Claude's connector callbacks, plus the Claude Code CLI's fixed local port.
DEFAULT_REDIRECT_URIS = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    "http://localhost:29352/callback",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("client_id", help="Client ID to create, e.g. 'claude'")
    ap.add_argument("--redirect-uri", action="append", default=[],
                    help="Extra redirect URI (repeatable)")
    ap.add_argument("--show", action="store_true", help="Print an existing client instead of creating")
    args = ap.parse_args()

    store = Store()
    existing = store.clients.get(args.client_id)

    if args.show:
        if not existing:
            print(f"no such client: {args.client_id}", file=sys.stderr)
            return 1
        print(f"client_id     : {existing['client_id']}")
        print(f"client_secret : {existing.get('client_secret', '(none)')}")
        print(f"redirect_uris : {existing.get('redirect_uris')}")
        return 0

    if existing:
        print(f"client {args.client_id!r} already exists; use --show to print it", file=sys.stderr)
        return 1

    secret = secrets.token_urlsafe(32)
    store.clients[args.client_id] = {
        "client_id": args.client_id,
        "client_secret": secret,
        "redirect_uris": DEFAULT_REDIRECT_URIS + args.redirect_uri,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        # A secret is issued, so the client must authenticate at /token.
        "token_endpoint_auth_method": "client_secret_post",
        "scope": "watertap",
    }
    store.save()
    print(f"client_id     : {args.client_id}")
    print(f"client_secret : {secret}")
    print(f"redirect_uris : {store.clients[args.client_id]['redirect_uris']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
