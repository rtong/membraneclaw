"""Google Drive and Sheets tools, acting as one specific user.

Deliberately *not* MCP, and deliberately not registered in qwen-agent's global
TOOL_REGISTRY. Both follow from the same requirement: each instance is bound to one
person's credentials, so the tools have to be constructed per user and handed to
the Assistant as instances (the `extra_tools` path in agent/core.py). A registered
tool is a singleton shared by every agent in the process, which is exactly the
cross-user leak this design exists to prevent.

These call the plain REST APIs rather than Google's official Workspace MCP servers
because those servers reject non-Workspace accounts outright — every tool, every
service, `"The caller does not have permission"` — while the same OAuth token works
against `sheets.googleapis.com`/`www.googleapis.com` (see OPERATIONS.md).

A user who has not authorized yet gets a login link back from the tool instead of
an error, so the model can hand it to them mid-conversation.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional

import httpx
import json5
from qwen_agent.tools.base import BaseTool

from agent import google_oauth

DRIVE_API = "https://www.googleapis.com/drive/v3"
SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
TIMEOUT = 30
# Enough for the model to work with, small enough not to blow the context window.
MAX_CHARS = 20_000

# Google Docs/Sheets/Slides have no bytes to download; they must be exported. Plain
# text keeps the payload small and readable for a 9B model.
_EXPORT_AS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

_ID_IN_URL = re.compile(r"/d/([a-zA-Z0-9_-]{20,})")


def extract_id(value: str) -> str:
    """Accept a bare id or any Google URL containing one.

    Models are given URLs by users and paste them straight through, so treating a
    URL as an id is the common case rather than an edge case.
    """
    value = (value or "").strip()
    match = _ID_IN_URL.search(value)
    return match.group(1) if match else value


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n… [truncated]"


def _quote_range(cell_range: str) -> str:
    """Percent-encode an A1 range for use as a URL path segment.

    'Sheet1!A1:D20' carries '!' and ':', and tab names routinely contain spaces or
    '/', all of which change the path if passed through raw.
    """
    return urllib.parse.quote(cell_range, safe="")


class _GoogleTool(BaseTool):
    """Shared auth/HTTP plumbing. Subclasses implement `run`."""

    def __init__(self, token_path: Path, login_url: Optional[Callable[[], str]] = None, cfg=None):
        super().__init__(cfg)
        self.token_path = token_path
        self._login_url = login_url

    def _needs_login(self) -> str:
        url = self._login_url() if self._login_url else ""
        if url:
            return (
                "Not connected to Google yet. Give the user this link to authorize, "
                f"then retry: {url}"
            )
        return (
            "Not connected to Google, and no login link is configured. Run "
            "`python -m agent.google_oauth login` on the server."
        )

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        token = google_oauth.access_token(self.token_path)
        headers = {"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        return httpx.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)

    def _fail(self, resp: httpx.Response) -> str:
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:200]
        if resp.status_code == 403:
            return f"Error: access denied by Google ({detail or 'no detail'})"
        if resp.status_code == 404:
            return "Error: not found, or this account cannot see it"
        return f"Error: Google returned {resp.status_code}: {detail}"

    def call(self, params: str, **kwargs) -> str:
        try:
            args = json5.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return f"Error: could not parse arguments: {exc}"
        try:
            return self.run(args)
        except google_oauth.NotAuthorized:
            return self._needs_login()
        except httpx.HTTPError as exc:
            return f"Error: could not reach Google: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

    def run(self, args: dict) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class DriveSearch(_GoogleTool):
    name = "google_drive_search"
    description = (
        "Search the user's Google Drive by file name or content. "
        "Returns matching files with their ids, names and types."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words to look for in the name or contents"},
            "limit": {"type": "integer", "description": "Max results, default 10"},
        },
        "required": ["query"],
    }

    def run(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Error: query is required"
        limit = max(1, min(int(args.get("limit") or 10), 50))
        # Escaping matters: a stray apostrophe would otherwise end the literal and
        # make Google reject the whole query.
        safe = query.replace("\\", "\\\\").replace("'", "\\'")
        resp = self._request("GET", f"{DRIVE_API}/files", params={
            "q": f"(name contains '{safe}' or fullText contains '{safe}') and trashed = false",
            "fields": "files(id,name,mimeType,modifiedTime,webViewLink)",
            "pageSize": limit,
        })
        if resp.status_code != 200:
            return self._fail(resp)
        files = resp.json().get("files", [])
        if not files:
            return f"No files matching {query!r}."
        return json.dumps(files, indent=2)


class DriveRead(_GoogleTool):
    name = "google_drive_read"
    description = (
        "Read the contents of one Google Drive file as text. "
        "Accepts a file id or a Drive URL. Google Docs are exported as text, Sheets as CSV."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "Drive file id, or a URL containing one"},
        },
        "required": ["file_id"],
    }

    def run(self, args: dict) -> str:
        file_id = extract_id(str(args.get("file_id", "")))
        if not file_id:
            return "Error: file_id is required"
        meta = self._request("GET", f"{DRIVE_API}/files/{file_id}", params={"fields": "name,mimeType"})
        if meta.status_code != 200:
            return self._fail(meta)
        info = meta.json()
        mime = info.get("mimeType", "")

        if mime in _EXPORT_AS:
            resp = self._request(
                "GET", f"{DRIVE_API}/files/{file_id}/export",
                params={"mimeType": _EXPORT_AS[mime]},
            )
        elif mime.startswith(("text/", "application/json")) or mime == "application/xml":
            resp = self._request("GET", f"{DRIVE_API}/files/{file_id}", params={"alt": "media"})
        else:
            return (
                f"{info.get('name')!r} is {mime}, which has no text form. "
                "Only text files and Google Docs/Sheets/Slides can be read."
            )
        if resp.status_code != 200:
            return self._fail(resp)
        return f"# {info.get('name')}\n\n{_clip(resp.text)}"


class SheetsInfo(_GoogleTool):
    name = "google_sheets_info"
    description = (
        "Get a spreadsheet's title and the name, size and id of each of its tabs. "
        "Call this before reading values when the tab names are not known."
    )
    parameters = {
        "type": "object",
        "properties": {
            "spreadsheet_id": {"type": "string", "description": "Spreadsheet id, or a URL containing one"},
        },
        "required": ["spreadsheet_id"],
    }

    def run(self, args: dict) -> str:
        sid = extract_id(str(args.get("spreadsheet_id", "")))
        if not sid:
            return "Error: spreadsheet_id is required"
        resp = self._request("GET", f"{SHEETS_API}/{sid}", params={
            "fields": "properties.title,sheets.properties(sheetId,title,gridProperties)",
        })
        if resp.status_code != 200:
            return self._fail(resp)
        data = resp.json()
        tabs = [{
            "title": s["properties"]["title"],
            "sheetId": s["properties"].get("sheetId"),
            "rows": s["properties"].get("gridProperties", {}).get("rowCount"),
            "columns": s["properties"].get("gridProperties", {}).get("columnCount"),
        } for s in data.get("sheets", [])]
        return json.dumps({"title": data.get("properties", {}).get("title"), "tabs": tabs}, indent=2)


class SheetsGetValues(_GoogleTool):
    name = "google_sheets_get_values"
    description = (
        "Read cell values from a Google Sheet. "
        "Range is A1 notation such as 'Sheet1!A1:D20'; omit it to read the first tab."
    )
    parameters = {
        "type": "object",
        "properties": {
            "spreadsheet_id": {"type": "string", "description": "Spreadsheet id, or a URL containing one"},
            "range": {"type": "string", "description": "A1 notation, e.g. 'Sheet1!A1:D20'"},
        },
        "required": ["spreadsheet_id"],
    }

    def run(self, args: dict) -> str:
        sid = extract_id(str(args.get("spreadsheet_id", "")))
        if not sid:
            return "Error: spreadsheet_id is required"
        cell_range = str(args.get("range") or "").strip()
        if not cell_range:
            # No range given: fall back to the first tab, which is what someone
            # asking "what's in this spreadsheet" almost always means.
            info = self._request("GET", f"{SHEETS_API}/{sid}", params={"fields": "sheets.properties.title"})
            if info.status_code != 200:
                return self._fail(info)
            sheets = info.json().get("sheets", [])
            if not sheets:
                return "Error: this spreadsheet has no tabs"
            cell_range = sheets[0]["properties"]["title"]
        resp = self._request("GET", f"{SHEETS_API}/{sid}/values/{_quote_range(cell_range)}")
        if resp.status_code != 200:
            return self._fail(resp)
        values = resp.json().get("values", [])
        if not values:
            return f"{cell_range} is empty."
        return _clip(json.dumps({"range": cell_range, "values": values}, indent=2))


class SheetsUpdateValues(_GoogleTool):
    name = "google_sheets_update_values"
    description = (
        "Write cell values into a Google Sheet, overwriting what is there. "
        "Values are a list of rows. Confirm with the user before overwriting data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "spreadsheet_id": {"type": "string", "description": "Spreadsheet id, or a URL containing one"},
            "range": {"type": "string", "description": "A1 notation of the top-left target, e.g. 'Sheet1!A1'"},
            "values": {
                "type": "array",
                "description": "Rows to write, each row a list of cell values",
                "items": {"type": "array", "items": {"type": "string"}},
            },
        },
        "required": ["spreadsheet_id", "range", "values"],
    }

    def run(self, args: dict) -> str:
        sid = extract_id(str(args.get("spreadsheet_id", "")))
        cell_range = str(args.get("range") or "").strip()
        values = args.get("values")
        if not sid or not cell_range:
            return "Error: spreadsheet_id and range are required"
        if not isinstance(values, list) or not values:
            return "Error: values must be a non-empty list of rows"
        rows = [v if isinstance(v, list) else [v] for v in values]
        resp = self._request(
            "PUT", f"{SHEETS_API}/{sid}/values/{_quote_range(cell_range)}",
            # USER_ENTERED so '=SUM(A1:A9)' and dates behave as if typed in the UI.
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": rows},
        )
        if resp.status_code != 200:
            return self._fail(resp)
        out = resp.json()
        return (
            f"Updated {out.get('updatedCells', 0)} cells "
            f"({out.get('updatedRows', 0)} rows) in {out.get('updatedRange', cell_range)}."
        )


class GoogleDisconnect(_GoogleTool):
    name = "google_disconnect"
    description = "Disconnect the user's Google account, revoking this app's access to their Drive and Sheets."
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params: str, **kwargs) -> str:
        # Deliberately does not go through _GoogleTool.call's NotAuthorized
        # handling — disconnecting when already disconnected should say so
        # plainly, not hand back a login link.
        if not google_oauth.load_token(self.token_path).get("refresh_token"):
            return "Already disconnected — no Google account is linked."
        ok = google_oauth.revoke(self.token_path)
        if ok:
            return "Disconnected. Google access has been revoked."
        return (
            "Removed local access, but Google's revoke request failed — the app "
            "may still be listed at https://myaccount.google.com/permissions. "
            "Local tools will ask you to reconnect either way."
        )


TOOL_CLASSES = [DriveSearch, DriveRead, SheetsInfo, SheetsGetValues, SheetsUpdateValues, GoogleDisconnect]
TOOL_NAMES = [c.name for c in TOOL_CLASSES]


def google_tools(
    token_path: Path,
    login_url: Optional[Callable[[], str]] = None,
    allow: Optional[set] = None,
) -> List[BaseTool]:
    """Build this user's Google tools, bound to their own credentials."""
    return [
        cls(token_path, login_url)
        for cls in TOOL_CLASSES
        if allow is None or cls.name in allow
    ]
