import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from qwen_agent.agents import Assistant
from qwen_agent.tools import MCPManager

from agent import compat, config, google_oauth
from agent.tools import builtin  # noqa: F401  (registers tools by import)
from agent.tools.google import google_tools

compat.apply()

logger = logging.getLogger("membraneclaw")

DEFAULT_TOOLS = ["calculator", "http_get", "now"]

# The RO + chemistry tools run as a separate MCP server (heavy Pyomo/IDAES/
# Reaktoro stack, its own conda environment — Reaktoro is conda-forge only).
# RO_MCP_URL points at a remote streamable-http server; unset, it falls back to
# spawning the local one over stdio. Set RO_MCP=off to skip it entirely.
# RO_MCP_PYTHON overrides the interpreter for that stdio fallback.
_RO_PY = Path(
    os.environ.get("RO_MCP_PYTHON", "")
    or (Path.home() / "reaktoro-mcp/env/bin/python")
)
_RO_SERVER = config.ROOT / "mcp_watertap/server.py"
RO_MCP_URL = os.environ.get("RO_MCP_URL", "").strip()
RO_MCP_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "").strip()

RO_MCP_ENABLED = os.environ.get("RO_MCP", "on").lower() not in ("off", "false", "0") and (
    bool(RO_MCP_URL) or (_RO_PY.exists() and _RO_SERVER.exists())
)

if RO_MCP_URL:
    _ro_server: dict = {"type": "streamable-http", "url": RO_MCP_URL}
    if RO_MCP_TOKEN:
        _ro_server["headers"] = {"Authorization": f"Bearer {RO_MCP_TOKEN}"}
else:
    _ro_server = {"command": str(_RO_PY), "args": [str(_RO_SERVER)]}

# The mcpServers key becomes the tool-name prefix qwen-agent exposes to the
# model (mcp_manager.py: register_name = server_name + '-' + tool.name), so
# "watertap-ro" would undersell the Reaktoro-PSE chemistry tools this server
# also carries. "ro-chem" names the domain — RO hydraulics + the scaling
# chemistry that limits it — rather than either library, so it doesn't need to
# change if either backing library changes again.
RO_MCP_CONFIG = {"mcpServers": {"ro-chem": _ro_server}}

# Google's official Workspace MCP servers are remote streamable-http endpoints
# behind OAuth (developers.google.com/workspace/guides/configure-mcp-servers). The
# bearer token is attached per request by the httpx.Auth that compat.apply()
# threads into mcp's client; nothing about it belongs in the server config here.
GOOGLE_MCP_URLS = {
    "drive": "https://drivemcp.googleapis.com/mcp/v1",
    "sheets": "https://sheetsmcp.googleapis.com/mcp/v1",
    "docs": "https://docsmcp.googleapis.com/mcp/v1",
    "slides": "https://slidesmcp.googleapis.com/mcp/v1",
    "gmail": "https://gmailmcp.googleapis.com/mcp/v1",
    "calendar": "https://calendarmcp.googleapis.com/mcp/v1",
    "chat": "https://chatmcp.googleapis.com/mcp/v1",
    "people": "https://people.googleapis.com/mcp/v1",
}

# Only the scopes for drive+sheets are requested at login (see google_oauth.SCOPES),
# so enabling another server here also means widening SCOPES and re-running login.
GOOGLE_MCP_SERVERS = [
    s.strip().lower()
    for s in os.environ.get("GOOGLE_MCP_SERVERS", "drive,sheets").split(",")
    if s.strip()
]

# Drive declares 8 tools and Sheets 6. With the 3 builtins and 5 ro-chem tools that
# is 22 declared, and this model is measured reliable at <=18 and calls *nothing* at
# 27 (OPERATIONS.md). So the default is a subset that fits the budget rather than
# everything the servers offer; "*" or "all" opts out and will be warned about.
DEFAULT_GOOGLE_TOOLS = (
    "search_files,get_file_metadata,read_file_content,create_file,"
    "get_values,get_spreadsheet,update_values"
)
_google_allow = os.environ.get("GOOGLE_MCP_TOOLS", DEFAULT_GOOGLE_TOOLS).strip()
GOOGLE_TOOL_ALLOW = (
    None if _google_allow in ("*", "all") else {t.strip() for t in _google_allow.split(",") if t.strip()}
)

# Which of the REST Drive/Sheets tools to declare (agent/tools/google.py). Separate
# from GOOGLE_MCP_TOOLS above, which names tools on Google's official MCP servers —
# the two sets share no names, so one allowlist cannot serve both. All five fit the
# budget comfortably, so the default is everything.
_rest_allow = os.environ.get("GOOGLE_TOOLS", "").strip()
GOOGLE_TOOL_ALLOW_REST = (
    None if _rest_allow in ("", "*", "all")
    else {t.strip() for t in _rest_allow.split(",") if t.strip()}
)

# Declared-tool count past which tool selection gets unreliable on this model.
TOOL_COUNT_WARN = int(os.environ.get("AGENT_TOOL_COUNT_WARN", "18"))

GOOGLE_MCP_ENABLED = (
    os.environ.get("GOOGLE_MCP", "on").lower() not in ("off", "false", "0")
    and bool(GOOGLE_MCP_SERVERS)
    # No credential means "not set up", not "broken" — a fresh clone should start.
    and google_oauth.have_token()
)

# mcp_watertap/server.py: 5 @mcp.tool() defs. RO_MCP_CONFIG is handed to Assistant()
# as a raw dict and expanded internally, so its tool count isn't observable here —
# this constant is what makes the declared-count log below accurate.
RO_CHEM_TOOL_COUNT = 5

SYSTEM_MESSAGE = (
    "You are a capable assistant running on a local Qwen3.5 deployment. "
    "Use the provided tools when they give you a more accurate answer than reasoning alone; "
    "otherwise answer directly. When documents are attached, ground your answer in them and "
    "say so if the answer is not present."
)


def _google_mcp_tools() -> List:
    """Connect to each enabled Google server and return the allow-listed BaseTools.

    One MCPManager().initConfig() call per server rather than one call for the whole
    GOOGLE_MCP_SERVERS dict, so a single unreachable service (e.g. Sheets down)
    doesn't cost you the others.
    """
    tools = []
    for name in GOOGLE_MCP_SERVERS:
        url = GOOGLE_MCP_URLS.get(name)
        if not url:
            logger.warning("unknown Google MCP server %r in GOOGLE_MCP_SERVERS, skipping", name)
            continue
        cfg = {"mcpServers": {name: {"type": "streamable-http", "url": url}}}
        try:
            server_tools = MCPManager().initConfig(cfg)
        except Exception as exc:
            logger.warning("google-%s MCP unavailable, continuing without it: %s", name, exc)
            continue
        if GOOGLE_TOOL_ALLOW is not None:
            server_tools = [t for t in server_tools if t.name.split("-", 1)[1] in GOOGLE_TOOL_ALLOW]
        tools.extend(server_tools)
    return tools


def build_agent(
    tools: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    system_message: str = SYSTEM_MESSAGE,
    extra_tools: Optional[List] = None,
    google_token_path: Optional[Path] = None,
    google_login_url=None,
) -> Assistant:
    """extra_tools takes BaseTool instances — used to inject client-supplied tools
    per request without registering them in qwen-agent's global TOOL_REGISTRY.

    google_token_path selects *whose* Google credentials the Google tools act as.
    None means no verified user, and therefore no Google tools at all — never a
    fallback to the owner's token, which on the multi-user server would hand one
    person's Drive to everyone (see agent/identity.py).
    """
    base = list(DEFAULT_TOOLS if tools is None else tools) + list(extra_tools or [])

    # The official Workspace MCP servers authenticate as a single identity per
    # process: compat.py holds one module-global GoogleAuth for every connection,
    # so there is no way to serve two users from one process. They are therefore
    # restricted to the operator's own token — i.e. the CLI. server.py passes a
    # per-user path and so can never reach this branch.
    if GOOGLE_MCP_ENABLED and google_token_path == google_oauth.TOKEN_FILE:
        base += _google_mcp_tools()

    # Drive/Sheets over the plain REST APIs, bound to this user's token. Declared
    # even before they have authorized: the tools answer with a login link, which
    # is how someone in a chat window finds out they need to connect at all.
    if google_token_path is not None:
        base += google_tools(google_token_path, google_login_url, allow=GOOGLE_TOOL_ALLOW_REST)

    declared = len(base) + (RO_CHEM_TOOL_COUNT if RO_MCP_ENABLED else 0)
    if declared > TOOL_COUNT_WARN:
        logger.warning(
            "declaring %d tools to the model, past the measured reliability ceiling of %d",
            declared, TOOL_COUNT_WARN,
        )
    else:
        logger.info("declaring %d tools to the model", declared)

    def _make(function_list):
        return Assistant(
            llm=config.llm_config(),
            function_list=function_list,
            system_message=system_message,
            files=files or None,
        )

    if RO_MCP_ENABLED:
        # The MCP handshake happens inside Assistant(); if the server fails to
        # start, fall back to the remaining tools rather than losing the agent.
        try:
            return _make(base + [RO_MCP_CONFIG])
        except Exception as exc:
            logger.warning("ro-chem MCP unavailable, continuing without it: %s", exc)
    return _make(base)


def stream_reply(agent: Assistant, messages: List[dict]) -> Iterable[List[dict]]:
    """Yield successive full-history snapshots of the agent's response."""
    return agent.run(messages=messages)
