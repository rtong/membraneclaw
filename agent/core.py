import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from qwen_agent.agents import Assistant

from agent import compat, config
from agent.tools import builtin  # noqa: F401  (registers tools by import)

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

SYSTEM_MESSAGE = (
    "You are a capable assistant running on a local Qwen3.5 deployment. "
    "Use the provided tools when they give you a more accurate answer than reasoning alone; "
    "otherwise answer directly. When documents are attached, ground your answer in them and "
    "say so if the answer is not present."
)


def build_agent(
    tools: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    system_message: str = SYSTEM_MESSAGE,
    extra_tools: Optional[List] = None,
) -> Assistant:
    """extra_tools takes BaseTool instances — used to inject client-supplied tools
    per request without registering them in qwen-agent's global TOOL_REGISTRY."""
    base = list(DEFAULT_TOOLS if tools is None else tools) + list(extra_tools or [])

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
