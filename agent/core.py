import logging
import os
from typing import Iterable, List, Optional

from qwen_agent.agents import Assistant

from agent import compat, config
from agent.tools import builtin  # noqa: F401  (registers tools by import)

compat.apply()

logger = logging.getLogger("membraneclaw")

DEFAULT_TOOLS = ["calculator", "http_get", "now"]

# WaterTAP lives in its own venv (heavy Pyomo/IDAES stack), so the RO model is
# reached over MCP stdio rather than imported. Set RO_MCP=off to skip it.
_RO_PY = config.ROOT / ".venv-watertap/bin/python"
_RO_SERVER = config.ROOT / "mcp_watertap/server.py"
RO_MCP_ENABLED = (
    os.environ.get("RO_MCP", "on").lower() not in ("off", "false", "0")
    and _RO_PY.exists()
    and _RO_SERVER.exists()
)

RO_MCP_CONFIG = {
    "mcpServers": {
        "watertap-ro": {
            "command": str(_RO_PY),
            "args": [str(_RO_SERVER)],
        }
    }
}

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
            logger.warning("watertap-ro MCP unavailable, continuing without it: %s", exc)
    return _make(base)


def stream_reply(agent: Assistant, messages: List[dict]) -> Iterable[List[dict]]:
    """Yield successive full-history snapshots of the agent's response."""
    return agent.run(messages=messages)
