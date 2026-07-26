from typing import Iterable, List, Optional

from qwen_agent.agents import Assistant

from agent import compat, config
from agent.tools import builtin  # noqa: F401  (registers tools by import)

compat.apply()

DEFAULT_TOOLS = ["calculator", "http_get", "now"]

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
    function_list = list(DEFAULT_TOOLS if tools is None else tools)
    function_list += extra_tools or []
    return Assistant(
        llm=config.llm_config(),
        function_list=function_list,
        system_message=system_message,
        files=files or None,
    )


def stream_reply(agent: Assistant, messages: List[dict]) -> Iterable[List[dict]]:
    """Yield successive full-history snapshots of the agent's response."""
    return agent.run(messages=messages)
