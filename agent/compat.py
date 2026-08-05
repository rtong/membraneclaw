"""Compatibility shims for qwen-agent 0.0.34 against vLLM's OpenAI server, and for
its MCP client against OAuth-protected remote servers."""
import asyncio

import mcp
import mcp.client.streamable_http as _shttp
import qwen_agent.llm.base as _base
import qwen_agent.llm.oai as _oai

_BaseFunctionCall = _oai.FunctionCall
_orig_conv = _base.BaseChatModel._conv_qwen_agent_messages_to_oai
_orig_streamablehttp_client = _shttp.streamablehttp_client
_orig_list_resources = mcp.ClientSession.list_resources

# Hosts whose MCP endpoints authenticate with a refreshing Google bearer token.
# Google's official Workspace servers are all <product>mcp.googleapis.com, except
# People which is people.googleapis.com/mcp/v1.
_GOOGLE_MCP_HOSTS = ("mcp.googleapis.com", "people.googleapis.com")
_google_auth = None


class _LenientFunctionCall(_BaseFunctionCall):
    # vLLM's first streaming tool_call delta carries name/arguments as None, but
    # qwen_agent.llm.oai builds a FunctionCall from them directly and its pydantic
    # model requires str. Coerce so the deltas can accumulate normally.
    def __init__(self, name=None, arguments=None, **kwargs):
        super().__init__(name=name or "", arguments=arguments or "", **kwargs)


def _conv_with_tool_call_id(messages):
    # qwen_agent labels the tool result with `id`, but the OpenAI schema (and the
    # Qwen chat template) key results off `tool_call_id`. Without it the model
    # never sees the result and returns an empty follow-up turn.
    converted = _orig_conv(messages)
    for msg in converted:
        if msg.get("role") == "tool":
            msg["tool_call_id"] = msg.pop("id", "1")
    return converted


def _streamablehttp_client_with_auth(url, *args, **kwargs):
    # qwen_agent.tools.mcp_manager only ever passes *static* headers, which cannot
    # carry a token that expires hourly. mcp's own client already accepts an
    # httpx.Auth per request — this routes ours in for Google's hosts and leaves
    # every other server (ro-chem) on the untouched path.
    #
    # MCPClient.connection_server imports streamablehttp_client inside the function
    # body, so replacing the module attribute is picked up at call time.
    from urllib.parse import urlparse

    global _google_auth
    host = (urlparse(url).hostname or "").lower()
    if kwargs.get("auth") is None and host.endswith(_GOOGLE_MCP_HOSTS):
        from agent.google_oauth import GoogleAuth

        if _google_auth is None:
            # One instance across servers: Drive and Sheets share a token, so this
            # also means one refresh rather than one per connection.
            _google_auth = GoogleAuth()
        kwargs["auth"] = _google_auth
    return _orig_streamablehttp_client(url, *args, **kwargs)


async def _list_resources_tolerant(self, *args, **kwargs):
    # mcp_manager.py:379-385 calls this to probe whether a server has resources,
    # wrapped in `except Exception: pass` because most servers don't implement it.
    # Google's official Workspace MCP servers (drivemcp.googleapis.com et al.)
    # return HTTP 400 for resources/list, and the streamable-http transport's
    # background task group turns that into an asyncio.CancelledError rather than
    # a normal McpError. CancelledError is a BaseException, not an Exception, so it
    # skips straight past that guard and kills the *whole* MCP connection — Drive's
    # tools/list already succeeded by that point, but the process throws it away.
    # Re-raise as a plain exception so the existing guard actually catches it.
    try:
        return await _orig_list_resources(self, *args, **kwargs)
    except asyncio.CancelledError as exc:
        raise RuntimeError(f"resources/list not supported by this server: {exc}") from exc


def apply() -> None:
    _oai.FunctionCall = _LenientFunctionCall
    _base.BaseChatModel._conv_qwen_agent_messages_to_oai = staticmethod(_conv_with_tool_call_id)
    _shttp.streamablehttp_client = _streamablehttp_client_with_auth
    mcp.ClientSession.list_resources = _list_resources_tolerant
