"""Compatibility shims for qwen-agent 0.0.34 against vLLM's OpenAI server, and for
its MCP client against OAuth-protected remote servers."""
import asyncio
import concurrent.futures
import json
import logging
import os

import mcp
import mcp.client.streamable_http as _shttp
import qwen_agent.llm.base as _base
import qwen_agent.llm.oai as _oai
import qwen_agent.tools.mcp_manager as _mcp_manager

logger = logging.getLogger("membraneclaw")

_BaseFunctionCall = _oai.FunctionCall
_orig_conv = _base.BaseChatModel._conv_qwen_agent_messages_to_oai
_orig_streamablehttp_client = _shttp.streamablehttp_client
_orig_list_resources = mcp.ClientSession.list_resources
_orig_create_tool_class = _mcp_manager.MCPManager.create_tool_class

#: Seconds to wait for one MCP tool call before giving up on it. There is no
#: qwen-agent setting for this: `mcp_manager.ToolClass.call` does a bare
#: `future.result()`, which waits forever. 300s matches the `sse_read_timeout`
#: the manager already logs per server, so this never fires before the transport
#: would have.
MCP_CALL_TIMEOUT = float(os.environ.get("MCP_CALL_TIMEOUT", "300"))

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


def _create_tool_class_with_timeout(
    self, register_name, register_client_id, tool_name, tool_desc, tool_parameters
):
    # mcp_manager.py:280 waits on the tool call with a bare `future.result()` --
    # no timeout, so a call whose result never comes back blocks its caller
    # forever. On 2026-09-08 an ro-chem call did exactly that: the MCP server
    # logged the tool returning `200 OK`, the reply never arrived over the
    # streamable-http session, and the agent sat in `future.result()` for 2h40m.
    #
    # Patching the class rather than the instance because BaseTool declares
    # `__slots__ = ()`. `create_tool_class` builds a fresh ToolClass per tool, so
    # this rebinds one tool's `call` and nothing else's.
    tool = _orig_create_tool_class(
        self, register_name, register_client_id, tool_name, tool_desc, tool_parameters
    )

    def call(self, params, **kwargs) -> str:
        tool_args = json.loads(params) if isinstance(params, str) else params
        manager = _mcp_manager.MCPManager()
        client = manager.clients[register_client_id]
        future = asyncio.run_coroutine_threadsafe(
            client.execute_function(tool_name, tool_args), manager.loop
        )
        try:
            return future.result(timeout=MCP_CALL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            # Cancel is best-effort: the coroutine may already be past the point
            # where the loop can stop it. Either way this caller stops waiting.
            future.cancel()
            logger.warning(
                "MCP tool %s did not return within %.0fs; giving up on it",
                register_name, MCP_CALL_TIMEOUT,
            )
            raise TimeoutError(
                f"{register_name} did not return within {MCP_CALL_TIMEOUT:.0f}s"
            ) from None
        except Exception as exc:
            logger.info("Failed in executing MCP tool: %s", exc)
            raise

    type(tool).call = call
    return tool


def apply() -> None:
    _oai.FunctionCall = _LenientFunctionCall
    _base.BaseChatModel._conv_qwen_agent_messages_to_oai = staticmethod(_conv_with_tool_call_id)
    _shttp.streamablehttp_client = _streamablehttp_client_with_auth
    mcp.ClientSession.list_resources = _list_resources_tolerant
    _mcp_manager.MCPManager.create_tool_class = _create_tool_class_with_timeout
