"""Adapters reuse existing implementations and full native MCP results."""
from __future__ import annotations

import asyncio
import copy
import json
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from dataclasses import dataclass
from .tracing.schema import digest

from .providers.types import ConfigurationError
from .tools import ToolFailure, ToolSpec


def builtin_tools():
    from agent.tools import builtin_impl
    version = "builtin_v1:" + digest(Path(builtin_impl.__file__).read_bytes())
    # Implementations are shared with Qwen wrappers, with no runtime truncation.
    def checked(function, args):
        value = function(args)
        if isinstance(value, str) and value.startswith("Error:"):
            raise ToolFailure(value)
        return value
    return [
        ToolSpec("calculator", "Evaluate an arithmetic expression.", {
            "type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            lambda a: checked(lambda x: builtin_impl.calculator(x["expression"]), a), version, True, True),
        ToolSpec("http_get", "Fetch the full body of a public HTTP(S) resource.", {
            "type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            lambda a: checked(lambda x: builtin_impl.http_get(x["url"]), a), version, True, False),
    ]


def identity_google_tools(token_path, login_url=None, *, allow=None, credential_observer=None):
    if token_path is None:
        return []  # Never fall back to the operator's token for an unidentified caller.
    from agent.tools import google_impl
    google_tools = google_impl.google_tools
    version = "google_rest_v1:" + digest(Path(google_impl.__file__).read_bytes())
    tools = google_tools(Path(token_path), login_url, allow, max_chars=None,
                         credential_observer=credential_observer)
    result = []
    for tool in tools:
        read_only = tool.name not in {"google_sheets_update_values", "google_disconnect"}
        def invoke(arguments, t=tool):
            value = t.call(json.dumps(arguments))
            if getattr(t, "outcome_unknown", False) or value.startswith("Removed local access, but"):
                raise RuntimeError("google_tool_outcome_unknown")
            if value.startswith("Error:"):
                raise ToolFailure(value)
            return value
        result.append(ToolSpec(tool.name, tool.description, tool.parameters,
                               invoke, version, read_only, False))
    return result


async def mcp_tools(session, namespace, *, read_only_names=(), version="mcp_v1"):
    response = await session.list_tools()
    tools, names = [], set()
    for item in response.tools:
        if item.name in names:
            raise ConfigurationError("duplicate MCP tool name")
        names.add(item.name)
        async def invoke(arguments, name=item.name):
            response = await session.call_tool(name, arguments)
            # Full structured/native result, not just text content and no UI clipping.
            return response.model_dump(mode="json", by_alias=True, exclude_none=False)
        tools.append(ToolSpec(namespace + ":" + item.name, item.description or "", copy.deepcopy(item.inputSchema),
                              invoke, version, item.name in read_only_names, item.name in read_only_names,
                              {"server_info": getattr(session, "server_info", None)}))
    if getattr(response, "nextCursor", None):
        raise ConfigurationError("paginated MCP discovery is not supported; supply a bounded server tool set")
    return tools


@dataclass
class McpConnection:
    session: object
    server_info: dict

    async def list_tools(self):
        return await self.session.list_tools()

    async def call_tool(self, name, arguments):
        return await self.session.call_tool(name, arguments)


def _mcp_http_client(headers=None, timeout=None, auth=None):
    import httpx
    return httpx.AsyncClient(headers=headers, timeout=timeout or httpx.Timeout(30),
                             auth=auth, follow_redirects=False)


@asynccontextmanager
async def connect_mcp(config):
    """One explicit stdio or HTTP server. No Qwen patches, credential fallback or reconnect retry."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    async with AsyncExitStack() as stack:
        if config["type"] == "stdio":
            params = StdioServerParameters(command=config["command"], args=config.get("args", []),
                                            env=config.get("env"))
            read, write = await stack.enter_async_context(stdio_client(params))
        elif config["type"] == "streamable-http":
            read, write, _ = await stack.enter_async_context(streamablehttp_client(
                config["url"], headers=config.get("headers"), httpx_client_factory=_mcp_http_client))
        else:
            raise ConfigurationError("unsupported MCP transport")
        session = await stack.enter_async_context(ClientSession(read, write))
        initialized = await session.initialize()
        info = initialized.serverInfo
        yield McpConnection(session, {"name": info.name, "version": info.version})
