"""Provider selection boundary. Importing this module never imports Qwen or compat."""
from contextlib import AsyncExitStack, asynccontextmanager
import os
from pathlib import Path
import sys


def build_qwen_agent(**kwargs):
    from agent.core import build_agent
    return build_agent(**kwargs)


def _enabled(name, default="on"):
    return os.environ.get(name, default).lower() not in {"off", "false", "0"}


def mcp_configurations(root):
    calc_url = os.environ.get("RO_CALC_MCP_URL", "").strip()
    token = os.environ.get("MCP_BEARER_TOKEN", "").strip()
    calc_token = os.environ.get("RO_CALC_MCP_TOKEN", "").strip() or token
    result = []
    for namespace, prefix, url, key, command, script, readonly in [
        ("ro-calc", "RO_CALC_MCP", calc_url, calc_token, sys.executable, root / "mcp_rocalc/server.py",
         ("convert_units", "check_limits")),
        ("ro-chem", "RO_MCP", os.environ.get("RO_MCP_URL", "").strip(), token,
         os.environ.get("RO_MCP_PYTHON") or str(Path.home() / "reaktoro-mcp/env/bin/python"),
         root / "mcp_watertap/server.py", ()),
    ]:
        if not _enabled(prefix) or not (url or (Path(command).exists() and script.exists())):
            continue
        if url:
            cfg = {"type": "streamable-http", "url": url}
            if key:
                cfg["headers"] = {"Authorization": f"Bearer {key}"}
        else:
            cfg = {"type": "stdio", "command": command, "args": [str(script)]}
        result.append((namespace, cfg, readonly, (key,) if key else ()))
    return result


@asynccontextmanager
async def runtime_session(*, provider, model, trace_backend="files", trace_dir="./var/traces", prompt,
                          system_instruction="Use the provided tools when they improve accuracy. Do not guess tool results.",
                          no_mcp=False, google_token_path=None, google_allow=None, limits=None, budget=None,
                          transport=None, adapter=None, extra_tools=(), role="solver", tenant="local",
                          resume_episode=None, request_override=None, generation=None, trace_context=None, include_builtin_tools=True):
    from agent import config
    from membraneclaw_runtime.gateway import ModelGateway
    from membraneclaw_runtime.providers.registry import default_registry
    from membraneclaw_runtime.providers.gemini_transport import GeminiTransport
    from membraneclaw_runtime.providers.openai_transport import OpenAITransport
    from membraneclaw_runtime.providers.anthropic_transport import AnthropicTransport
    from membraneclaw_runtime.providers.types import ConfigurationError, ContentBlock, ModelRequest, ModelRoute
    from membraneclaw_runtime.runtime import AgentRuntime
    from membraneclaw_runtime.tools import ToolRegistry
    from membraneclaw_runtime.tool_adapters import builtin_tools, connect_mcp, identity_google_tools, mcp_tools
    from membraneclaw_runtime.tracing import FileTraceStore

    if not model:
        raise ConfigurationError("explicit model ID required (--model or AGENT_MODEL/GEMINI_MODEL_ID)")
    if adapter is None:
        adapter, route = default_registry().create_for_model(provider, model)
    else:
        route = ModelRoute(provider, model, adapter.api_family)
        adapter.capabilities(model)
    if trace_backend != "files":
        raise ConfigurationError("only the durable files backend is configured for the CLI")
    if transport is None:
        factories = {"gemini": GeminiTransport.from_env, "openai": OpenAITransport.from_env, "anthropic": AnthropicTransport.from_env}
        factory = factories.get(provider)
        if factory is None:
            raise ConfigurationError("provider transport is not configured")
        transport = factory()
    async with AsyncExitStack() as stack:
        tools = (builtin_tools() if include_builtin_tools else []) + list(extra_tools)
        credentials = []
        if not no_mcp:
            for namespace, cfg, readonly, secrets in mcp_configurations(config.ROOT):
                session = await stack.enter_async_context(connect_mcp(cfg))
                tools.extend(await mcp_tools(session, namespace, read_only_names=readonly))
                credentials.extend(secrets)
        # Google tools are bound to THIS identity. None creates no Google tools.
        # The observer closes over gateway, which is assigned before a tool can run.
        def register(values):
            gateway.register_authentication_values(values)
        tools.extend(identity_google_tools(google_token_path, allow=google_allow, credential_observer=register))
        registry = ToolRegistry(tools)
        request = request_override or ModelRequest(route, (ContentBlock.text_block(prompt),), system_instruction=system_instruction,
                               tools=registry.definitions(), **(generation or {}))
        # Check all tool schema/provider constraints before creating an episode or calling a model.
        adapter.prepare(request, adapter.initial_state(model))
        if resume_episode is None:
            gateway = await ModelGateway.start(FileTraceStore(trace_dir), adapter, transport, route, budget=budget, role=role, tenant=tenant, context=trace_context)
        else:
            gateway = await ModelGateway.restore(FileTraceStore(trace_dir), resume_episode, adapter, transport)
            previous = gateway.application_state["tools"]
            if previous != registry.document():
                raise ConfigurationError("tool implementation/schema changed; continuation cannot resume")
        gateway.register_authentication_values(credentials)
        yield AgentRuntime(gateway, registry, request, limits=limits)
