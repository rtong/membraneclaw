import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.factory import runtime_session
from membraneclaw_runtime import cli
from membraneclaw_runtime.providers.types import ConfigurationError
from membraneclaw_runtime.tool_adapters import builtin_tools, connect_mcp, identity_google_tools, mcp_tools
from membraneclaw_runtime.tools import ToolRegistry

from helpers import FakeAdapter, FakeTransport


class Adapters(unittest.IsolatedAsyncioTestCase):
    async def test_real_local_mcp_units_and_limits_no_duplicate_numerical_logic(self):
        from agent import config
        async with connect_mcp({"type": "stdio", "command": sys.executable,
                                "args": [str(config.ROOT / "mcp_rocalc/server.py")]}) as session:
            tools = await mcp_tools(session, "ro-calc", read_only_names=("convert_units", "check_limits"))
            by_name = {t.name: t for t in tools}
            converted = await by_name["ro-calc:convert_units"].handler({"value": 1, "from_unit": "bar", "to_unit": "kPa"})
            self.assertEqual(json.loads(converted["content"][0]["text"])["value"], 100)
            checked = await by_name["ro-calc:check_limits"].handler({"limits": {"permeate.nacl_ppm": "<= 500 mg/L"},
                "values": {"a": {"permeate.nacl_ppm": 300}}})
            self.assertIn("pass", checked["content"][0]["text"])
            # SDK-generated title/map schemas are preserved and provider-compatible.
            from membraneclaw_runtime.providers.gemini import GeminiAdapter
            from membraneclaw_runtime.providers.types import ContentBlock, ModelRequest
            adapter = GeminiAdapter()
            state = adapter.initial_state("gemini-2.5-flash")
            adapter.prepare(ModelRequest(state.route, (ContentBlock.text_block("fixture"),),
                                          tools=ToolRegistry(tools).definitions()), state)

    async def test_full_mcp_result_and_error_flag_are_preserved(self):
        huge = "x" * 250000
        raw = {"content": [{"type": "text", "text": huge}, {"type": "image", "data": "AQ==", "mimeType": "image/png"}],
               "structuredContent": {"unknown": [1, 2]}, "isError": True}
        class Session:
            async def list_tools(self):
                return SimpleNamespace(tools=[SimpleNamespace(name="read", description="fixture",
                    inputSchema={"type": "object", "properties": {}})], nextCursor=None)
            async def call_tool(self, name, arguments):
                return SimpleNamespace(model_dump=lambda **kwargs: raw)
        tools = await mcp_tools(Session(), "server")
        self.assertEqual(await tools[0].handler({}), raw)

    async def test_factory_uses_reused_calculator_without_qwen(self):
        adapter = FakeAdapter()
        def first(native):
            definitions = json.loads(native.body)["tools"]
            self.assertIn("calculator", {d["name"] for d in definitions})
            return {"calls": [{"name": "calculator", "args": {"expression": "(2+3)*4"}}]}
        transport = FakeTransport([first, {"text": "20"}])
        with tempfile.TemporaryDirectory() as directory:
            async with runtime_session(provider="synthetic", model="fixture", trace_dir=directory, prompt="fixture",
                                       no_mcp=True, adapter=adapter, transport=transport) as runtime:
                result = await runtime.run()
            self.assertEqual(result.text, "20")
            self.assertEqual(len(transport.requests), 2)

    async def test_google_binding_has_no_implicit_operator_fallback(self):
        self.assertEqual(identity_google_tools(None), [])
        tools = identity_google_tools(Path("/synthetic/user-a.json"))
        self.assertEqual({t.name for t in tools}, {"google_drive_read", "google_sheets_get_values", "google_disconnect"})
        self.assertEqual(next(t for t in tools if t.name == "google_disconnect").read_only, False)

    async def test_google_full_result_and_dynamic_credential_observer(self):
        import httpx
        from agent import google_oauth
        registered = []
        huge = "synthetic content" * 4000
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token.json"
            google_oauth._store({"access_token": "synthetic-access", "expires_in": 3600, "refresh_token": "fixture"}, token_path)
            tools = identity_google_tools(token_path, allow={"google_drive_read"}, credential_observer=registered.extend)
            def reply(method, url, **kwargs):
                if kwargs.get("params", {}).get("fields") == "name,mimeType":
                    return httpx.Response(200, json={"name": "fixture", "mimeType": "text/plain"})
                return httpx.Response(200, text=huge)
            with patch.object(httpx, "request", side_effect=reply):
                out = tools[0].handler({"file_id": "fixture"})
            self.assertTrue(out.endswith(huge))
            self.assertNotIn("[truncated]", out)
            self.assertEqual(registered, ["synthetic-access", "synthetic-access"])


class CLIBoundaries(unittest.TestCase):
    def test_importing_cli_and_shared_tools_never_imports_qwen(self):
        code = "import sys; import membraneclaw_runtime.cli; from membraneclaw_runtime.tool_adapters import builtin_tools; builtin_tools(); from membraneclaw_runtime.tool_adapters import identity_google_tools; from pathlib import Path; identity_google_tools(Path('/synthetic/user')); assert not any(n.startswith('qwen_agent') or n == 'agent.compat' for n in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_qwen_route_keeps_history_and_does_not_use_gemini_env_model(self):
        class Agent:
            def __init__(self):
                self.calls = []
            def run(self, messages):
                self.calls.append(list(messages))
                yield [{"role": "assistant", "content": "legacy answer"}]
        agent = Agent()
        with patch.dict(os.environ, {"AGENT_PROVIDER": "qwen", "GEMINI_MODEL_ID": "not-a-qwen-model"}), \
             patch("agent.factory.build_qwen_agent", return_value=agent) as factory, \
             patch("builtins.input", side_effect=["first", "second", "exit"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([]), 0)
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.calls[1][1], {"role": "assistant", "content": "legacy answer"})
        self.assertEqual(agent.calls[1][2], {"role": "user", "content": "second"})
        self.assertEqual(factory.call_args.kwargs["files"], [])

    def test_new_route_dispatch_and_configuration_flags(self):
        received = []
        async def fake(args, prompt):
            received.append((args.provider, args.model, args.trace_backend, args.trace_dir, prompt))
            return 0
        with patch.object(cli, "run_traced", side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
            result = cli.main(["--provider", "gemini", "--model", "gemini-2.5-flash", "--trace-dir", "/fixture", "-p", "task"])
        self.assertEqual(result, 0)
        self.assertEqual(received, [("gemini", "gemini-2.5-flash", "files", "/fixture", "task")])

    def test_new_provider_errors_hide_arbitrary_exception_details(self):
        async def fail(*args):
            raise ConfigurationError("private credential detail")
        out = io.StringIO()
        with patch.object(cli, "run_traced", side_effect=fail), contextlib.redirect_stderr(out):
            self.assertEqual(cli.main(["--provider", "gemini", "--model", "gemini-2.5-flash", "-p", "task"]), 1)
        self.assertNotIn("private credential detail", out.getvalue())

    def test_new_provider_rag_is_rejected_before_any_call(self):
        with patch.object(cli, "run_traced") as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["--provider", "gemini", "--model", "gemini-2.5-flash", "--file", "fixture", "-p", "task"])
            run.assert_not_called()

class MCPHTTPPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_client_does_not_follow_redirects(self):
        from membraneclaw_runtime.tool_adapters import _mcp_http_client
        with patch.dict(os.environ, {}, clear=True):
            async with _mcp_http_client() as client:
                self.assertFalse(client.follow_redirects)
