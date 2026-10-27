import copy
import http.client
import json
import unittest
from dataclasses import replace

from membraneclaw_runtime.gateway import ModelGateway, RetryPolicy
from membraneclaw_runtime.providers.openai import OpenAIAdapter
from membraneclaw_runtime.providers.openai_transport import OpenAITransport
from membraneclaw_runtime.providers.types import (ContentBlock, ConfigurationError, ModelRequest, NativeResponse,
    ProtocolError, ToolDefinition, ToolResult, TransportError)
from membraneclaw_runtime.tracing import MemoryTraceStore
from membraneclaw_runtime.tracing.schema import canonical_json
from test_transport import server

MODEL = "o4-mini-2025-04-16"
TOOL = ToolDefinition("calculate", "synthetic", {"type": "object", "properties": {"x": {"type": "number"}}})

def response(calls=False):
    items = [{"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "opaque-private", "unknown": 7}]
    if calls:
        items += [{"type": "function_call", "id": "fc" + str(i), "call_id": "c" + str(i),
                   "name": "calculate", "arguments": '{"x":%d}' % i} for i in range(2)]
    else:
        items += [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]
    return {"id": "resp1", "status": "completed", "output": items, "unknown": [1]}

class OpenAIContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.a = OpenAIAdapter()
        self.s = self.a.initial_state(MODEL)
        self.req = ModelRequest(self.s.route, (ContentBlock.text_block("synthetic"),), tools=(TOOL,))

    def norm(self, raw, req=None):
        return self.a.normalize(NativeResponse(200, {}, canonical_json(raw)), req or self.a.prepare(self.req, self.s),
                                response_ref="synthetic", model_call_id="model1")

    def test_two_tool_rounds_preserve_reasoning_and_pair_reversed_results(self):
        state = self.s
        for n in range(2):
            native = self.a.prepare(self.req if n == 0 else replace(self.req, content=()), state)
            raw = response(True)
            out = self.norm(raw, native)
            self.assertEqual(out.finish_status, "tool_calls")
            self.assertIsNone(out.usage.total_tokens)
            state = self.a.advance(state, out)
            with self.assertRaises(ConfigurationError):
                self.a.prepare(self.req, state)
            state = self.a.add_tool_results(state, tuple(ToolResult(c.id, {"x": c.arguments["x"]}) for c in reversed(out.tool_calls)))
            self.assertEqual(state.native_history[-2]["call_id"], "c0")
            self.assertEqual(state.native_history[-4]["call_id"], "c0")
            self.assertEqual(state.native_history[-5], raw["output"][0])
        self.assertEqual(len(state.native_history), 11)
        self.assertEqual(self.norm(response()).text, "done")

    def test_status_usage_refusal_and_invalid_arguments(self):
        raw = response(True)
        raw["output"][1]["arguments"] = '{broken'
        self.assertEqual(self.norm(raw).tool_calls[0].arguments_raw, '{broken')
        raw["status"] = "incomplete"
        self.assertEqual(self.norm(raw).finish_status, "incomplete")
        raw = response()
        raw["usage"] = {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                        "input_tokens_details": {"cached_tokens": 5}, "output_tokens_details": {"reasoning_tokens": 2}}
        self.assertEqual(self.norm(raw).usage.total_tokens, 13)
        raw["output"][-1]["content"] = [{"type": "refusal", "refusal": "no"}]
        self.assertEqual(self.norm(raw).finish_status, "blocked")
        raw["usage"]["total_tokens"] = -1
        with self.assertRaises(ProtocolError): self.norm(raw)

    def test_invalid_configuration_and_duplicates(self):
        for req in [replace(self.req, temperature=1), replace(self.req, provider_options={"openai": {"previous_response_id": "x"}}),
                    replace(self.req, tools=(TOOL, TOOL)), replace(self.req, provider_options={"anthropic": {}})]:
            with self.assertRaises(ConfigurationError): self.a.prepare(req, self.s)
        raw = response(True)
        raw["output"][-1]["call_id"] = "c0"
        with self.assertRaises(ProtocolError): self.norm(raw)

    async def test_real_transport_retry_raw_bytes_and_malformed_capture(self):
        for payload, expected in [(canonical_json(response()), None), (b'{malformed', ProtocolError)]:
            with server([(429, {"Retry-After": "0"}, b'busy'), (200, {}, payload)]) as httpd:
                transport = OpenAITransport("synthetic-key")
                transport._connection = lambda timeout: http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=timeout)
                store = MemoryTraceStore()
                g = await ModelGateway.start(store, self.a, transport, self.s.route)
                if expected:
                    with self.assertRaises(expected): await g.generate(self.req)
                else: self.assertEqual((await g.generate(self.req)).text, "done")
                ep = store.read_episode(g.recorder.episode)
                self.assertEqual(len(httpd.received), 2)
                self.assertEqual(httpd.received[0][2]["Authorization"], "Bearer synthetic-key")
                self.assertEqual([store.read_payload(e["payload_ref"]) for e in ep.events if e["type"] == "attempt.response"][-1], payload)
                self.assertNotIn(b"synthetic-key", canonical_json(ep.events))

    async def test_timeout_is_unknown_and_not_retried(self):
        from membraneclaw_runtime.gateway import Budget
        with server([(200, {"synthetic-delay": "0.08"}, canonical_json(response()))]) as httpd:
            t = OpenAITransport("synthetic-key")
            t._connection = lambda timeout: http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=timeout)
            g = await ModelGateway.start(MemoryTraceStore(), self.a, t, self.s.route,
                                         budget=Budget(attempt_timeout_seconds=.02))
            with self.assertRaises(TransportError): await g.generate(self.req)
            self.assertEqual(g.external_outcome, "unknown")
            self.assertEqual(len(httpd.received), 1)
