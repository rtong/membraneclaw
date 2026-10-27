import copy
import http.client
import unittest
from dataclasses import replace
from membraneclaw_runtime.providers.anthropic import AnthropicAdapter
from membraneclaw_runtime.providers.anthropic_transport import AnthropicTransport
from membraneclaw_runtime.providers.types import (ContentBlock, ConfigurationError, ModelRequest, NativeResponse,
    ProtocolError, ToolResult)
from membraneclaw_runtime.tracing import MemoryTraceStore
from membraneclaw_runtime.tracing.schema import canonical_json
from membraneclaw_runtime.gateway import ModelGateway
from test_openai import TOOL
from test_transport import server

MODEL = "claude-sonnet-4-20250514"
def response(calls=False):
    blocks = [{"type": "thinking", "thinking": "summary", "signature": "opaque-signature"},
              {"type": "redacted_thinking", "data": "opaque-redacted"}]
    blocks += ([{"type": "tool_use", "id": "c"+str(i), "name": "calculate", "input": {"x": i}} for i in range(2)]
               if calls else [{"type": "text", "text": "done"}])
    return {"type": "message", "role": "assistant", "id": "m1", "content": blocks,
            "stop_reason": "tool_use" if calls else "end_turn", "unknown": [7]}

class AnthropicContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.a = AnthropicAdapter(); self.s = self.a.initial_state(MODEL)
        self.req = ModelRequest(self.s.route, (ContentBlock.text_block("synthetic"),), tools=(TOOL,))
    def norm(self, raw, req=None):
        return self.a.normalize(NativeResponse(200, {}, canonical_json(raw)), req or self.a.prepare(self.req, self.s),
                                response_ref="synthetic", model_call_id="m1")
    def test_two_rounds_preserve_thinking_and_pair_errors(self):
        s = self.s
        for i in range(2):
            n = self.a.prepare(self.req if i == 0 else replace(self.req, content=()), s)
            out = self.norm(response(True), n)
            self.assertEqual(out.finish_status, "tool_calls")
            s = self.a.advance(s, out)
            self.assertEqual(s.native_history[-1]["content"][:2], response()["content"][:2])
            s = self.a.add_tool_results(s, tuple(ToolResult(c.id, {"ok": True}, True) for c in reversed(out.tool_calls)))
            self.assertEqual([b["tool_use_id"] for b in s.native_history[-1]["content"]], ["c0", "c1"])
            self.assertTrue(s.native_history[-1]["content"][0]["is_error"])
        self.assertEqual(len(s.native_history), 5)
    def test_usage_stops_and_unknown_server_tools(self):
        r=response();r["usage"]={"input_tokens":10,"output_tokens":3,"cache_read_input_tokens":5,"cache_creation_input_tokens":2}
        self.assertEqual(self.norm(r).usage.total_tokens,20)
        for reason,status in [("refusal","blocked"),("max_tokens","incomplete"),("pause_turn","incomplete"),("new","unknown")]:
            r["stop_reason"]=reason; self.assertEqual(self.norm(r).finish_status,status)
        r=response(True);r["content"].append({"type":"server_tool_use","id":"s","name":"web_search"})
        self.assertEqual(self.norm(r).finish_status,"unknown")
        self.assertIsNone(self.norm(response()).usage.total_tokens)
    def test_config_and_duplicate_validation(self):
        for options in [{"thinking":{"type":"enabled","budget_tokens":4096}}, {"tool_choice":{"type":"tool","name":"missing"}},
                        {"tools":[{"type":"web_search"}]}, {"thinking":{"type":"adaptive"}}]:
            with self.assertRaises(ConfigurationError):self.a.prepare(replace(self.req,provider_options={"anthropic":options}),self.s)
        r=response(True);r["content"][-1]["id"]="c0"
        with self.assertRaises(ProtocolError): self.norm(r)
    async def test_transport_429_and_raw_malformed(self):
        for raw,error in [(canonical_json(response()),False),(b'{bad',True)]:
            with server([(429,{"Retry-After":"0"},b'busy'),(200,{},raw)]) as h:
                t=AnthropicTransport("synthetic-key");t._connection=lambda timeout:http.client.HTTPConnection("127.0.0.1",h.server_port,timeout=timeout)
                store=MemoryTraceStore();g=await ModelGateway.start(store,self.a,t,self.s.route)
                if error:
                    with self.assertRaises(ProtocolError):await g.generate(self.req)
                else:self.assertEqual((await g.generate(self.req)).text,"done")
                self.assertEqual(len(h.received),2)
                self.assertEqual(h.received[0][2]["anthropic-version"],"2023-06-01")
                self.assertEqual([store.read_payload(e["payload_ref"]) for e in store.read_episode(g.recorder.episode).events if e["type"]=="attempt.response"][-1],raw)
