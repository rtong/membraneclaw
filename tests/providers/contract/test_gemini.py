"""Pure offline adapter contract tests; every fixture is synthetic."""
import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from membraneclaw_runtime.providers.gemini import GeminiAdapter, strict_json
from membraneclaw_runtime.providers.registry import default_registry
from membraneclaw_runtime.providers.types import (
    ConfigurationError, ContentBlock, ModelRequest, ModelRoute, NativeResponse,
    ProtocolError, ToolDefinition, ToolResult)
from membraneclaw_runtime.tracing.schema import canonical_json

ROUTE = ModelRoute("gemini", "gemini-2.5-flash", "generate_content_v1beta")
TOOL = ToolDefinition("read_value", "Synthetic read only", {
    "type": "object", "properties": {"value": {"type": "integer", "minimum": 0}},
    "required": ["value"], "additionalProperties": False})
FIXTURE = Path(__file__).parents[1] / "fixtures/gemini_tool_response.json"


def request(**kwargs):
    return ModelRequest(ROUTE, (ContentBlock.text_block("Synthetic user input"),), tools=(TOOL,), **kwargs)


def response(raw=None):
    return NativeResponse(200, {}, raw if raw is not None else FIXTURE.read_bytes())


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.adapter = GeminiAdapter()
        self.state = self.adapter.initial_state(ROUTE.model)
        self.req = request()
        self.native = self.adapter.prepare(self.req, self.state)

    def normalize(self, raw=None):
        return self.adapter.normalize(response(raw), self.native,
                                      response_ref="synthetic-ref", model_call_id="logical")

    def test_candidates_signatures_and_usage_not_double_counted(self):
        out = self.normalize()
        self.assertEqual(len(out.native_response["candidates"]), 2)
        self.assertEqual(out.selected_candidate, 0)
        self.assertEqual(out.finish_status, "tool_calls")
        self.assertEqual(out.text, "")
        self.assertEqual(out.thought_summaries, ("Synthetic thought summary",))
        self.assertEqual([c.id for c in out.tool_calls], ["logical_part_1", "logical_part_2"])
        self.assertEqual(out.usage.total_tokens, 33)
        self.assertEqual(out.usage.input_tokens, 20)
        self.assertEqual(out.usage.cached_input_tokens, 5)
        self.assertEqual(out.usage.reasoning_tokens, 4)
        self.assertEqual(out.usage.raw["futureUsage"], 99)
        self.assertEqual(out.native_response["futureResponse"]["nested"], [1, 2, 3])

    def test_native_continuation_and_same_name_result_pairing(self):
        out = self.normalize()
        state = self.adapter.advance(self.state, out)
        with self.assertRaises(ConfigurationError):
            self.adapter.prepare(self.req, state)
        results = (ToolResult(out.tool_calls[1].id, {"value": 14}),
                   ToolResult(out.tool_calls[0].id, {"message": "synthetic failure"}, True))
        state = self.adapter.add_tool_results(state, results)
        next_native = self.adapter.prepare(replace(self.req, content=()), state)
        body = strict_json(next_native.body)
        self.assertEqual(len(body["contents"]), 3)
        self.assertEqual(body["contents"][0]["parts"], [{"text": "Synthetic user input"}])
        self.assertEqual(body["contents"][1], out.native_response["candidates"][0]["content"])
        parts = body["contents"][2]["parts"]
        self.assertEqual(parts[0]["functionResponse"]["id"], "provider-one")
        self.assertIn("error", parts[0]["functionResponse"]["response"])
        self.assertEqual(parts[1]["functionResponse"]["response"]["result"]["value"], 14)

    def test_prepare_and_advance_do_not_mutate_inputs(self):
        before = copy.deepcopy(self.state)
        out = self.normalize()
        state = self.adapter.advance(self.state, out)
        out.native_response["candidates"][0]["content"]["parts"][0]["thoughtSignature"] = "mutated"
        self.assertEqual(self.state, before)
        self.assertEqual(state.native_history[1]["parts"][0]["thoughtSignature"], "AQID+/==")

    def test_no_provider_call_id_uses_internal_id_only(self):
        raw = json.loads(FIXTURE.read_bytes())
        for part in raw["candidates"][0]["content"]["parts"]:
            part.get("functionCall", {}).pop("id", None)
        out = self.normalize(canonical_json(raw))
        state = self.adapter.advance(self.state, out, tuple(ToolResult(c.id, {}) for c in out.tool_calls))
        self.assertNotIn("id", state.native_history[-1]["parts"][0]["functionResponse"])

    def test_result_set_must_be_exact(self):
        out = self.normalize()
        state = self.adapter.advance(self.state, out)
        for results in [(), (ToolResult("unknown", {}),),
                        (ToolResult(out.tool_calls[0].id, {}),) * 2]:
            with self.subTest(results=results), self.assertRaises(ConfigurationError):
                self.adapter.add_tool_results(state, results)

    def test_parameters_and_reasoning_are_explicit(self):
        req = replace(self.req, temperature=0, max_output_tokens=256,
                      provider_options={"gemini": {"thinkingConfig": {"thinkingBudget": 0, "includeThoughts": True}}})
        body = strict_json(self.adapter.prepare(req, self.state).body)
        self.assertEqual(body["generationConfig"]["temperature"], 0)
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingBudget"], 0)
        self.assertEqual(body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"], TOOL.parameters)

    def test_invalid_capabilities_fail_before_transport(self):
        bad = [
            replace(self.req, route=replace(ROUTE, model="unreviewed")),
            replace(self.req, temperature=float("nan")), replace(self.req, temperature=True),
            replace(self.req, max_output_tokens=0), replace(self.req, max_output_tokens=65537),
            replace(self.req, content=(ContentBlock("image", "uri"),)),
            replace(self.req, provider_options={"openai": {"effort": "high"}}),
            replace(self.req, provider_options={"gemini": {"unknown": True}}),
            replace(self.req, provider_options={"gemini": {"thinkingConfig": {"thinkingBudget": 30000}}}),
            replace(self.req, tools=(replace(TOOL, parameters={"type": "object", "$ref": "#/x"}),)),
            replace(self.req, tools=(TOOL, TOOL)),
        ]
        for req in bad:
            with self.subTest(req=req), self.assertRaises(ConfigurationError):
                self.adapter.prepare(req, self.state)

    def test_config_and_state_versions_cannot_silently_change(self):
        out = self.normalize()
        state = self.adapter.advance(self.state, out, tuple(ToolResult(c.id, {}) for c in out.tool_calls))
        with self.assertRaises(ConfigurationError):
            self.adapter.prepare(replace(self.req, temperature=1), state)
        with self.assertRaises(ConfigurationError):
            self.adapter.prepare(self.req, replace(self.state, adapter_version="future"))
        with self.assertRaises(ConfigurationError):
            self.adapter.prepare(self.req, replace(self.state, schema_version=999))

    def test_finish_and_missing_usage_branches(self):
        cases = [
            ({}, "unknown"),
            ({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked"),
            ({"candidates": [{"content": {"parts": [{"text": "partial"}]}, "finishReason": "MAX_TOKENS"}]}, "incomplete"),
            ({"candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}]}, "completed"),
            ({"candidates": [{"content": {"parts": [{"text": "text alone"}]}}]}, "unknown"),
            ({"candidates": [{"content": {"parts": [{"inlineData": {"data": "AQ=="}}]}, "finishReason": "STOP"}]}, "incomplete"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                out = self.normalize(canonical_json(raw))
                self.assertEqual(out.finish_status, expected)
                self.assertIsNone(out.usage.total_tokens)
                self.assertEqual(out.usage.unavailable_reason, "not_provided")

    def test_malformed_json_and_tool_arguments_stay_errors(self):
        for raw in [b'{"broken":', b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"]:
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                self.normalize(raw)
        raw = json.loads(FIXTURE.read_bytes())
        raw["candidates"][0]["content"]["parts"][1]["functionCall"]["args"] = "not an object"
        with self.assertRaises(ProtocolError):
            self.normalize(canonical_json(raw))

    def test_registry_has_no_silent_fallback(self):
        registry = default_registry()
        self.assertIsInstance(registry.create(ROUTE), GeminiAdapter)
        with self.assertRaises(ConfigurationError):
            registry.create(replace(ROUTE, provider="openai"))
        with self.assertRaises(ConfigurationError):
            registry.create(replace(ROUTE, api_family="vertex"))
        with self.assertRaises(ConfigurationError):
            registry.register("gemini", GeminiAdapter)
        self.assertFalse(self.adapter.capabilities(ROUTE.model).streaming)


class StateSerialization(unittest.TestCase):
    def test_native_state_has_an_explicit_json_document_boundary(self):
        adapter = GeminiAdapter()
        state = adapter.initial_state(ROUTE.model)
        canonical_json(state.to_document())
        native = adapter.prepare(request(), state)
        out = adapter.normalize(response(), native, response_ref="synthetic", model_call_id="logical")
        advanced = adapter.advance(state, out)
        document = strict_json(canonical_json(advanced.to_document()))
        self.assertIsInstance(document["native_history"], list)
        self.assertIsInstance(document["pending_calls"], list)
        self.assertEqual(document["native_history"][1]["parts"][0]["thoughtSignature"], "AQID+/==")


class ContinuationIsolation(unittest.TestCase):
    def test_mutating_output_arguments_cannot_change_authoritative_state(self):
        adapter = GeminiAdapter()
        state = adapter.initial_state(ROUTE.model)
        native = adapter.prepare(request(), state)
        out = adapter.normalize(response(), native, response_ref="synthetic", model_call_id="logical")
        new = adapter.advance(state, out)
        out.tool_calls[0].arguments["value"] = 999
        self.assertEqual(new.pending_calls[0].arguments["value"], 2)

    def test_null_total_usage_is_explicitly_unavailable(self):
        adapter = GeminiAdapter()
        native = adapter.prepare(request(), adapter.initial_state(ROUTE.model))
        raw = canonical_json({"usageMetadata": {"promptTokenCount": 5, "totalTokenCount": None}})
        out = adapter.normalize(response(raw), native, response_ref="synthetic", model_call_id="logical")
        self.assertIsNone(out.usage.total_tokens)
        self.assertEqual(out.usage.unavailable_reason, "total_not_provided")
