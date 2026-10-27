"""Gateway tests share behavioral assertions across durable and memory stores."""
import asyncio
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from membraneclaw_runtime.gateway import Budget, ModelGateway, RetryPolicy, TraceStorageError
from membraneclaw_runtime.providers.gemini import GeminiAdapter
from membraneclaw_runtime.providers.registry import ProviderRegistry
from membraneclaw_runtime.providers.types import (
    BudgetExceeded, Capabilities, ConfigurationError, ContentBlock, ModelOutput, ModelRequest,
    ModelRoute, NativeRequest, NativeResponse, ProtocolError, ProviderHTTPError,
    ProviderState, ToolResult, TransportError, Usage)
from membraneclaw_runtime.tracing import FileTraceStore, MemoryTraceStore
from membraneclaw_runtime.tracing.policy import CredentialRejected
from membraneclaw_runtime.tracing.schema import Busy, canonical_json, digest
from membraneclaw_runtime.tracing.validation import validate

from test_gemini import FIXTURE, ROUTE, request


def final(text="synthetic answer", usage=True):
    data = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]}
    if usage:
        data["usageMetadata"] = {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
    return NativeResponse(200, {}, canonical_json(data))


class ScriptedTransport:
    authentication_values = (b"synthetic-auth-value",)

    def __init__(self, script, store=None):
        self.script, self.calls, self.store = list(script), [], store
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.slow = False

    async def exchange(self, native, *, before_send, after_receive, timeout):
        await before_send(native)
        if self.store:
            # Before even the fake I/O, the exact final body must be durable.
            events = self.store.read_episode(self.episode).events
            self.asserted_start = events[-1]["type"] == "attempt.started"
            assert self.store.read_payload(events[-1]["payload_ref"]) == native.body
        self.calls.append(native)
        self.started.set()
        if self.slow:
            await self.release.wait()
        response = self.script.pop(0)
        if isinstance(response, Exception):
            raise response
        await after_receive(response)
        return response


async def no_wait(_):
    pass


class GatewayContracts:
    def make_store(self, root):
        raise NotImplementedError

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = self.make_store(self.tmp.name)

    async def gateway(self, script=None, **kwargs):
        self.transport = ScriptedTransport(script if script is not None else [final()], self.store)
        gateway = await ModelGateway.start(self.store, GeminiAdapter(), self.transport, ROUTE,
                                           sleep=no_wait, jitter=lambda: 0, **kwargs)
        self.transport.episode = gateway.recorder.episode
        return gateway

    def events(self, gw, kind=None):
        events = self.store.read_episode(gw.recorder.episode).events
        return [e for e in events if kind is None or e["type"] == kind]

    async def test_commit_before_send_response_before_parse(self):
        gw = await self.gateway()
        original = gw.adapter.normalize
        def normalize(response, *args, **kwargs):
            events = self.events(gw, "attempt.response")
            assert len(events) == 1
            assert self.store.read_payload(events[0]["payload_ref"]) == response.body
            return original(response, *args, **kwargs)
        with patch.object(gw.adapter, "normalize", side_effect=normalize):
            out = await gw.generate(request())
        self.assertEqual(out.text, "synthetic answer")
        self.assertTrue(self.transport.asserted_start)
        await gw.finish()
        ep = self.store.read_episode(gw.recorder.episode)
        self.assertEqual(ep.status, "completed")
        self.assertEqual(ep.completeness["capture"], "complete")
        self.assertEqual(validate(self.store, gw.recorder.episode).status, "completed")

    async def test_retry_attempts_and_request_count_are_identical(self):
        gw = await self.gateway([NativeResponse(429, {"retry-after": "0"}, b'{"error":"busy"}'), final()])
        await gw.generate(request())
        starts = self.events(gw, "attempt.started")
        self.assertEqual(len(starts), len(self.transport.calls))
        self.assertEqual(len({e["attempt_id"] for e in starts}), 2)
        self.assertEqual(len({e["model_call_id"] for e in starts}), 1)
        self.assertTrue(gw.usage_unknown)
        self.assertEqual(gw.tokens_observed, 5)
        self.assertEqual(len(self.events(gw, "attempt.retry_scheduled")), 1)

    async def test_400_does_not_retry(self):
        gw = await self.gateway([NativeResponse(400, {}, b'{"error":"bad request"}')])
        with self.assertRaises(ProviderHTTPError) as error:
            await gw.generate(request())
        self.assertEqual(error.exception.status, 400)
        self.assertEqual(len(self.transport.calls), 1)
        await gw.finish("failed")
        self.assertEqual(self.store.read_episode(gw.recorder.episode).completeness["capture"], "complete")

    async def test_malformed_body_is_saved_without_retry(self):
        raw = b'{"malformed":\xff'
        gw = await self.gateway([NativeResponse(200, {}, raw)])
        with self.assertRaises(ProtocolError):
            await gw.generate(request())
        event = self.events(gw, "attempt.response")[0]
        self.assertEqual(self.store.read_payload(event["payload_ref"]), raw)
        self.assertEqual(len(self.events(gw, "attempt.parse_failed")), 1)
        self.assertEqual(len(self.transport.calls), 1)
        with self.assertRaises(ConfigurationError):
            await gw.generate(request())

    async def test_timeout_default_does_not_retry_unknown_outcome(self):
        gw = await self.gateway([TransportError("request_timeout")])
        with self.assertRaises(TransportError):
            await gw.generate(request())
        self.assertEqual(len(self.transport.calls), 1)
        await gw.finish("failed")
        comp = self.store.read_episode(gw.recorder.episode).completeness
        self.assertEqual(comp["external_outcome"], "unknown")
        self.assertEqual(comp["capture"], "partial")

    async def test_timeout_opt_in_is_a_separate_attempt(self):
        gw = await self.gateway([TransportError("request_timeout"), final()],
                                retry=RetryPolicy(retry_timeouts=True))
        await gw.generate(request())
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(gw.external_outcome, "unknown")
        self.assertEqual(gw.capture, "partial")

    async def test_request_and_token_budget_stop_before_extra_send(self):
        gw = await self.gateway([NativeResponse(503, {}, b"busy")], budget=Budget(max_requests=1))
        with self.assertRaises(BudgetExceeded):
            await gw.generate(request())
        self.assertEqual(len(self.transport.calls), 1)
        gw = await self.gateway(budget=Budget(max_total_tokens=10))
        with self.assertRaises(BudgetExceeded):
            await gw.generate(request(max_output_tokens=8, input_token_upper_bound=4))
        self.assertEqual(len(self.transport.calls), 0)

    async def test_token_budget_requires_an_input_bound(self):
        gw = await self.gateway(budget=Budget(max_total_tokens=100))
        with self.assertRaises(ConfigurationError):
            await gw.generate(request())
        self.assertEqual(len(self.transport.calls), 0)

    async def test_blocked_missing_candidates_and_unknown_usage(self):
        for body, status in [({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked"), ({}, "unknown")]:
            gw = await self.gateway([NativeResponse(200, {}, canonical_json(body))])
            out = await gw.generate(request())
            self.assertEqual(out.finish_status, status)
            self.assertIsNone(out.usage.total_tokens)
            with self.assertRaises(ConfigurationError):
                await gw.finish()

    async def test_storage_failure_before_send_is_fail_closed(self):
        gw = await self.gateway()
        with patch.object(self.store, "_write_commit", side_effect=OSError("disk full")):
            with self.assertRaises(TraceStorageError):
                await gw.generate(request())
        self.assertEqual(self.transport.calls, [])

    async def test_response_storage_failure_prevents_normalization(self):
        gw = await self.gateway()
        write = self.store._write_commit
        def fail_response(ep, commit):
            if any(e["type"] == "attempt.response" for e in commit["events"]):
                raise OSError("disk full")
            return write(ep, commit)
        with patch.object(self.store, "_write_commit", side_effect=fail_response), patch.object(gw.adapter, "normalize") as parser:
            with self.assertRaises(TraceStorageError):
                await gw.generate(request())
            parser.assert_not_called()
        self.assertEqual(len(self.transport.calls), 1)

    async def test_no_credential_values_in_request_or_output_trace(self):
        gw = await self.gateway([final("synthetic-auth-value")])
        with self.assertRaises(CredentialRejected):
            await gw.generate(replace(request(), content=(ContentBlock.text_block("synthetic-auth-value"),)))
        self.assertEqual(self.transport.calls, [])
        with self.assertRaises(TraceStorageError):
            await gw.generate(request())
        for event in self.events(gw):
            self.assertNotIn("synthetic-auth-value", json.dumps(event))
            if event["payload_ref"]:
                self.assertNotIn(b"synthetic-auth-value", self.store.read_payload(event["payload_ref"]))

    async def test_execution_lock_covers_slow_network_and_cancellation(self):
        gw = await self.gateway()
        self.transport.slow = True
        task = asyncio.create_task(gw.generate(request()))
        await asyncio.wait_for(self.transport.started.wait(), 3)
        with self.assertRaises(Busy):
            self.store.recover(gw.recorder.episode)
        with self.assertRaises(ConfigurationError):
            await gw.generate(request())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.events(gw, "attempt.cancelled")), 1)
        await gw.finish("cancelled")

    async def test_tool_round_trip_saved_state_not_display_text(self):
        gw = await self.gateway([NativeResponse(200, {}, FIXTURE.read_bytes()), final()])
        async with gw.execution():
            out = await gw.generate(request())
            await gw.add_tool_results(tuple(ToolResult(c.id, {"value": n}) for n, c in enumerate(reversed(out.tool_calls))))
            out2 = await gw.generate(replace(request(), content=()))
            self.assertEqual(out2.finish_status, "completed")
        checkpoint = self.store.read_episode(gw.recorder.episode).checkpoint_ref
        state = json.loads(self.store.read_payload(checkpoint))["provider_state"]
        self.assertEqual(state["native_history"][1]["parts"][0]["thoughtSignature"], "AQID+/==")
        self.assertEqual(len(state["native_history"]), 4)
        self.assertEqual(len(self.events(gw, "tool.results_attached")), 1)

    async def test_route_change_is_rejected(self):
        gw = await self.gateway()
        with self.assertRaises(ConfigurationError):
            await gw.generate(replace(request(), route=replace(ROUTE, provider="openai")))
        self.assertEqual(self.transport.calls, [])

    async def test_retry_after_and_time_budget(self):
        gw = await self.gateway([NativeResponse(429, {"retry-after": "500"}, b"busy")],
                                budget=Budget(max_elapsed_seconds=1))
        with self.assertRaises(BudgetExceeded):
            await gw.generate(request())
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(gw._delay(1, "0"), 0)
        self.assertEqual(gw._delay(1, "invalid"), 0.5)

    async def test_partial_response_is_preserved_and_not_parsed(self):
        gw = await self.gateway([NativeResponse(200, {}, b'{"partial":', False, "truncated_body")])
        with self.assertRaises(TransportError):
            await gw.generate(request())
        self.assertEqual(len(self.events(gw, "attempt.partial")), 1)
        self.assertEqual(gw.capture, "partial")
        self.assertEqual(len(self.transport.calls), 1)


class MemoryGateway(GatewayContracts, unittest.IsolatedAsyncioTestCase):
    def make_store(self, root):
        return MemoryTraceStore()


class FileGateway(GatewayContracts, unittest.IsolatedAsyncioTestCase):
    def make_store(self, root):
        return FileTraceStore(root)


class DifferentAdapter:
    provider, api_family, adapter_version = "synthetic", "test_api", "test_v1"

    def initial_state(self, model):
        return ProviderState(ModelRoute(self.provider, model, self.api_family), self.adapter_version)

    def capabilities(self, model):
        return Capabilities(tool_calling=False, parallel_calls=False)

    def prepare(self, request, state):
        return NativeRequest(state.route, "POST", "/synthetic", b'{"input":"fixture"}', "synthetic")

    def normalize(self, response, native, *, response_ref, model_call_id):
        return ModelOutput((ContentBlock.text_block("other adapter"),), (), (), "completed", "test_stop",
                           Usage(), {"response": "fixture"}, native, response_ref, None, model_call_id)

    def advance(self, state, output):
        return state


class ProviderNeutralGateway(unittest.IsolatedAsyncioTestCase):
    async def test_same_gateway_for_an_independent_adapter_and_judge_role(self):
        registry = ProviderRegistry()
        registry.register("synthetic", DifferentAdapter)
        route = ModelRoute("synthetic", "synthetic-model", "test_api")
        transport = ScriptedTransport([NativeResponse(200, {}, b"synthetic raw")])
        gw = await ModelGateway.start(MemoryTraceStore(), registry.create(route), transport, route, role="judge")
        out = await gw.generate(ModelRequest(route))
        self.assertEqual(out.text, "other adapter")
        self.assertEqual(gw.role, "judge")
        self.assertEqual(len(transport.calls), 1)


class TransportContractEnforcement(unittest.IsolatedAsyncioTestCase):
    async def test_changed_response_cannot_bypass_capture(self):
        class WrongResponse(ScriptedTransport):
            async def exchange(self, native, *, before_send, after_receive, timeout):
                await before_send(native)
                await after_receive(final("recorded"))
                return final("different")
        gw = await ModelGateway.start(MemoryTraceStore(), GeminiAdapter(), WrongResponse([]), ROUTE)
        with self.assertRaises(ProtocolError):
            await gw.generate(request())
        self.assertIsNone(gw._last_output)

    async def test_no_capture_cannot_produce_success(self):
        class NoCapture(ScriptedTransport):
            async def exchange(self, native, **kwargs):
                return final()
        gw = await ModelGateway.start(MemoryTraceStore(), GeminiAdapter(), NoCapture([]), ROUTE)
        with self.assertRaises(ProtocolError):
            await gw.generate(request())
        self.assertIsNone(gw._last_output)


class UncertainServiceFailure(unittest.IsolatedAsyncioTestCase):
    async def test_later_success_does_not_clear_a_5xx_unknown_outcome(self):
        store = MemoryTraceStore()
        transport = ScriptedTransport([NativeResponse(503, {}, b"synthetic proxy failure"), final()])
        gw = await ModelGateway.start(store, GeminiAdapter(), transport, ROUTE, sleep=no_wait)
        await gw.generate(request())
        await gw.finish()
        self.assertEqual(store.read_episode(gw.recorder.episode).completeness["external_outcome"], "unknown")
        self.assertTrue(gw.usage_unknown)


class ConservativeTokenAccounting(unittest.IsolatedAsyncioTestCase):
    async def test_known_overrun_and_unknown_usage_are_both_retained(self):
        high = json.loads(final().body)
        high["usageMetadata"]["totalTokenCount"] = 12
        transport = ScriptedTransport([NativeResponse(200, {}, canonical_json(high)), final(usage=False), final()])
        gw = await ModelGateway.start(MemoryTraceStore(), GeminiAdapter(), transport, ROUTE,
                                      budget=Budget(max_total_tokens=35), sleep=no_wait)
        await gw.generate(request(max_output_tokens=8, input_token_upper_bound=2))
        await gw.generate(request(max_output_tokens=8, input_token_upper_bound=2))
        self.assertEqual(gw.tokens_reserved, 20)
        self.assertEqual(gw.tokens_observed, 12)
        self.assertEqual(gw.tokens_accounted, 22)
        with self.assertRaises(BudgetExceeded):
            await gw.generate(request(max_output_tokens=8, input_token_upper_bound=7))
        self.assertEqual(len(transport.calls), 2)


class CancellationAtCommit(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_during_final_commit_cannot_start_another_request(self):
        import threading
        committed, release = threading.Event(), threading.Event()
        class PausedStore(MemoryTraceStore):
            def _update_manifest(self, ep):
                if ep.events and ep.events[-1]["type"] == "model.completed":
                    committed.set()
                    if not release.wait(3):
                        raise OSError("synthetic test timeout")
        store = PausedStore()
        transport = ScriptedTransport([final(), final()])
        gw = await ModelGateway.start(store, GeminiAdapter(), transport, ROUTE)
        task = asyncio.create_task(gw.generate(request()))
        try:
            self.assertTrue(await asyncio.to_thread(committed.wait, 3))
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(ConfigurationError):
                await gw.generate(request())
            self.assertEqual(len(transport.calls), 1)
            await gw.finish("cancelled")
            self.assertEqual(store.read_episode(gw.recorder.episode).completeness["capture"], "partial")
        finally:
            release.set()

    async def test_cancellation_during_backoff_is_not_a_silent_new_attempt(self):
        waiting = asyncio.Event()
        async def wait(_):
            waiting.set()
            await asyncio.Event().wait()
        store = MemoryTraceStore()
        transport = ScriptedTransport([NativeResponse(429, {}, b"busy"), final()])
        gw = await ModelGateway.start(store, GeminiAdapter(), transport, ROUTE, sleep=wait)
        task = asyncio.create_task(gw.generate(request()))
        await asyncio.wait_for(waiting.wait(), 3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(ConfigurationError):
            await gw.generate(request())
        self.assertEqual(len(transport.calls), 1)
