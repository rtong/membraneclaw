import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from membraneclaw_runtime.gateway import Budget, ModelGateway, TraceStorageError
from membraneclaw_runtime.providers.types import ContentBlock, ModelRequest
from membraneclaw_runtime.runtime import AgentRuntime
from membraneclaw_runtime.state import RuntimeLimits
from membraneclaw_runtime.tools import ToolRegistry, ToolSpec
from membraneclaw_runtime.tracing import FileTraceStore, MemoryTraceStore
from membraneclaw_runtime.tracing.schema import Busy
from membraneclaw_runtime.tracing.validation import validate

from helpers import FakeAdapter, FakeTransport


def tool(name, handler, **kwargs):
    return ToolSpec(name, "synthetic tool", {"type": "object", "properties": {"n": {"type": "integer"}},
                    "required": ["n"], "additionalProperties": False}, handler, "fixture_v1", **kwargs)


class RuntimeContracts(unittest.IsolatedAsyncioTestCase):
    async def make(self, script, tools=(), limits=None, store=None, provider="synthetic"):
        adapter = FakeAdapter(provider)
        self.transport = FakeTransport(script)
        self.store = store or MemoryTraceStore()
        gw = await ModelGateway.start(self.store, adapter, self.transport, adapter.initial_state("fixture").route,
                                      budget=Budget(max_requests=20))
        registry = ToolRegistry(tools)
        request = ModelRequest(gw.route, (ContentBlock.text_block("synthetic task"),), tools=registry.definitions())
        self.runtime = AgentRuntime(gw, registry, request, limits=limits)
        return self.runtime

    def events(self, kind=None):
        events = self.store.read_episode(self.runtime.gateway.recorder.episode).events
        return [e for e in events if kind is None or e["type"] == kind]

    async def test_same_loop_with_three_independent_fake_provider_routes(self):
        for provider in ["fake_gemini", "fake_gpt", "fake_claude"]:
            calls = []
            runtime = await self.make([
                {"calls": [{"name": "simulate", "args": {"n": 1}}, {"name": "simulate", "args": {"n": 2}}]},
                {"calls": [{"name": "check", "args": {"n": 3}}]}, {"text": "fixture success"}],
                [tool("simulate", lambda a: calls.append(a["n"]) or {"value": a["n"]}),
                 tool("check", lambda a: {"pass": a["n"] == 3})], provider=provider)
            result = await runtime.run()
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tool_calls, 3)
            self.assertEqual(calls, [1, 2])
            self.assertEqual(len(self.transport.requests), 3)
            self.assertEqual(len(self.events("tool.started")), 3)
            self.assertEqual(validate(self.store, runtime.gateway.recorder.episode).status, "completed")

    async def test_bad_arguments_can_be_corrected_without_invoking_the_bad_call(self):
        calls = []
        runtime = await self.make([{"calls": [{"name": "read", "args": {"n": "wrong"}}]},
                                   {"calls": [{"name": "read", "args": {"n": 2}}]}, {"text": "done"}],
                                  [tool("read", lambda a: calls.append(a["n"]) or a)])
        result = await runtime.run()
        self.assertEqual(result.status, "completed")
        self.assertEqual(calls, [2])
        self.assertEqual(len(self.events("tool.validation_failed")), 1)
        self.assertEqual(len(self.events("tool.started")), 1)

    async def test_full_results_survive_file_backend_and_ui_size(self):
        huge = "synthetic" * 100000
        with tempfile.TemporaryDirectory() as directory:
            runtime = await self.make([{"calls": [{"name": "read", "args": {"n": 1}}]}, {"text": "done"}],
                                      [tool("read", lambda a: huge)], store=FileTraceStore(directory))
            await runtime.run()
            event = self.events("tool.completed")[0]
            self.assertEqual(self.store.read_payload(event["payload_ref"]), huge.encode())
            checkpoint = self.store.read_episode(runtime.gateway.recorder.episode).checkpoint_ref
            snapshot = json.loads(self.store.read_payload(checkpoint))
            self.assertEqual(snapshot["application_state"]["tools"]["tools"][0]["version"], "fixture_v1")

    async def test_parallel_read_only_completes_out_of_order_but_pairs_correctly(self):
        done = []
        async def read(a):
            await asyncio.sleep(0.02 if a["n"] == 1 else 0)
            done.append(a["n"])
            return a
        runtime = await self.make([{"calls": [{"name": "read", "args": {"n": 1}},
                                              {"name": "read", "args": {"n": 2}}]}, {"text": "done"}],
            [tool("read", read, read_only=True, parallel_safe=True)], RuntimeLimits(parallel_read_only=True))
        await runtime.run()
        self.assertEqual(done, [2, 1])
        results = json.loads(self.transport.requests[1].body)["history"][-1]["results"]
        self.assertEqual([r["result"]["n"] for r in results], [1, 2])
        completed = [json.loads(self.store.read_payload(e["payload_ref"]))["n"] for e in self.events("tool.completed")]
        self.assertEqual(completed, [2, 1])

    async def test_write_tools_stay_serial_even_with_parallel_flag(self):
        order = []
        async def write(a):
            order.append((a["n"], "start"))
            await asyncio.sleep(0)
            order.append((a["n"], "end"))
            return a
        runtime = await self.make([{"calls": [{"name": "write", "args": {"n": 1}},
                                              {"name": "write", "args": {"n": 2}}]}, {"text": "done"}],
            [tool("write", write)], RuntimeLimits(parallel_read_only=True))
        await runtime.run()
        self.assertEqual(order, [(1, "start"), (1, "end"), (2, "start"), (2, "end")])

    async def test_unknown_outcome_stops_loop_and_remaining_side_effects(self):
        calls = []
        async def slow(a):
            calls.append(a["n"])
            await asyncio.sleep(1)
        runtime = await self.make([{"calls": [{"name": "write", "args": {"n": 1}},
                                              {"name": "write", "args": {"n": 2}}]}],
                                 [tool("write", slow)], RuntimeLimits(tool_timeout_seconds=0.01))
        result = await runtime.run()
        self.assertEqual(result.reason, "tool_outcome_unknown")
        self.assertEqual(calls, [1])
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(self.store.read_episode(runtime.gateway.recorder.episode).completeness["external_outcome"], "unknown")

    async def test_tool_and_step_budgets_prevent_extra_external_work(self):
        calls = []
        runtime = await self.make([{"calls": [{"name": "read", "args": {"n": 1}},
                                              {"name": "read", "args": {"n": 2}}]}],
                                  [tool("read", lambda a: calls.append(a))], RuntimeLimits(max_tool_calls=1))
        self.assertEqual((await runtime.run()).reason, "tool_call_budget_exhausted")
        self.assertEqual(calls, [])
        runtime = await self.make([{"calls": [{"name": "read", "args": {"n": 1}}]}],
                                  [tool("read", lambda a: a)], RuntimeLimits(max_steps=1))
        self.assertEqual((await runtime.run()).reason, "step_budget_exhausted")
        self.assertEqual(len(self.transport.requests), 1)

    async def test_context_cap_fails_without_trimming_or_sending(self):
        runtime = await self.make([], limits=RuntimeLimits(max_context_bytes=1))
        result = await runtime.run()
        self.assertEqual(result.reason, "context_byte_budget_exhausted")
        self.assertEqual(self.transport.requests, [])

    async def test_blocked_incomplete_and_unknown_are_not_success(self):
        for status in ["blocked", "incomplete", "unknown"]:
            runtime = await self.make([{"text": "partial", "status": status}])
            result = await runtime.run()
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.reason, "model_" + status)

    async def test_tool_start_storage_failure_prevents_the_tool(self):
        called = []
        runtime = await self.make([{"calls": [{"name": "write", "args": {"n": 1}}]}],
                                  [tool("write", lambda a: called.append(a))])
        original = self.store._write_commit
        def fail(ep, commit):
            if any(e["type"] == "tool.started" for e in commit["events"]):
                raise OSError("synthetic disk full")
            return original(ep, commit)
        with patch.object(self.store, "_write_commit", side_effect=fail), self.assertRaises(TraceStorageError):
            await runtime.run()
        self.assertEqual(called, [])
        self.assertEqual(len(self.transport.requests), 1)

    async def test_execution_guard_is_held_during_tools_and_cancellation(self):
        started = asyncio.Event()
        async def slow(a):
            started.set()
            await asyncio.sleep(10)
        runtime = await self.make([{"calls": [{"name": "read", "args": {"n": 1}}]}], [tool("read", slow)])
        task = asyncio.create_task(runtime.run())
        await asyncio.wait_for(started.wait(), 3)
        with self.assertRaises(Busy):
            self.store.recover(runtime.gateway.recorder.episode)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.store.read_episode(runtime.gateway.recorder.episode).status, "cancelled")

    async def test_dynamic_tool_credential_is_rejected_before_result_persistence(self):
        async def credential_echo(a):
            self.runtime.gateway.register_authentication_values(["synthetic-oauth-secret"])
            return "synthetic-oauth-secret"
        runtime = await self.make([{"calls": [{"name": "private", "args": {"n": 1}}]}], [tool("private", credential_echo)])
        with self.assertRaises(TraceStorageError):
            await runtime.run()
        self.assertEqual(self.events("tool.completed"), [])

    async def test_sync_late_receipt_survives_timeout_and_cancellation_under_lock(self):
        import threading
        for cancel in (False,True):
            with self.subTest(cancel=cancel),tempfile.TemporaryDirectory() as directory:
                entered=threading.Event();release=threading.Event();calls=[]
                def worker(args):
                    calls.append(args);entered.set()
                    if not release.wait(5):raise RuntimeError("test barrier timed out")
                    return {"receipt":"receipt-123"}
                runtime=await self.make([{"calls":[{"name":"write","args":{"n":1}}]}],
                    [tool("write",worker)],RuntimeLimits(tool_timeout_seconds=.02),store=FileTraceStore(directory))
                task=asyncio.create_task(runtime.run())
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait,3))
                    if cancel:task.cancel()
                    await asyncio.sleep(.05)
                    self.assertFalse(task.done())
                    with self.assertRaises(Busy):
                        with self.store.execution_guard(runtime.gateway.recorder.episode):pass
                finally:release.set()
                if cancel:
                    with self.assertRaises(asyncio.CancelledError):await task
                else:self.assertEqual((await task).reason,"tool_outcome_unknown")
                events=self.events("tool.outcome_unknown")
                self.assertEqual(len(events),1)
                value=json.loads(self.store.read_payload(events[0]["payload_ref"]))
                self.assertEqual(value["late_result"],{"receipt":"receipt-123"})
                self.assertEqual(value["error"]["reason"],"cancelled" if cancel else "timeout")
                self.assertEqual(len(calls),1);self.assertEqual(len(self.transport.requests),1)

    async def test_parallel_sync_late_receipt_is_recorded_by_owner(self):
        import threading
        for cancel in (True,):
            with self.subTest(cancel=cancel),tempfile.TemporaryDirectory() as directory:
                entered=threading.Event();release=threading.Event();calls=[]
                def worker(args):
                    calls.append(args);entered.set()
                    if not release.wait(5):raise RuntimeError("test barrier timed out")
                    return {"receipt":"receipt-123"}
                runtime=await self.make([{"calls":[{"name":"write","args":{"n":1}}]}],
                    [tool("write",worker,read_only=True,parallel_safe=True)],RuntimeLimits(tool_timeout_seconds=.02,parallel_read_only=True),store=FileTraceStore(directory))
                task=asyncio.create_task(runtime.run())
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait,3))
                    if cancel:task.cancel()
                    await asyncio.sleep(.05)
                    self.assertFalse(task.done())
                    with self.assertRaises(Busy):
                        with self.store.execution_guard(runtime.gateway.recorder.episode):pass
                finally:release.set()
                if cancel:
                    with self.assertRaises(asyncio.CancelledError):await task
                else:self.assertEqual((await task).reason,"tool_outcome_unknown")
                events=self.events("tool.outcome_unknown")
                self.assertEqual(len(events),1)
                value=json.loads(self.store.read_payload(events[0]["payload_ref"]))
                self.assertEqual(value["late_result"],{"receipt":"receipt-123"})
                self.assertEqual(value["error"]["reason"],"cancelled" if cancel else "timeout")
                self.assertEqual(len(calls),1);self.assertEqual(len(self.transport.requests),1)

class GeminiToolLoop(unittest.IsolatedAsyncioTestCase):
    async def test_real_gemini_adapter_with_synthetic_responses_keeps_signature(self):
        from membraneclaw_runtime.providers.gemini import GeminiAdapter
        adapter = GeminiAdapter()
        seen = []
        registry = ToolRegistry([tool("read", lambda a: seen.append(a["n"]) or a)])
        def call(n):
            return {"candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": "read", "id": f"p{n}", "args": {"n": n}},
                 "thoughtSignature": "synthetic-opaque-signature"}]}, "finishReason": "STOP"}]}
        transport = FakeTransport([call(1), call(2), {"candidates": [{"content": {
            "role": "model", "parts": [{"text": "done"}]}, "finishReason": "STOP"}]}])
        route = adapter.initial_state("gemini-2.5-flash").route
        store = MemoryTraceStore()
        gateway = await ModelGateway.start(store, adapter, transport, route)
        runtime = AgentRuntime(gateway, registry, ModelRequest(route, (ContentBlock.text_block("task"),),
                                                              tools=registry.definitions()))
        self.assertEqual((await runtime.run()).status, "completed")
        self.assertEqual(seen, [1, 2])
        native = json.loads(transport.requests[2].body)
        self.assertEqual(native["contents"][1]["parts"][0]["thoughtSignature"], "synthetic-opaque-signature")
        events = store.read_episode(gateway.recorder.episode).events
        starts = [e for e in events if e["type"] == "tool.started"]
        self.assertTrue(all(e["tool_call_id"] and e["model_call_id"] for e in starts))

class DeadlineBudget(unittest.IsolatedAsyncioTestCase):
    async def test_tool_work_consumes_the_whole_execution_time_budget(self):
        now = [0]
        adapter = FakeAdapter()
        store = MemoryTraceStore()
        transport = FakeTransport([{"calls": [{"name": "write", "args": {"n": 1}},
                                               {"name": "write", "args": {"n": 2}}]}])
        invoked = []
        def write(a):
            invoked.append(a["n"])
            now[0] = 100
            return a
        registry = ToolRegistry([tool("write", write)])
        gateway = await ModelGateway.start(store, adapter, transport, adapter.initial_state("fixture").route,
                                          budget=Budget(max_elapsed_seconds=10), clock=lambda: now[0])
        runtime = AgentRuntime(gateway, registry, ModelRequest(gateway.route, (ContentBlock.text_block("task"),),
                                                               tools=registry.definitions()))
        self.assertEqual((await runtime.run()).reason, "execution_budget_exhausted")
        self.assertEqual(invoked, [1])
