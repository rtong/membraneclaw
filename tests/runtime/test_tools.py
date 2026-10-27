import asyncio
import unittest

from membraneclaw_runtime.providers.types import ConfigurationError, ToolCall
from membraneclaw_runtime.tools import ToolExecutor, ToolFailure, ToolRegistry, ToolSpec


class Tools(unittest.IsolatedAsyncioTestCase):
    def spec(self, handler=lambda args: args, **kwargs):
        return ToolSpec("server:read.value", "fixture", {
            "type": "object", "properties": {"n": {"type": "integer", "default": 4, "minimum": 1}},
            "additionalProperties": False}, handler, "fixture_v1", **kwargs)

    def call(self, registry, args):
        return ToolCall("internal", "provider", next(iter(registry.by_provider)), args, args, 0)

    async def test_defaults_are_expanded_then_validated_without_mutating_raw(self):
        registry = ToolRegistry([self.spec()])
        call = self.call(registry, {})
        prepared = registry.prepare(call)
        self.assertEqual(prepared.arguments, {"n": 4})
        self.assertEqual(call.arguments_raw, {})
        result = await ToolExecutor(registry).invoke(prepared)
        self.assertEqual(result.result.result, {"n": 4})
        self.assertFalse(result.result.is_error)

    async def test_invalid_arguments_and_json_do_not_invoke_handler(self):
        invoked = []
        registry = ToolRegistry([self.spec(lambda a: invoked.append(a))])
        for args in [{"n": False}, {"n": 0}, {"extra": 1}, '{bad', '[]']:
            result = await ToolExecutor(registry).invoke(registry.prepare(self.call(registry, args)))
            self.assertTrue(result.result.is_error)
            self.assertTrue(result.known)
        self.assertEqual(invoked, [])

    def test_collision_is_an_error_and_mapping_is_stable(self):
        first = self.spec()
        other = ToolSpec("server/read.value", "fixture", first.parameters, first.handler, "v1")
        with self.assertRaises(ConfigurationError):
            ToolRegistry([first, other])
        registry = ToolRegistry([first])
        self.assertEqual(registry.document()["original_to_provider"][first.name], "server_read_value")
        self.assertEqual(registry.definitions()[0].parameters, first.parameters)

    async def test_known_error_and_unknown_exception_are_distinct(self):
        def known(_):
            raise ToolFailure("fixture domain error")
        def unknown(_):
            raise RuntimeError("possibly performed a write")
        for function, expected in [(known, True), (unknown, False)]:
            registry = ToolRegistry([self.spec(function)])
            result = await ToolExecutor(registry).invoke(registry.prepare(self.call(registry, {})))
            self.assertEqual(result.known, expected)
            self.assertTrue(result.result.is_error)

    async def test_binary_payload_is_exact_and_normalized_with_explicit_encoding(self):
        registry = ToolRegistry([self.spec(lambda _: b'\x00\xff\x01')])
        result = await ToolExecutor(registry).invoke(registry.prepare(self.call(registry, {})))
        self.assertEqual(result.payload.data, b'\x00\xff\x01')
        self.assertEqual(result.result.result, {"encoding": "base64", "data": "AP8B"})

    async def test_timeout_is_unknown_and_invocation_is_not_repeated(self):
        invoked = []
        async def slow(_):
            invoked.append(1)
            await asyncio.sleep(1)
        registry = ToolRegistry([self.spec(slow)])
        result = await ToolExecutor(registry, 0.01).invoke(registry.prepare(self.call(registry, {})))
        self.assertFalse(result.known)
        self.assertEqual(invoked, [1])

class TimeoutConfiguration(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_timeout_override_invokes_nothing(self):
        invoked = []
        spec = ToolSpec("write", "fixture", {"type": "object", "properties": {}}, lambda a: invoked.append(a), "v1")
        registry = ToolRegistry([spec])
        call = ToolCall("id", None, "write", {}, {}, 0)
        with self.assertRaises(ConfigurationError):
            await ToolExecutor(registry).invoke(registry.prepare(call), timeout_seconds=-1)
        self.assertEqual(invoked, [])

    async def test_sync_late_exception_is_retained_without_retry(self):
        import threading
        entered=threading.Event();release=threading.Event();calls=[]
        def worker(args):
            calls.append(1);entered.set()
            if not release.wait(5):raise RuntimeError("test barrier timed out")
            raise LookupError("private exception detail")
        registry=ToolRegistry([ToolSpec("write","fixture",{"type":"object"},worker,"v1")])
        call=ToolCall("id",None,"write",{}, {},0)
        task=asyncio.create_task(ToolExecutor(registry,.01).invoke(registry.prepare(call)))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait,3));await asyncio.sleep(.03)
            self.assertFalse(task.done())
        finally:release.set()
        result=await task
        self.assertFalse(result.known)
        self.assertEqual(result.result.result["late_exception_type"],"LookupError")
        self.assertNotIn(b"private exception detail",result.payload.data)
        self.assertEqual(calls,[1])
