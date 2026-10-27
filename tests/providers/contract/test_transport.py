"""Real loopback HTTP tests; never contact Google or require credentials."""
import asyncio
import gzip
import http.client
import json
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from membraneclaw_runtime.gateway import Budget, ModelGateway
from membraneclaw_runtime.providers.gemini import GeminiAdapter
from membraneclaw_runtime.providers.gemini_transport import GeminiTransport
from membraneclaw_runtime.providers.types import ConfigurationError, ProviderHTTPError, TransportError
from membraneclaw_runtime.tracing import MemoryTraceStore
from membraneclaw_runtime.tracing.schema import canonical_json

from test_gateway import final, no_wait
from test_gemini import ROUTE, request


@contextmanager
def server(script):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.server.received.append((self.path, body, dict(self.headers)))
            status, headers, response = self.server.script.pop(0)
            self.send_response(status)
            for name, value in headers.items():
                if name != "synthetic-delay":
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            if "synthetic-delay" in headers:
                time.sleep(float(headers["synthetic-delay"]))
            try:
                self.wfile.write(response)
            except (OSError, BrokenPipeError):
                pass

        def do_GET(self):
            self.server.gets += 1
            self.send_response(500)
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.script, httpd.received, httpd.gets = list(script), [], 0
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(2)


def loopback_transport(httpd):
    transport = GeminiTransport("synthetic-loopback-key")
    transport._connection = lambda timeout: http.client.HTTPConnection(
        "127.0.0.1", httpd.server_port, timeout=timeout)
    return transport


class HTTPTransportContracts(unittest.IsolatedAsyncioTestCase):
    async def gateway(self, transport, **kwargs):
        self.store = MemoryTraceStore()
        return await ModelGateway.start(self.store, GeminiAdapter(), transport, ROUTE, sleep=no_wait, **kwargs)

    async def test_final_bytes_retry_count_and_auth_header_exclusion(self):
        with server([(429, {"Retry-After": "0", "Set-Cookie": "private"}, b"busy"),
                     (200, {"Content-Type": "application/json"}, final().body)]) as httpd:
            gw = await self.gateway(loopback_transport(httpd))
            out = await gw.generate(request())
            self.assertEqual(out.text, "synthetic answer")
            self.assertEqual(len(httpd.received), 2)
            events = self.store.read_episode(gw.recorder.episode).events
            starts = [e for e in events if e["type"] == "attempt.started"]
            self.assertEqual(len(starts), len(httpd.received))
            for event, sent in zip(starts, httpd.received):
                self.assertEqual(self.store.read_payload(event["payload_ref"]), sent[1])
                self.assertEqual(sent[2]["x-goog-api-key"], "synthetic-loopback-key")
            for event in events:
                self.assertNotIn("synthetic-loopback-key", json.dumps(event))
                self.assertNotIn("set-cookie", json.dumps(event).lower())
                if event["payload_ref"]:
                    self.assertNotIn(b"synthetic-loopback-key", self.store.read_payload(event["payload_ref"]))

    async def test_redirect_is_one_recorded_error_not_a_second_request(self):
        with server([(302, {"Location": "http://127.0.0.1/redirected"}, b"moved")]) as httpd:
            gw = await self.gateway(loopback_transport(httpd))
            with self.assertRaises(ProviderHTTPError) as caught:
                await gw.generate(request())
            self.assertEqual(caught.exception.status, 302)
            self.assertEqual(len(httpd.received), 1)
            self.assertEqual(httpd.gets, 0)

    async def test_gzip_preserves_encoded_and_client_visible_body(self):
        body = final().body
        encoded = gzip.compress(body, mtime=0)
        with server([(200, {"Content-Encoding": "gzip"}, encoded)]) as httpd:
            gw = await self.gateway(loopback_transport(httpd))
            await gw.generate(request())
            events = self.store.read_episode(gw.recorder.episode).events
            raw = next(e for e in events if e["type"] == "attempt.encoded_response")
            decoded = next(e for e in events if e["type"] == "attempt.response")
            self.assertEqual(self.store.read_payload(raw["payload_ref"]), encoded)
            self.assertEqual(self.store.read_payload(decoded["payload_ref"]), body)

    async def test_invalid_gzip_is_saved_and_not_normalized(self):
        with server([(200, {"Content-Encoding": "gzip"}, b"not-gzip")]) as httpd:
            gw = await self.gateway(loopback_transport(httpd))
            with self.assertRaises(TransportError):
                await gw.generate(request())
            events = self.store.read_episode(gw.recorder.episode).events
            raw = next(e for e in events if e["type"] == "attempt.response")
            self.assertEqual(self.store.read_payload(raw["payload_ref"]), b"not-gzip")
            self.assertEqual(raw["data"]["error_code"], "gzip_decode_failed")

    async def test_callback_failure_sends_nothing(self):
        with server([]) as httpd:
            transport = loopback_transport(httpd)
            native = GeminiAdapter().prepare(request(), GeminiAdapter().initial_state(ROUTE.model))
            async def fail(_):
                raise OSError("synthetic storage failure")
            with self.assertRaises(OSError):
                await transport.exchange(native, before_send=fail, after_receive=fail, timeout=1)
            self.assertEqual(httpd.received, [])

    async def test_timeout_remains_unknown_and_does_not_retry_by_default(self):
        with server([(200, {"synthetic-delay": "0.4"}, final().body)]) as httpd:
            gw = await self.gateway(loopback_transport(httpd),
                                    budget=Budget(max_elapsed_seconds=2, attempt_timeout_seconds=0.1))
            with self.assertRaises(TransportError):
                await gw.generate(request())
            self.assertEqual(len(httpd.received), 1)
            self.assertEqual(gw.external_outcome, "unknown")
            self.assertEqual(gw.capture, "partial")

    async def test_invalid_path_and_missing_credentials_fail_without_io(self):
        with self.assertRaises(ConfigurationError):
            GeminiTransport("")
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ConfigurationError):
            GeminiTransport.from_env()
        adapter = GeminiAdapter()
        native = adapter.prepare(request(), adapter.initial_state(ROUTE.model))
        transport = GeminiTransport("synthetic")
        called = []
        async def capture(value):
            called.append(value)
        with self.assertRaises(ConfigurationError):
            await transport.exchange(replace(native, path="/escape?key=credential"),
                                     before_send=capture, after_receive=capture, timeout=1)
        self.assertEqual(called, [])

    async def test_parser_sees_malformed_bytes_only_after_capture(self):
        from membraneclaw_runtime.providers.types import ProtocolError
        body = b'  {"not": valid json '
        with server([(200, {}, body)]) as httpd:
            gw = await self.gateway(loopback_transport(httpd))
            with self.assertRaises(ProtocolError):
                await gw.generate(request())
            events = self.store.read_episode(gw.recorder.episode).events
            event = next(e for e in events if e["type"] == "attempt.response")
            self.assertEqual(self.store.read_payload(event["payload_ref"]), body)
            self.assertEqual(len(httpd.received), 1)
