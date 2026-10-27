"""Pull-based SSE capture and deterministic native protocol assembly.

One frame per durable commit. No producer queue; storage and consumers backpressure I/O.
"""
import asyncio
import copy
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from .providers.json_utils import strict_json
from .providers.types import NativeResponse, ProtocolError, TransportError
from .tracing.schema import canonical_json


@dataclass(frozen=True)
class StreamFrame:
    body: bytes
    received_at: str
    sequence: int
    complete: bool = True


def prepare_request(native):
    body = strict_json(native.body)
    path = native.path
    if native.route.api_family == "generate_content_v1beta":
        path = path.replace(":generateContent", ":streamGenerateContent?alt=sse")
    else:
        body["stream"] = True
    return replace(native, path=path, body=canonical_json(body))


def unstream_request(native):
    body = strict_json(native.body)
    body.pop("stream", None)
    return replace(native, path=native.path.replace(":streamGenerateContent?alt=sse", ":generateContent"), body=canonical_json(body))


class SSEFramer:
    def __init__(self, maximum=4 * 1024 * 1024):
        self.buffer = b""
        self.maximum = maximum

    def append(self, chunk):
        self.buffer += chunk

    def peek(self):
        match = re.search(b"\r\n\r\n|\r\n\n|\n\r\n|\n\n|\r\r", self.buffer)
        return self.buffer[:match.end()] if match else None

    def acknowledge(self, raw):
        if not self.buffer.startswith(raw):
            raise ProtocolError("stream capture acknowledgement mismatch")
        self.buffer = self.buffer[len(raw):]

    def feed(self, chunk):
        self.buffer += chunk
        frames = []
        while True:
            match = re.search(b"\r\n\r\n|\r\n\n|\n\r\n|\n\n|\r\r", self.buffer)
            if not match:
                break
            frames.append(self.buffer[:match.end()])
            self.buffer = self.buffer[match.end():]
        return frames


def data_of(frame):
    lines = frame.decode("utf-8").splitlines()
    return "\n".join(line[5:].removeprefix(" ") for line in lines if line.startswith("data:"))


class StreamAssembler:
    def __init__(self, api_family):
        self.family = api_family
        self.value = {}
        self.done = False
        self.blocks, self.closed, self.fragments = {}, set(), {}
        self.item_done = {}
        self.last_sequence = -1
        self.final_usage = False

    def feed(self, frame):
        data = data_of(frame)
        if not data or data == "[DONE]":
            return
        event = strict_json(data.encode())
        if not isinstance(event, dict): raise ProtocolError("stream event must be an object")
        if self.done: raise ProtocolError("data after terminal stream event")
        if "error" in event or event.get("type") == "error": raise ProtocolError("provider stream error")
        if self.family == "responses_v1": self._openai(event)
        elif self.family == "messages_v1": self._anthropic(event)
        elif self.family == "generate_content_v1beta": self._gemini(event)
        else: raise ProtocolError("unsupported stream protocol")

    def _openai(self, e):
        sequence = e.get("sequence_number")
        if sequence is not None:
            if type(sequence) is not int or sequence <= self.last_sequence: raise ProtocolError("duplicate or unordered stream event")
            self.last_sequence = sequence
        if e.get("type") == "response.output_item.done":
            item = e["item"]
            key = e["output_index"]
            if key in self.item_done: raise ProtocolError("duplicate completed output item")
            self.item_done[key] = copy.deepcopy(item)
        if e.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
            final = e.get("response")
            if not isinstance(final, dict) or not isinstance(final.get("output"), list): raise ProtocolError("terminal response missing output")
            for index, item in self.item_done.items():
                if index >= len(final["output"]) or final["output"][index] != item: raise ProtocolError("terminal output differs from completed items")
            expected = {"response.completed":"completed", "response.incomplete":"incomplete", "response.failed":"failed"}[e["type"]]
            if final.get("status") != expected: raise ProtocolError("stream status mismatch")
            self.value, self.done = copy.deepcopy(final), True

    def _anthropic(self, e):
        kind = e.get("type")
        if kind == "message_start":
            if self.value: raise ProtocolError("duplicate message start")
            self.value = copy.deepcopy(e["message"])
            self.value["content"] = []
        elif kind == "content_block_start":
            index = e["index"]
            if not self.value or index in self.blocks or index != len(self.blocks): raise ProtocolError("invalid block start")
            self.blocks[index] = copy.deepcopy(e["content_block"])
        elif kind == "content_block_delta":
            index, delta = e["index"], e["delta"]
            if index not in self.blocks or index in self.closed: raise ProtocolError("delta outside open block")
            b, t = self.blocks[index], delta.get("type")
            fields = {"text_delta": ("text", "text"), "thinking_delta": ("thinking", "thinking"), "signature_delta": ("signature", "signature")}
            if t in fields:
                field, source = fields[t]
                if not isinstance(delta.get(source), str): raise ProtocolError("invalid string delta")
                b[field] = b.get(field, "") + delta[source]
            elif t == "input_json_delta":
                if b.get("type") != "tool_use" or not isinstance(delta.get("partial_json"), str): raise ProtocolError("invalid tool delta")
                self.fragments[index] = self.fragments.get(index, "") + delta["partial_json"]
            else: raise ProtocolError("unsupported content delta")
        elif kind == "content_block_stop":
            index = e["index"]
            if index not in self.blocks or index in self.closed: raise ProtocolError("invalid block stop")
            if index in self.fragments:
                self.blocks[index]["input"] = strict_json(self.fragments[index].encode())
            self.closed.add(index)
        elif kind == "message_delta":
            if not self.value: raise ProtocolError("message delta without start")
            self.value.update(copy.deepcopy(e.get("delta", {})))
            if "usage" in e:
                self.value.setdefault("usage", {}).update(copy.deepcopy(e["usage"]))
                self.final_usage = "output_tokens" in e["usage"]
        elif kind == "message_stop":
            if not self.value or set(self.blocks) != self.closed or not self.value.get("stop_reason"): raise ProtocolError("premature message stop")
            self.value["content"] = [self.blocks[i] for i in range(len(self.blocks))]
            if not self.final_usage: self.value.pop("usage", None)
            self.done = True
        elif kind != "ping":
            raise ProtocolError("unsupported Messages stream event")

    def _gemini(self, e):
        # Parts are kept in received order, including empty text parts bearing signatures.
        # We never merge signed parts or infer signature ownership from adjacent text.
        candidates = self.value.setdefault("candidates", [])
        for candidate in e.get("candidates", []):
            index = candidate.get("index", 0)
            if type(index) is not int or not 0 <= index <= 15: raise ProtocolError("invalid candidate index")
            while len(candidates) <= index: candidates.append({"index":len(candidates),"content":{"role":"model","parts":[]}})
            target = candidates[index]
            if "finishReason" in target and candidate.get("content", {}).get("parts"):
                raise ProtocolError("candidate data after finish")
            target["content"]["parts"].extend(copy.deepcopy(candidate.get("content", {}).get("parts", [])))
            target.update(copy.deepcopy({k:v for k,v in candidate.items() if k != "content"}))
        for k,v in e.items():
            if k != "candidates": self.value[k] = copy.deepcopy(v)
        # Gemini has no separate done event. Completion requires explicit finish reasons plus EOF.
        self.final_usage = "usageMetadata" in e

    def finish(self):
        if self.family == "generate_content_v1beta":
            cs = self.value.get("candidates", [])
            self.done = bool(cs) and all(c.get("finishReason") for c in cs) or bool(self.value.get("promptFeedback", {}).get("blockReason"))
            if not self.final_usage: self.value.pop("usageMetadata", None)
        return copy.deepcopy(self.value), self.done


async def exchange_stream(transport, request, *, before_send, after_receive, on_frame, timeout, after_capture=None):
    import httpx
    from .providers.gemini_transport import SAFE_RESPONSE_HEADERS
    transport._validate_request(unstream_request(request), timeout)
    started = time.monotonic()
    await before_send(request)
    timeout -= time.monotonic() - started
    if timeout <= 0:
        raise TransportError("deadline_before_send")
    assembler, framer, sequence = StreamAssembler(request.route.api_family), SSEFramer(), 0
    async def capture(raw, complete=True):
        nonlocal sequence
        frame = StreamFrame(raw, datetime.now(timezone.utc).isoformat(), sequence, complete)
        # Cancellation cannot leave an ambiguous commit acknowledgement. Drain
        # the capture task before removing its bytes from the pending buffer.
        task = asyncio.create_task(on_frame(frame))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        framer.acknowledge(raw)
        sequence += 1
        if cancelled:
            raise asyncio.CancelledError()
        return frame

    async def consume(raw, complete=True):
        frame = await capture(raw, complete)
        if after_capture is not None:
            await after_capture(frame)
        if not complete or len(raw) > framer.maximum: raise ProtocolError("incomplete or oversized SSE frame")
        assembler.feed(raw)

    async def flush_received():
        # Capture only: never parse, forward, or execute anything during cleanup.
        while framer.buffer:
            raw = framer.peek()
            await capture(raw or framer.buffer, raw is not None)

    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
                async with client.stream("POST", "https://" + transport.host + request.path, content=request.body,
                                         headers=transport._headers()) as response:
                    headers = {k:v for k,v in response.headers.items() if k in SAFE_RESPONSE_HEADERS}
                    if not 200 <= response.status_code < 300:
                        raw = await response.aread()
                        result = NativeResponse(response.status_code, headers, raw)
                    else:
                        if headers.get("content-encoding", "identity") != "identity": raise ProtocolError("encoded SSE not supported")
                        async for chunk in response.aiter_raw():
                            framer.append(chunk)
                            while (raw := framer.peek()) is not None:
                                await consume(raw)
                            if len(framer.buffer) > framer.maximum:
                                await consume(framer.buffer, False)
                        if framer.buffer: await consume(framer.buffer, False)
                        value, complete = assembler.finish()
                        result = NativeResponse(response.status_code, headers, canonical_json(value), complete,
                                                None if complete else "stream_missing_terminal")
                    await after_receive(result)
                    return result
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(flush_received())
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
        raise
    except (httpx.HTTPError, TimeoutError):
        await flush_received()
        raise TransportError("stream_transport_failed") from None
