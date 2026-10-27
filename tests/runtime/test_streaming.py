import asyncio
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from membraneclaw_runtime.streaming import SSEFramer, StreamAssembler, StreamFrame, prepare_request
from membraneclaw_runtime.providers.openai import OpenAIAdapter
from membraneclaw_runtime.providers.openai_transport import OpenAITransport
from membraneclaw_runtime.providers.anthropic import AnthropicAdapter
from membraneclaw_runtime.providers.types import ContentBlock, ModelRequest, NativeResponse, ProtocolError, TransportError
from membraneclaw_runtime.gateway import ModelGateway, TraceStorageError
from membraneclaw_runtime.tracing import MemoryTraceStore, FileTraceStore
from membraneclaw_runtime.tracing.schema import canonical_json


def frame(value): return b'data: ' + canonical_json(value) + b'\r\n\r\n'
def oa_events():
    item={"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}
    return [{"type":"response.output_text.delta","delta":"hi","sequence_number":0},
            {"type":"response.output_item.done","output_index":0,"item":item,"sequence_number":1},
            {"type":"response.completed","sequence_number":2,"response":{"id":"r1","status":"completed","output":[item]}}]

def ant_events():
    return [{"type":"message_start","message":{"type":"message","role":"assistant","content":[],"usage":{"input_tokens":7,"output_tokens":1}}},
            {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}},
            {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"summary"}},
            {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-"}},
            {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"opaque"}},
            {"type":"content_block_stop","index":0},
            {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"c1","name":"calc","input":{}}},
            {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":'{"x":'}},
            {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":'2}'}},
            {"type":"content_block_stop","index":1},
            {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}},
            {"type":"message_stop"}]

class Assembly(unittest.TestCase):
    def test_chunk_boundaries_are_not_frame_boundaries(self):
        raw=b''.join(frame(v) for v in oa_events());p=SSEFramer();out=[]
        for b in raw:out.extend(p.feed(bytes([b])))
        self.assertEqual(b''.join(out),raw);self.assertFalse(p.buffer)
        a=StreamAssembler("responses_v1")
        for f in out:a.feed(f)
        self.assertTrue(a.finish()[1]);self.assertEqual(a.value["output"][0]["content"][0]["text"],"hi")
    def test_signature_and_arguments_fragmentation(self):
        a=StreamAssembler("messages_v1")
        for e in ant_events():a.feed(frame(e))
        value,done=a.finish();self.assertTrue(done)
        self.assertEqual(value["content"][0]["signature"],"sig-opaque")
        self.assertEqual(value["content"][1]["input"],{"x":2})
        self.assertEqual(value["usage"],{"input_tokens":7,"output_tokens":9})
    def test_interrupt_at_every_signature_and_argument_boundary(self):
        events=ant_events()
        for cut in range(len(events)):
            a=StreamAssembler("messages_v1")
            for e in events[:cut]:a.feed(frame(e))
            self.assertFalse(a.finish()[1])
    def test_duplicates_and_terminal_item_mismatch(self):
        a=StreamAssembler("responses_v1");a.feed(frame(oa_events()[0]))
        with self.assertRaises(ProtocolError):a.feed(frame(oa_events()[0]))
        a=StreamAssembler("responses_v1");a.feed(frame(oa_events()[1]))
        e=copy.deepcopy(oa_events()[2]);e["response"]["output"][0]["content"][0]["text"]="altered"
        with self.assertRaises(ProtocolError):a.feed(frame(e))
    def test_gemini_parts_and_final_usage_not_summed(self):
        a=StreamAssembler("generate_content_v1beta")
        for e in [{"candidates":[{"index":0,"content":{"parts":[{"text":"hi"}]}}],"usageMetadata":{"totalTokenCount":1}},
                  {"candidates":[{"index":0,"content":{"parts":[{"text":"","thoughtSignature":"opaque"}]},"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":5}}]:a.feed(frame(e))
        v,done=a.finish();self.assertTrue(done);self.assertEqual(v["usageMetadata"]["totalTokenCount"],5)
        self.assertEqual(v["candidates"][0]["content"]["parts"][-1],{"text":"","thoughtSignature":"opaque"})

class ScriptedStream:
    authentication_values=()
    def __init__(self,events):self.events=events;self.pulled=0;self.closed=False
    async def exchange_stream(self,request,*,before_send,after_receive,on_frame,timeout,after_capture=None):
        await before_send(request);a=StreamAssembler(request.route.api_family)
        try:
            for i,e in enumerate(self.events):
                raw=frame(e);self.pulled+=1
                await on_frame(StreamFrame(raw,"synthetic",i))
                if after_capture is not None:await after_capture(StreamFrame(raw,"synthetic",i))
                a.feed(raw)
            value,complete=a.finish();r=NativeResponse(200,{},canonical_json(value),complete)
            await after_receive(r);return r
        finally:self.closed=True

class DurableStreaming(unittest.IsolatedAsyncioTestCase):
    async def gateway(self,store,t):
        a=OpenAIAdapter();s=a.initial_state("gpt-4.1-2025-04-14")
        return await ModelGateway.start(store,a,t,s.route),ModelRequest(s.route,(ContentBlock.text_block("hi"),))
    async def test_commit_before_consumer_and_backpressure(self):
        store=MemoryTraceStore();t=ScriptedStream(oa_events());g,r=await self.gateway(store,t)
        async def consumer(f):
            events=store.read_episode(g.recorder.episode).events
            self.assertEqual(events[-1]["type"],"stream.frame")
            self.assertEqual(store.read_payload(events[-1]["payload_ref"]),f.body)
            self.assertEqual(t.pulled,f.sequence+1)
            await asyncio.sleep(.001)
            self.assertEqual(t.pulled,f.sequence+1)
        self.assertEqual((await g.generate(r,stream=True,on_frame=consumer)).text,"hi")
        self.assertTrue(t.closed)
    async def test_partial_stream_no_retry_or_tool_execution(self):
        store=MemoryTraceStore();t=ScriptedStream(oa_events()[:-1]);g,r=await self.gateway(store,t)
        with self.assertRaises(TransportError):await g.generate(r,stream=True)
        self.assertEqual(g.requests_used,1);self.assertEqual(g.capture,"partial")
        self.assertEqual(len([e for e in store.read_episode(g.recorder.episode).events if e["type"]=="stream.frame"]),2)
    async def test_storage_failure_stops_before_forward(self):
        store=MemoryTraceStore();t=ScriptedStream(oa_events());g,r=await self.gateway(store,t);seen=[]
        original=store.commit
        def fail(ep,*args,**kwargs):
            events=kwargs.get("events",args[2] if len(args)>2 else [])
            if any(e.type=="stream.frame" for e in events):raise OSError("disk full")
            return original(ep,*args,**kwargs)
        with patch.object(store,"commit",side_effect=fail):
            with self.assertRaises(TraceStorageError):await g.generate(r,stream=True,on_frame=lambda f:seen.append(f))
        self.assertEqual(seen,[]);self.assertEqual(t.pulled,1);self.assertTrue(t.closed)
    async def test_cancellation_closes_and_marks_unknown(self):
        t=ScriptedStream(oa_events());g,r=await self.gateway(MemoryTraceStore(),t)
        async def stop(f):raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):await g.generate(r,stream=True,on_frame=stop)
        self.assertTrue(t.closed);self.assertEqual(g.external_outcome,"unknown")
    async def test_real_httpx_stream_closes_on_cancel_and_preserves_wire(self):
        import httpx
        raw=b''.join(frame(e) for e in oa_events());closed=[]
        class Bytes(httpx.AsyncByteStream):
            async def __aiter__(self):
                for i in range(0,len(raw),17):yield raw[i:i+17]
            async def aclose(self):closed.append(True)
        async def handler(req):
            self.assertTrue(json.loads(req.content)["stream"])
            return httpx.Response(200,headers={"content-type":"text/event-stream"},stream=Bytes())
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        t=OpenAITransport("synthetic-key");g,r=await self.gateway(MemoryTraceStore(),t)
        with patch("httpx.AsyncClient",return_value=client):
            out=await g.generate(r,stream=True)
        self.assertEqual(out.text,"hi");self.assertTrue(closed)
        ep=g.recorder.store.read_episode(g.recorder.episode)
        self.assertEqual(b''.join(g.recorder.store.read_payload(e["payload_ref"]) for e in ep.events if e["type"]=="stream.frame"),raw)

    async def test_coalesced_chunk_cancel_retains_every_received_byte_once(self):
        import httpx
        raw=b''.join(frame(e) for e in oa_events())+b'data: {"partial":'
        closed=[]; forwarded=[]
        class Bytes(httpx.AsyncByteStream):
            async def __aiter__(self):yield raw
            async def aclose(self):closed.append(True)
        async def handler(req):return httpx.Response(200,stream=Bytes())
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        store=MemoryTraceStore();g,r=await self.gateway(store,OpenAITransport("synthetic-key"))
        async def cancel(f):
            forwarded.append(f.sequence)
            raise asyncio.CancelledError()
        with patch("httpx.AsyncClient",return_value=client):
            with self.assertRaises(asyncio.CancelledError):await g.generate(r,stream=True,on_frame=cancel)
        events=store.read_episode(g.recorder.episode).events
        frames=[e for e in events if e["type"]=="stream.frame"]
        self.assertEqual(b''.join(store.read_payload(e["payload_ref"]) for e in frames),raw)
        self.assertEqual([e["data"]["sequence"] for e in frames],list(range(4)))
        self.assertFalse(frames[-1]["data"]["complete_frame"])
        self.assertEqual(forwarded,[0]);self.assertTrue(closed)
        self.assertEqual(g.capture,"partial");self.assertEqual(g.external_outcome,"unknown")
        self.assertFalse(any(e["type"] in {"tool.started","attempt.response"} for e in events))

    async def test_cancel_during_capture_does_not_duplicate_committed_frame(self):
        import httpx
        import threading
        raw=b''.join(frame(e) for e in oa_events())
        entered=threading.Event();release=threading.Event()
        class Bytes(httpx.AsyncByteStream):
            async def __aiter__(self):yield raw
        async def handler(req):return httpx.Response(200,stream=Bytes())
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        store=MemoryTraceStore();g,r=await self.gateway(store,OpenAITransport("synthetic-key"))
        original=store.commit
        def wait(*args,**kwargs):
            events=kwargs.get("events",args[2] if len(args)>2 else [])
            if any(e.type=="stream.frame" and e.data["sequence"]==0 for e in events):
                entered.set()
                if not release.wait(5):raise RuntimeError("test barrier timed out")
            return original(*args,**kwargs)
        with patch("httpx.AsyncClient",return_value=client),patch.object(store,"commit",side_effect=wait):
            task=asyncio.create_task(g.generate(r,stream=True))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait,3))
                task.cancel();await asyncio.sleep(0)
            finally:release.set()
            with self.assertRaises(asyncio.CancelledError):await task
        frames=[e for e in store.read_episode(g.recorder.episode).events if e["type"]=="stream.frame"]
        self.assertEqual(b''.join(store.read_payload(e["payload_ref"]) for e in frames),raw)
        self.assertEqual(len(frames),3)

    async def test_escaped_credentials_rejected_before_response_or_frame_commit(self):
        import httpx
        from membraneclaw_runtime.tracing.policy import CapturePolicy,CredentialRejected
        secret="synthetic-key"
        bodies=[b'{"text":"\\u0073ynthetic-key"}',
                b'{"nested":[{"value":"prefix-\\u0073ynthetic-key-suffix"}]}',
                b'{"\\u0073ynthetic-key":"key"}',
                b'{"x":"\\u0073ynthetic-key","x":"overwritten"}']
        for body in bodies:
            with self.subTest(body=body):
                with self.assertRaises(CredentialRejected):CapturePolicy([secret]).check(body)
                class Transport:
                    authentication_values=(secret,)
                    async def exchange(self,native,*,before_send,after_receive,timeout):
                        await before_send(native)
                        response=NativeResponse(200,{},body)
                        await after_receive(response);return response
                store=MemoryTraceStore();g,r=await self.gateway(store,Transport())
                with self.assertRaises(TraceStorageError):await g.generate(r)
                self.assertFalse(any(e["type"]=="attempt.response" for e in store.read_episode(g.recorder.episode).events))
                wire=b'data: '+body+b'\n\n'
                class Bytes(httpx.AsyncByteStream):
                    async def __aiter__(self):yield wire
                async def handler(req):return httpx.Response(200,stream=Bytes())
                client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
                store=MemoryTraceStore();g,r=await self.gateway(store,OpenAITransport(secret))
                with patch("httpx.AsyncClient",return_value=client):
                    with self.assertRaises(TraceStorageError):await g.generate(r,stream=True)
                self.assertFalse(any(e["type"]=="stream.frame" for e in store.read_episode(g.recorder.episode).events))
