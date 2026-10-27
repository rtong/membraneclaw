import asyncio
import copy
import tempfile
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from agent.factory import runtime_session
from membraneclaw_runtime.http import HTTPService, dispatch
from membraneclaw_runtime.continuations import ContinuationError
from membraneclaw_runtime.gateway import disk
from membraneclaw_runtime.streaming import StreamFrame
from membraneclaw_runtime.tracing.schema import Busy, Event
from helpers import FakeAdapter, FakeTransport

TOOLS=[{"type":"function","function":{"name":name,"description":"synthetic client tool","parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}} for name in ("client_a","client_b")]
BODY={"messages":[{"role":"user","content":"synthetic client task"}],"tools":TOOLS}
USER=SimpleNamespace(id="synthetic-user")

class StreamFake(FakeTransport):
    async def exchange_stream(self,native,*,before_send,after_receive,on_frame,timeout,after_capture=None):
        async def sent(n):
            await before_send(n)
            f = StreamFrame(b'data: {"synthetic":true}\n\n',"synthetic",0)
            await on_frame(f)
            if after_capture is not None:await after_capture(f)
        return await self.exchange(native,before_send=sent,after_receive=after_receive,timeout=timeout)

class HTTPContinuations(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.t=StreamFake([{"calls":[{"name":"client_a","args":{"x":1}},{"name":"client_b","args":{"x":2}}]}, {"text":"finished"}])
        @asynccontextmanager
        async def factory(**kwargs):
            kwargs["google_token_path"]=None
            async with runtime_session(**kwargs,adapter=FakeAdapter("openai"),transport=self.t) as rt:yield rt
        self.factory=factory
        self.service=HTTPService(trace_dir=self.temp.name,provider="openai",model="synthetic",factory=factory,no_mcp=True)
    async def handoff(self,stream=False):
        body={**BODY,"stream":stream}
        out=await self.service.execute(body,user=USER)
        self.assertEqual(out["trace"]["status"],"awaiting_client_tool")
        prefix=BODY["messages"]+[out["choices"][0]["message"]]
        calls=prefix[-1]["tool_calls"]
        return prefix,[{"role":"tool","tool_call_id":c["id"],"content":"result-"+str(i)} for i,c in enumerate(calls)]
    async def test_restart_partial_duplicate_then_single_resume(self):
        prefix,results=await self.handoff()
        self.service=HTTPService(trace_dir=self.temp.name,provider="openai",model="synthetic",factory=self.factory,no_mcp=True)
        partial=await self.service.execute({"messages":prefix+results[:1]},user=USER)
        self.assertEqual(partial["trace"]["status"],"awaiting_client_tool");self.assertEqual(len(self.t.requests),1)
        repeated=await self.service.execute({"messages":prefix+results[:1]},user=USER)
        self.assertEqual(repeated["trace"]["status"],"awaiting_client_tool")
        for ack in (partial,repeated):
            self.assertNotIn("tool_calls",ack["choices"][0]["message"])
            self.assertEqual(ack["choices"][0]["finish_reason"],"stop")
            self.assertEqual(ack["continuation"]["accepted_handles"],[results[0]["tool_call_id"]])
            self.assertEqual(ack["continuation"]["pending_handles"],[results[1]["tool_call_id"]])
        final=await self.service.execute({"messages":prefix+results[1:]},user=USER)
        self.assertEqual(final["choices"][0]["message"]["content"],"finished");self.assertEqual(len(self.t.requests),2)
        cached=await self.service.execute({"messages":prefix+results},user=USER)
        self.assertEqual(cached["choices"],final["choices"]);self.assertEqual(len(self.t.requests),2)
        ref=self.service.continuations.find(results[0]["tool_call_id"])
        self.assertEqual(self.service.store.read_episode(ref).status,"completed")
    async def test_owner_history_expiry_and_conflict(self):
        prefix,results=await self.handoff()
        for user,msgs,code in [(None,prefix+results,"verified_identity_required"),(SimpleNamespace(id="other"),prefix+results,"owner_mismatch"),
                               (USER,[{"role":"user","content":"edited"}]+prefix[1:]+results,"history_mismatch")]:
            with self.assertRaises(ContinuationError) as caught:await self.service.execute({"messages":msgs},user=user)
            self.assertEqual(caught.exception.code,code)
        await self.service.execute({"messages":prefix+results[:1]},user=USER)
        conflict=copy.deepcopy(results[:1]);conflict[0]["content"]="different"
        with self.assertRaises(ContinuationError) as caught:await self.service.execute({"messages":prefix+conflict},user=USER)
        self.assertEqual(caught.exception.code,"result_conflict")
        self.service.continuations.clock=lambda:float('inf')
        with self.assertRaises(ContinuationError) as caught:await self.service.execute({"messages":prefix+results},user=USER)
        self.assertEqual(caught.exception.code,"continuation_expired");self.assertEqual(len(self.t.requests),1)
    async def test_concurrent_duplicate_only_one_model_request(self):
        prefix,results=await self.handoff();entered=asyncio.Event();release=asyncio.Event();original=self.t.exchange
        async def wait(*args,**kwargs):entered.set();await release.wait();return await original(*args,**kwargs)
        with patch.object(self.t,"exchange",side_effect=wait):
            task=asyncio.create_task(self.service.execute({"messages":prefix+results},user=USER));await entered.wait()
            with self.assertRaises(Busy):await self.service.execute({"messages":prefix+results},user=USER)
            release.set();await task
        self.assertEqual(len(self.t.requests),2)
    async def test_running_after_crash_is_never_reissued(self):
        prefix,results=await self.handoff();ref=self.service.continuations.find(results[0]["tool_call_id"])
        ep=self.service.store.read_episode(ref)
        self.service.store.commit(ref,"synthetic_crash",ep.revision,[Event("synthetic.sent")],[],status="running")
        with self.assertRaises(ContinuationError) as caught:await self.service.execute({"messages":prefix+results},user=USER)
        self.assertEqual(caught.exception.code,"outcome_unknown");self.assertEqual(len(self.t.requests),1)
    async def test_stream_and_nonstream_handoff_same_episode_link(self):
        prefix,results=await self.handoff(stream=True)
        final=await self.service.execute({"messages":prefix+results,"stream":True},user=USER)
        self.assertEqual(final["trace"]["status"],"completed")
        ref=self.service.continuations.find(results[0]["tool_call_id"])
        events=self.service.store.read_episode(ref).events
        self.assertEqual(len([e for e in events if e["type"]=="stream.frame"]),2)
    async def test_asgi_success_and_error_sse_endings(self):
        import httpx
        from fastapi import FastAPI,Request
        app=FastAPI()
        @app.post('/v1/chat/completions')
        async def endpoint(request:Request):return await dispatch(self.service,await request.json(),user=USER)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://synthetic") as client:
            first=await client.post('/v1/chat/completions',json={**BODY,"stream":True})
            self.assertEqual(first.status_code,200);self.assertIn('data: [DONE]',first.text);self.assertIn('tool_calls',first.text)
            self.assertNotIn('synthetic":true',first.text)
    async def test_anonymous_tools_rejected_before_model(self):
        with self.assertRaises(ContinuationError):await self.service.execute(BODY)
        self.assertEqual(len(self.t.requests),0)

    async def test_sse_partial_ack_contains_handles_without_tool_declarations(self):
        import json
        prefix,results=await self.handoff()
        response=await dispatch(self.service,{"messages":prefix+results[:1],"stream":True},user=USER)
        chunks=[]
        async for value in response.body_iterator:
            text=value.decode() if isinstance(value,bytes) else value
            for line in text.splitlines():
                if line.startswith("data: ") and line!="data: [DONE]":chunks.append(json.loads(line[6:]))
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertNotIn("tool_calls",chunk["choices"][0]["delta"])
            self.assertEqual(chunk["continuation"]["accepted_handles"],[results[0]["tool_call_id"]])
            self.assertEqual(chunk["continuation"]["pending_handles"],[results[1]["tool_call_id"]])
        self.assertEqual(len(self.t.requests),1)

class ServerBoundary(unittest.TestCase):
    def test_native_import_has_no_qwen_side_effects(self):
        import subprocess,sys,os
        env={**os.environ,"AGENT_PROVIDER":"openai","AGENT_MODEL":"gpt-4.1-2025-04-14"}
        p=subprocess.run([sys.executable,"-c",'import server,sys; assert "qwen_agent" not in sys.modules; assert "agent.compat" not in sys.modules'],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_legacy_http_mocked_smoke(self):
        import subprocess,sys,os
        code='''from unittest.mock import patch
from fastapi.testclient import TestClient
import server
with patch.object(server,"_first_tool_call_or_text",return_value=("legacy synthetic",None)):
    with TestClient(server.app) as client:
        r=client.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"hello"}]})
        assert r.status_code==200,r.text
        assert r.json()["choices"][0]["message"]["content"]=="legacy synthetic"
'''
        env={**os.environ,"AGENT_PROVIDER":"qwen","AGENT_API_KEY":""}
        p=subprocess.run([sys.executable,"-c",code],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
