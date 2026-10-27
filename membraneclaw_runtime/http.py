"""OpenAI-compatible HTTP service with server-side native continuation state."""
import asyncio
import copy
import os
import time
from dataclasses import asdict, replace
from uuid import uuid4

from .continuations import Continuations, ContinuationError
from .gateway import disk
from .providers.json_utils import strict_json
from .providers.types import (ConfigurationError, ContentBlock, ModelRequest, ModelRoute, ToolDefinition, ToolResult)
from .state import RuntimeLimits
from .tools import ToolSpec, outcome
from .tracing import FileTraceStore
from .tracing.schema import Busy, Event, Payload, canonical_json


def client_tools(definitions, owner, allow=None):
    if definitions and not owner: raise ContinuationError("verified_identity_required",401)
    tools=[]
    for spec in definitions or []:
        if not isinstance(spec,dict) or spec.get("type")!="function":raise ConfigurationError("only client function tools supported")
        f=spec["function"];name=f["name"]
        if allow is not None and not any(p.lower() in name.lower() for p in allow):continue
        def never(**args):raise ConfigurationError("client tool cannot execute on server")
        tools.append(ToolSpec(name,f.get("description",""),f.get("parameters",{"type":"object","properties":{}}),
                              never,"client_v1",metadata={"execution":"client"}))
    return tools


def restore_request(value):
    value=copy.deepcopy(value);value["route"]=ModelRoute(**value["route"])
    value["content"]=tuple(ContentBlock(**b) for b in value["content"])
    value["tools"]=tuple(ToolDefinition(**t) for t in value["tools"])
    return ModelRequest(**value)


class HTTPService:
    def __init__(self, *, trace_dir, provider, model, factory=None, no_mcp=False, allow=None):
        from agent.factory import runtime_session
        self.store=FileTraceStore(trace_dir);self.continuations=Continuations(self.store)
        self.provider,self.model,self.factory,self.no_mcp,self.allow=provider,model,factory or runtime_session,no_mcp,allow

    def _envelope(self,message,episode,status,usage,request_id):
        finish="tool_calls" if status=="awaiting_client_tool" else "stop" if status=="completed" else "error"
        return {"id":"chatcmpl-"+request_id,"object":"chat.completion","created":int(time.time()),"model":self.model,
                "choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":usage,
                "trace":{"schema_version":1,"episode_id":episode.episode_id,"status":status,"request_id":request_id,
                         "locator":{"tenant":episode.tenant,"run_id":episode.run_id,"episode_id":episode.episode_id}}}

    def _usage(self,g):
        if g.usage_unknown:return None
        return {"prompt_tokens":None,"completion_tokens":None,"total_tokens":g.tokens_observed}

    async def execute(self, body, *, user=None, request_id=None, on_frame=None):
        if any(k in body for k in ("temperature","max_tokens","max_completion_tokens","top_p","response_format","tool_choice","parallel_tool_calls")):
            raise ConfigurationError("HTTP generation overrides are not supported by this route")
        if body.get("model") not in (None, self.model, os.environ.get("AGENT_MODEL_ID", "qwen3.5-9b-agent")):
            raise ConfigurationError("model does not match configured route")
        rid=request_id or uuid4().hex;messages=copy.deepcopy(body.get("messages") or [])
        if not messages:raise ConfigurationError("messages required")
        owner=user.id if user else None
        # Only the final consecutive tool messages are results for this handoff.
        split=len(messages)
        while split and messages[split-1].get("role")=="tool":split-=1
        incoming={}
        for m in messages[split:]:
            handle=m.get("tool_call_id");value={"content":m.get("content"),"is_error":m.get("is_error",False)}
            if not isinstance(value["content"],str) or type(value["is_error"]) is not bool:raise ConfigurationError("tool result must be text and boolean error flag")
            if handle in incoming and incoming[handle]!=value:raise ContinuationError("result_conflict")
            incoming[handle]=value
        if incoming:
            if not owner:raise ContinuationError("verified_identity_required",401)
            ref=await disk(self.continuations.find,next(iter(incoming)))
            with self.store.execution_guard(ref):
                ep=await disk(self.store.read_episode,ref)
                state=strict_json(await disk(self.store.read_payload,ep.checkpoint_ref))
                phase,cached=self.continuations.verify(ep,state,owner,messages[:split],incoming)
                if cached:return self._envelope(phase["reply"],ref,phase.get("final_status","awaiting_client_tool"),None,rid)
                request=restore_request(phase["request"])
                if (request.route.provider,request.route.model)!=(self.provider,self.model):raise ContinuationError("provider_disabled")
                # Reconstruct the exact client declarations; never accept changed tools on resume.
                stored=[t for t in state["application_state"]["tools"]["tools"] if t["metadata"].get("execution")=="client"]
                specs=[{"type":"function","function":{"name":t["original_name"],"description":t["description"],"parameters":t["schema"]}} for t in stored]
                if body.get("tools") is not None:
                    supplied=client_tools(body["tools"],owner,self.allow)
                    if [(t.name,t.description,t.parameters) for t in supplied]!=[(t["original_name"],t["description"],t["schema"]) for t in stored]:raise ContinuationError("tool_configuration_changed")
                async with self.factory(provider=self.provider,model=self.model,trace_dir=str(self.store.root),prompt="",no_mcp=self.no_mcp,
                        extra_tools=client_tools(specs,owner),google_token_path=self._google_path(user),resume_episode=ref,
                        request_override=request,limits=RuntimeLimits(**phase["limits"])) as runtime:
                    g=runtime.gateway;g._owner=asyncio.current_task()
                    runtime.steps=state["application_state"]["steps_used"]
                    runtime.tool_calls=state["application_state"]["tool_calls_used"]
                    runtime.pending=copy.deepcopy(state["application_state"].get("pending",[]))
                    try:
                        # Verify once more against the restored checkpoint and mutate only under execution ownership.
                        local=copy.deepcopy(g.application_state)
                        active=next(p for p in local["continuations"] if p["handles"]==phase["handles"])
                        active["results"].update(incoming)
                        g.set_application_state(local)
                        await g.record_event("http.request",payload=Payload.json({"request_id":rid}),data={"request_id":rid})
                        for h,v in incoming.items():
                            await g.record_event("client.result_received",payload=Payload.json({"handle":h,**v}),tool_call_id=active["handles"][h]["call_id"])
                        if set(active["results"])!=set(active["handles"]):
                            await g.pause(kind="client.partial_results")
                            ack = self._envelope({"role":"assistant","content":None},ref,"awaiting_client_tool",None,rid)
                            ack["choices"][0]["finish_reason"] = "stop"
                            ack["continuation"] = {"status":"partial_results",
                                "accepted_handles":list(active["results"]),
                                "pending_handles":[h for h in active["handles"] if h not in active["results"]],
                                "preserve_history_prefix":True}
                            return ack
                        active["status"]="resuming"
                        g.set_application_state(local)
                        await g.pause(status="running",kind="client.resuming")
                        results=[ToolResult(**v) for v in active["local_results"]]
                        for h,v in active["results"].items():
                            call=next(c for c in g.state.pending_calls if c.id==active["handles"][h]["call_id"])
                            result=outcome(call,v["content"],is_error=v["is_error"])
                            await runtime._received(runtime.registry.prepare(call),result)
                            results.append(result.result)
                        await g.add_tool_results(tuple(results))
                        runtime.request=replace(request,content=())
                        runtime.steps=state["application_state"]["steps_used"]
                        runtime.tool_calls=state["application_state"]["tool_calls_used"]
                        runtime.resumed=True
                        return await self._run(runtime,body,user,messages,rid,on_frame)
                    finally:g._owner=None
        # No ordinary cross-turn opaque continuation is inferred from user ID or text.
        if any(m.get("role")=="tool" or m.get("tool_calls") for m in messages):raise ContinuationError("orphan_tool_history")
        portable=[];system=[]
        for m in messages:
            if m.get("role") not in {"system","user","assistant"} or not isinstance(m.get("content"),str):raise ConfigurationError("text chat messages required")
            if m["role"]=="system":system.append(m["content"])
            else:portable.append({"role":m["role"],"text":m["content"]})
        async with self.factory(provider=self.provider,model=self.model,trace_dir=str(self.store.root),
                prompt=canonical_json({"portable_chat_history":portable}).decode(),system_instruction="\n".join(system),
                no_mcp=self.no_mcp,extra_tools=client_tools(body.get("tools"),owner,self.allow),google_token_path=self._google_path(user),
                tenant="http") as runtime:
            async with runtime.gateway.execution():
                await runtime.gateway.record_event("http.request",payload=Payload.json({"request_id":rid}),data={"request_id":rid})
                return await self._run(runtime,body,user,messages,rid,on_frame)

    def _google_path(self,user):
        if user is None:return None
        from agent.google_oauth import token_path_for
        return token_path_for(user.id)

    async def _run(self,runtime,body,user,messages,rid,on_frame):
        async def handoff(rt,prepared,outcomes):
            await self.continuations.issue(rt,prepared,outcomes,owner=user.id if user else None,messages=messages)
        runtime.client_handoff=handoff;runtime.on_finish=self.continuations.complete
        result=await runtime.run(stream=bool(body.get("stream")),on_frame=on_frame)
        g=runtime.gateway
        if result.status=="awaiting_client_tool":message=g.application_state["continuations"][-1]["reply"]
        else:message={"role":"assistant","content":result.text or None}
        return self._envelope(message,g.recorder.episode,result.status,self._usage(g),rid)


async def dispatch(service, body, *, user=None):
    """ASGI response boundary. No raw frames, signatures or private state leave HTTP."""
    import json
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from .gateway import TraceStorageError
    rid=uuid4().hex
    async def execute(on_frame=None):
        try:return await service.execute(body,user=user,request_id=rid,on_frame=on_frame)
        except ContinuationError as e:raise HTTPException(e.status,e.code) from None
        except Busy:raise HTTPException(409,"continuation_busy") from None
        except ConfigurationError:raise HTTPException(400,"configuration_error") from None
        except TraceStorageError:raise HTTPException(503,"trace_storage_error") from None
        except Exception:raise HTTPException(502,"provider_execution_failed") from None
    if not body.get("stream"):
        result=await execute()
        return JSONResponse(result,headers={"X-Request-ID":rid,"X-Trace-Episode-ID":result["trace"]["episode_id"]})
    async def stream():
        queue=asyncio.Queue(maxsize=1)
        async def forwarded(frame):await queue.put(("frame",None))
        async def produce():
            try:await queue.put(("result",await execute(forwarded)))
            except Exception as error:await queue.put(("error",error.detail if hasattr(error,"detail") else "execution_failed"))
        task=asyncio.create_task(produce())
        try:
            while True:
                kind,value=await queue.get()
                if kind=="frame":
                    yield ": durable frame\n\n"  # No opaque provider data in OpenAI-compatible UI.
                    continue
                if kind=="error":
                    yield "data: "+json.dumps({"error":{"code":value},"request_id":rid})+"\n\n"
                else:
                    choice=value["choices"][0]
                    yield "data: "+json.dumps({"id":value["id"],"object":"chat.completion.chunk","model":value["model"],
                        "choices":[{"index":0,"delta":choice["message"],"finish_reason":None}],"trace":value["trace"], **({"continuation":value["continuation"]} if "continuation" in value else {})})+"\n\n"
                    yield "data: "+json.dumps({"id":value["id"],"object":"chat.completion.chunk","model":value["model"],
                        "choices":[{"index":0,"delta":{},"finish_reason":choice["finish_reason"]}],"usage":value["usage"],"trace":value["trace"], **({"continuation":value["continuation"]} if "continuation" in value else {})})+"\n\n"
                yield "data: [DONE]\n\n"
                break
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"X-Request-ID":rid,"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
