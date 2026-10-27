"""Network-free recording playback and strict execution against recorded I/O."""
import copy
import json
from dataclasses import replace

from .schema import Corruption, Payload, canonical_json, digest
from .validation import validate
from ..providers.json_utils import strict_json
from ..providers.types import ConfigurationError, NativeResponse, ToolResult, TransportError


class ReplayMismatch(Exception):pass


def playback(store,episode):
    report=validate(store,episode);ep=store.read_episode(episode)
    return {"mode":"recording_playback","revision":report.revision,"status":report.status,
            "completeness":report.completeness,"events":copy.deepcopy(ep.events),
            "state":strict_json(store.read_payload(ep.checkpoint_ref)) if ep.checkpoint_ref else None}


class RecordedProvider:
    authentication_values=()
    def __init__(self,store,episode):
        self.store,self.index=store,0;self.attempts=[]
        for event in store.read_episode(episode).events:
            if event["type"]=="attempt.started":self.attempts.append({"request":event,"frames":[]})
            elif self.attempts:
                current=self.attempts[-1]
                if event["type"]=="stream.frame":current["frames"].append(event)
                if event["type"]=="attempt.response":current["response"]=event
                if event["type"]=="attempt.failed":current["error"]=event["data"].get("error_code","recorded_transport_error")
    async def exchange(self,native,*,before_send,after_receive,timeout,on_frame=None,after_capture=None):
        from ..streaming import StreamFrame
        if self.index>=len(self.attempts):raise ReplayMismatch("unexpected model request")
        attempt=self.attempts[self.index];self.index+=1
        event=attempt["request"]
        if (native.method,native.path,native.body)!=(event["data"]["method"],event["data"]["path"],self.store.read_payload(event["payload_ref"])):
            raise ReplayMismatch("model request hash or sequence mismatch")
        await before_send(native)
        if on_frame is not None:
            for e in attempt["frames"]:
                d=e["data"]
                frame = StreamFrame(self.store.read_payload(e["payload_ref"]),d["received_at"],d["sequence"],d["complete_frame"])
                await on_frame(frame)
                if after_capture is not None:await after_capture(frame)
        elif attempt["frames"]:raise ReplayMismatch("stream mode mismatch")
        if "response" not in attempt:raise TransportError(attempt.get("error","recorded_unknown_outcome"))
        e=attempt["response"];d=e["data"]
        response=NativeResponse(d["status"],d["headers"],self.store.read_payload(e["payload_ref"]),d["complete"],d.get("error_code"))
        await after_receive(response);return response
    async def exchange_stream(self,*args,**kwargs):return await self.exchange(*args,**kwargs)
    def verify_consumed(self):
        if self.index!=len(self.attempts):raise ReplayMismatch("unconsumed model requests")


class RecordedTool:
    def __init__(self,store,episode):
        self.store,self.index,self.calls=store,0,[];by_id={}
        for e in store.read_episode(episode).events:
            cid=e.get("tool_call_id")
            if e["type"]=="tool.requested":
                record={"request":strict_json(store.read_payload(e["payload_ref"]))};self.calls.append(record);by_id[cid]=record
            elif cid in by_id:
                if e["type"]=="tool.arguments_validated":by_id[cid]["arguments"]=strict_json(store.read_payload(e["payload_ref"]))["arguments"]
                if e["type"]=="tool.validation_failed":by_id[cid]["validation_error"]=strict_json(store.read_payload(e["payload_ref"]))["error"]
                if e["type"] in {"tool.completed","tool.outcome_unknown"}:by_id[cid]["result"]=e
    async def invoke(self,prepared,**kwargs):
        from ..tools import ToolOutcome, outcome
        if self.index>=len(self.calls):raise ReplayMismatch("unexpected tool execution")
        row=self.calls[self.index];self.index+=1;c=prepared.call;r=row["request"]
        if (c.name,c.provider_call_id,c.arguments_raw,prepared.arguments,prepared.validation_error)!=(r["name"],r.get("provider_call_id"),r["arguments_raw"],row.get("arguments"),row.get("validation_error")):
            raise ReplayMismatch("tool ID, arguments hash or sequence mismatch")
        if "result" not in row:raise ReplayMismatch("recorded tool outcome unavailable")
        e=row["result"];d=e["data"];raw=self.store.read_payload(e["payload_ref"])
        kind=d["payload_kind"]
        value=raw if kind=="bytes" else raw.decode() if kind=="utf8" else strict_json(raw)
        result=outcome(c,value,is_error=d["is_error"],known=e["type"]=="tool.completed",code=d.get("error_code"))
        if result.payload.data!=raw:raise ReplayMismatch("tool payload representation changed")
        return result
    def verify_consumed(self):
        if self.index!=len(self.calls):raise ReplayMismatch("unconsumed tool results")


async def execute_replay(store,episode):
    from ..gateway import ModelGateway, Budget, RetryPolicy
    from ..http import restore_request
    from ..providers.registry import default_registry
    from ..providers.types import ModelRequest, ModelRoute, ContentBlock
    from ..runtime import AgentRuntime
    from ..state import RuntimeLimits
    from ..tools import ToolRegistry, ToolSpec
    from .memory_store import MemoryTraceStore
    ep=store.read_episode(episode);validate(store,episode)
    # Version gating is explicit: interrupted/handoff/parallel recordings still support playback.
    if ep.metadata.get("gateway_version")!="gateway_v1" or ep.status!="completed":raise ConfigurationError("execution replay requires a completed gateway_v1 episode")
    starts=[e for e in ep.events if e["type"]=="runtime.started"]
    if len(starts)!=1 or any(e["type"]=="client.handoff" for e in ep.events):raise ConfigurationError("this recording supports playback only")
    initial=strict_json(store.read_payload(starts[0]["payload_ref"]))["request"]
    snapshot=strict_json(store.read_payload(starts[0]["state_after_ref"]))
    if snapshot["application_state"].get("runtime_version")!="runtime_v1":raise ConfigurationError("runtime version unavailable")
    limits=RuntimeLimits(**snapshot["application_state"]["limits"])
    def forbidden(*args,**kwargs):raise ReplayMismatch("real tool invocation forbidden")
    tools=[ToolSpec(t["original_name"],t["description"],t["schema"],forbidden,t["version"],t["read_only"],t["parallel_safe"],t["metadata"])
           for t in initial["tools"]["tools"]]
    registry=ToolRegistry(tools)
    route=ModelRoute(**ep.metadata["route"]);adapter=default_registry().create(route)
    if adapter.adapter_version!=ep.metadata["adapter_version"]:raise ConfigurationError("adapter version unavailable")
    first=next(e for e in ep.events if e["type"]=="attempt.started")
    native=strict_json(store.read_payload(first["payload_ref"]))
    # Reconstruct exact generation config from the native request, without network or SDK imports.
    from ..providers.types import NativeRequest
    content=tuple(ContentBlock(**b) for b in initial["content"])
    request=ModelRequest(route,content,initial["system_instruction"],registry.definitions())
    if route.provider=="gemini":
        cfg=native.get("generationConfig",{})
        request=replace(request,temperature=cfg.get("temperature"),max_output_tokens=cfg.get("maxOutputTokens"),
                        provider_options={"gemini":{"thinkingConfig":cfg["thinkingConfig"]}} if "thinkingConfig" in cfg else {})
    elif route.provider=="openai":
        request=replace(request,temperature=native.get("temperature"),max_output_tokens=native.get("max_output_tokens"),
                        provider_options={"openai":{"reasoning":native["reasoning"]}} if "reasoning" in native else {})
    else:
        options={k:native[k] for k in ("thinking","tool_choice") if k in native}
        # max_tokens default is semantically equivalent but config_hash retains explicit-vs-default distinction.
        explicit=snapshot["provider_state"].get("config_hash")
        request=replace(request,temperature=native.get("temperature"),max_output_tokens=None,
                        provider_options={"anthropic":options} if options else {})
        # Try explicit output value when the first default config does not match the recorded request configuration.
        if adapter.prepare(request,adapter.initial_state(route.model)).config_hash!=strict_json(store.read_payload(next(e for e in ep.events if e["type"]=="model.completed")["payload_ref"]))["config_hash"]:
            request=replace(request,max_output_tokens=native["max_tokens"])
    full_initial = strict_json(store.read_payload(starts[0]["payload_ref"]))
    if "model_request" in full_initial:
        request = restore_request(full_initial["model_request"])
    transport=RecordedProvider(store,episode);executor=RecordedTool(store,episode)
    async def no_wait(delay):pass
    gateway=await ModelGateway.start(MemoryTraceStore(),adapter,transport,route,role=ep.metadata["role"],
        budget=Budget(**ep.metadata["budget"]),retry=RetryPolicy(**ep.metadata["retry"]),sleep=no_wait)
    runtime=AgentRuntime(gateway,registry,request,limits=limits);runtime.executor=executor
    streaming=bool(first["data"]["path"].endswith("alt=sse") or native.get("stream"))
    try:result=await runtime.run(stream=streaming)
    except Exception as error:
        if isinstance(error,ReplayMismatch):raise
        if isinstance(error.__cause__,ReplayMismatch):raise error.__cause__
        raise ReplayMismatch("runtime diverged: "+type(error).__name__) from None
    transport.verify_consumed();executor.verify_consumed()
    final=strict_json(store.read_payload(ep.checkpoint_ref))["provider_state"]
    if gateway.state.to_document()["native_history"]!=final["native_history"] or result.status!=ep.status:
        raise ReplayMismatch("final native state/status mismatch")
    return {"mode":"execution_replay","status":result.status,"text":result.text,"model_requests":transport.index,"tool_calls":executor.index,"network":False}
