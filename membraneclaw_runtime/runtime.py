"""Provider-neutral model/tool loop; all external actions follow durable start events."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, replace
from contextlib import AsyncExitStack

from .gateway import TraceStorageError
from .providers.types import BudgetExceeded, ConfigurationError
from .state import RunResult, RuntimeLimits
from .tools import ToolExecutor
from .tracing import Payload


class AgentRuntime:
    def __init__(self, gateway, registry, request, *, limits=None, client_handoff=None, on_finish=None):
        self.gateway, self.registry, self.request = gateway, registry, request
        self.limits = limits or RuntimeLimits()
        self.limits.validate()
        if request.tools != registry.definitions():
            raise ConfigurationError("request tool definitions must match executor registry")
        self.client_handoff, self.on_finish = client_handoff, on_finish
        self.resumed = False
        self.executor = ToolExecutor(registry, self.limits.tool_timeout_seconds)
        self.steps = self.tool_calls = 0
        self.pending = []
        self.next_action = "model"
        self._used = False
        self.current_model_call_id = None

    def _state(self):
        self.gateway.set_application_state({**self.gateway.application_state, "runtime_version": "runtime_v1", "limits": asdict(self.limits),
            "tools": self.registry.document(), "steps_used": self.steps, "tool_calls_used": self.tool_calls,
            "pending": copy.deepcopy(self.pending), "next_action": self.next_action,
            "current_model_call_id": self.current_model_call_id})

    async def _event(self, kind, value=None, data=None):
        self._state()
        return await self.gateway.record_event(kind, payload=Payload.json(value) if value is not None else None, data=data,
            tool_call_id=value.get("tool_call_id") if isinstance(value, dict) else None,
            model_call_id=self.current_model_call_id)

    async def _finish(self, status, reason, text=""):
        self.next_action = "stop"
        if self.on_finish is not None:
            await self.on_finish(self, status, reason, text)
        await self._event("runtime.finished", {"status": status, "reason": reason})
        await self.gateway.finish(status)
        return RunResult(self.gateway.recorder.episode.episode_id, status, reason, text, self.steps, self.tool_calls)

    async def _received(self, prepared, result):
        call = prepared.call
        for pending in self.pending:
            if pending["id"] == call.id:
                pending["status"] = "completed" if result.known else "outcome_unknown"
                pending["result_ref"] = result.payload.ref(self.gateway.recorder.episode)
        if not result.known:
            self.gateway.mark_external_unknown()
        self._state()
        await self.gateway.record_event("tool.completed" if result.known else "tool.outcome_unknown",
            payload=result.payload, tool_call_id=call.id, model_call_id=self.current_model_call_id,
            data={"tool_call_id": call.id, "provider_call_id": call.provider_call_id,
                "tool_name": call.name, "payload_kind": result.payload_kind,
                "is_error": result.result.is_error, "error_code": result.error_code})
        return result

    async def _start(self, prepared):
        if self.gateway.remaining_seconds() <= 0:
            raise BudgetExceeded("execution time budget exhausted")
        call = prepared.call
        await self._event("tool.requested", {"tool_call_id": call.id, "provider_call_id": call.provider_call_id,
                         "name": call.name, "arguments_raw": call.arguments_raw})
        if prepared.validation_error:
            await self._event("tool.validation_failed", {"tool_call_id": call.id, "error": prepared.validation_error})
        else:
            await self._event("tool.arguments_validated", {"tool_call_id": call.id, "arguments": prepared.arguments})
            if prepared.spec.metadata.get("execution") == "client":
                return
            for pending in self.pending:
                if pending["id"] == call.id:
                    pending["status"] = "started"
            await self._event("tool.started", {"tool_call_id": call.id, "name": prepared.spec.name,
                                              "arguments": prepared.arguments})

    async def _invoke(self, prepared, *, on_cancelled_outcome=None):
        remaining = self.gateway.remaining_seconds()
        if remaining <= 0:
            raise BudgetExceeded("execution time budget exhausted before tool action")
        return await self.executor.invoke(prepared, timeout_seconds=min(self.limits.tool_timeout_seconds, remaining),
            on_cancelled_outcome=on_cancelled_outcome or (lambda result: self._received(prepared, result)))

    async def _tools(self, calls):
        prepared = [self.registry.prepare(call) for call in calls]
        self.pending = [{"id": c.id, "name": c.name, "status": "requested", "result_ref": None} for c in calls]
        external = [p for p in prepared if p.spec and p.spec.metadata.get("execution") == "client" and not p.validation_error]
        if external and self.client_handoff is None:
            raise ConfigurationError("client tools require an authorized handoff")
        can_parallel = not external and self.limits.parallel_read_only and all(
            p.spec and p.spec.read_only and p.spec.parallel_safe for p in prepared)
        outcomes = {}
        if not can_parallel:
            for p in prepared:
                await self._start(p)  # A failed commit must invoke no tool.
                if p in external:
                    continue
                result = await self._invoke(p)
                outcomes[p.call.id] = await self._received(p, result)
                if not result.known:
                    break
        else:
            # One recorder/owner commits all events; child tasks only invoke tools.
            for p in prepared:
                await self._start(p)
            cancelled_results = []
            async def invoke(p):
                async def retain(result):
                    cancelled_results.append((p,result))
                return p, await self._invoke(p,on_cancelled_outcome=retain)
            tasks = [asyncio.create_task(invoke(p)) for p in prepared]
            try:
                for task in asyncio.as_completed(tasks):
                    p, result = await task
                    outcomes[p.call.id] = await self._received(p, result)
            except BaseException:
                self.gateway.mark_external_unknown()
                raise
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                settled = await asyncio.gather(*tasks, return_exceptions=True)
                # Child tasks retain evidence; only the execution owner records it.
                for value in settled:
                    if isinstance(value,tuple):
                        p,result = value
                        if p.call.id not in outcomes:
                            outcomes[p.call.id] = await self._received(p,result)
                for p,result in cancelled_results:
                    if p.call.id not in outcomes:
                        outcomes[p.call.id] = await self._received(p,result)
        if any(not value.known for value in outcomes.values()):
            return False
        if external:
            self._state()
            await self.client_handoff(self, external, outcomes)
            return None
        # Protocol assembly always uses original call order, not completion order.
        await self.gateway.add_tool_results(tuple(outcomes[c.id].result for c in calls))
        self.pending = []
        return True

    async def run(self, *, stream=False, on_frame=None):
        if self._used:
            raise ConfigurationError("runtime episodes are single-use; create a new runtime")
        self._used = True
        async with AsyncExitStack() as stack:
            if self.gateway._owner is None:
                await stack.enter_async_context(self.gateway.execution())
            elif self.gateway._owner is not asyncio.current_task():
                raise ConfigurationError("runtime requires execution ownership")
            try:
                await self._event("runtime.resumed" if self.resumed else "runtime.started", {"model_request": self.request.to_document(), "request": {"system_instruction": self.request.system_instruction,
                    "content": [asdict(b) for b in self.request.content], "tools": self.registry.document()}})
                request = self.request
                while self.steps < self.limits.max_steps:
                    self.next_action = "model"
                    self._state()
                    # No trimming: compare the FULL serialized request against an explicit byte cap.
                    native = self.gateway.adapter.prepare(request, self.gateway.state)
                    if self.limits.max_context_bytes is not None and len(native.body) > self.limits.max_context_bytes:
                        return await self._finish("failed", "context_byte_budget_exhausted")
                    self.steps += 1
                    self._state()
                    output = await self.gateway.generate(request, stream=stream, on_frame=on_frame)
                    self.current_model_call_id = output.model_call_id
                    if output.finish_status == "completed" and not output.tool_calls:
                        return await self._finish("completed", "model_completed", output.text)
                    if output.finish_status != "tool_calls":
                        return await self._finish("failed", "model_" + output.finish_status, output.text)
                    if not output.tool_calls:
                        return await self._finish("failed", "empty_tool_call_round")
                    if self.tool_calls + len(output.tool_calls) > self.limits.max_tool_calls:
                        return await self._finish("failed", "tool_call_budget_exhausted")
                    self.tool_calls += len(output.tool_calls)
                    self.next_action = "tools"
                    tool_state = await self._tools(output.tool_calls)
                    if tool_state is None:
                        return RunResult(self.gateway.recorder.episode.episode_id, "awaiting_client_tool", "client_handoff", output.text, self.steps, self.tool_calls)
                    if not tool_state:
                        return await self._finish("failed", "tool_outcome_unknown")
                    request = replace(self.request, content=())
                return await self._finish("failed", "step_budget_exhausted")
            except TraceStorageError:
                # No new external actions or speculative terminal writes to a broken store.
                raise
            except BudgetExceeded:
                return await self._finish("failed", "execution_budget_exhausted")
            except asyncio.CancelledError:
                self.gateway.mark_external_unknown()
                await self._finish("cancelled", "cancelled")
                raise
            except Exception as error:
                await self._event("runtime.error", {"exception_type": type(error).__name__})
                await self._finish("failed", "runtime_error")
                raise
