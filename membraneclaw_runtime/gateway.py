"""Mandatory capture, explicit attempts, route locking and native-state commits."""
from __future__ import annotations

import asyncio
import copy
import math
import random
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from uuid import uuid4

from .providers.types import (BudgetExceeded, ConfigurationError, ProviderHTTPError,
                              ProtocolError, TransportError)
from .tracing import Event, Payload, TraceRecorder
from .tracing.policy import CapturePolicy


class TraceStorageError(Exception):
    pass


@dataclass(frozen=True)
class Budget:
    max_requests: int = 3
    max_elapsed_seconds: float = 120
    attempt_timeout_seconds: float = 30
    max_total_tokens: int | None = None

    def validate(self):
        if type(self.max_requests) is not int or self.max_requests < 1:
            raise ConfigurationError("request budget must be a positive integer")
        for value in (self.max_elapsed_seconds, self.attempt_timeout_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError("time budgets must be finite and positive")
        if self.max_total_tokens is not None and (type(self.max_total_tokens) is not int or self.max_total_tokens < 1):
            raise ConfigurationError("token budget must be a positive integer")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30
    retry_timeouts: bool = False  # New attempts may be billed separately.

    def validate(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ConfigurationError("max attempts must be a positive integer")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in (self.base_delay_seconds, self.max_delay_seconds)):
            raise ConfigurationError("invalid retry delay")
        if type(self.retry_timeouts) is not bool:
            raise ConfigurationError("retry_timeouts must be boolean")


async def disk(function, *args, **kwargs):
    """Off-loop synchronous I/O; settle in-flight writes on cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class ModelGateway:
    def __init__(self, recorder, adapter, transport, route, *, role, budget, retry,
                 sleep=asyncio.sleep, clock=time.monotonic, jitter=random.random):
        self.recorder, self.adapter, self.transport = recorder, adapter, transport
        self.route, self.role, self.budget, self.retry = route, role, budget, retry
        self.state = adapter.initial_state(route.model)
        self.policy = CapturePolicy(transport.authentication_values)
        self.sleep, self.clock, self.jitter = sleep, clock, jitter
        self.started_at = clock()
        self.requests_used = self.tokens_reserved = self.tokens_observed = self.tokens_accounted = 0
        self.usage_unknown = False
        self.capture, self.external_outcome = "complete", "known"
        self._owner = self._last_output = None
        self._failed = False
        self.application_state = {}

    @classmethod
    async def start(cls, store, adapter, transport, route, *, role="solver", budget=None, retry=None,
                    tenant="local", run_id=None, context=None, **test_controls):
        budget, retry = budget or Budget(), retry or RetryPolicy()
        budget.validate()
        retry.validate()
        if not isinstance(role, str) or not role:
            raise ConfigurationError("model role required")
        if adapter.initial_state(route.model).route != route:
            raise ConfigurationError("adapter does not match route")
        metadata = {"route": asdict(route), "adapter_version": adapter.adapter_version,
                    "role": role, "budget": asdict(budget), "retry": asdict(retry),
                    "gateway_version": "gateway_v1", "context": copy.deepcopy(context or {})}
        CapturePolicy(transport.authentication_values).check(metadata)
        recorder = await disk(TraceRecorder.begin_episode, store, metadata, tenant=tenant, run_id=run_id)
        return cls(recorder, adapter, transport, route, role=role, budget=budget, retry=retry, **test_controls)

    @classmethod
    async def restore(cls, store, episode, adapter, transport):
        """Caller holds execution_guard; never resume a running/terminal episode."""
        from .providers.types import ModelRoute, ProviderState, ToolCall
        from .providers.json_utils import strict_json
        ep = await disk(store.read_episode, episode)
        if ep.status != "awaiting_client_tool" or ep.checkpoint_ref is None:
            raise ConfigurationError("episode is not awaiting client tools")
        value = strict_json(await disk(store.read_payload, ep.checkpoint_ref))
        ps = value["provider_state"]
        route = ModelRoute(**ps["route"])
        if adapter.adapter_version != ps["adapter_version"] or adapter.initial_state(route.model).route != route:
            raise ConfigurationError("provider version unavailable")
        g = cls(TraceRecorder(store, episode), adapter, transport, route, role=value["role"],
                budget=Budget(**value["budget"]), retry=RetryPolicy(**value["retry"]))
        g.state = ProviderState(route, ps["adapter_version"], tuple(ps["native_history"]),
                               tuple(ToolCall(**c) for c in ps["pending_calls"]), ps["config_hash"], ps["schema_version"])
        for name in ("requests_used", "tokens_reserved", "tokens_observed", "tokens_accounted",
                     "usage_unknown", "capture", "external_outcome", "application_state"):
            setattr(g, name, copy.deepcopy(value[name]))
        g.started_at -= value.get("elapsed_seconds", 0)
        return g

    async def pause(self, status="awaiting_client_tool", kind="client.handoff"):
        if self._owner is not asyncio.current_task():
            raise ConfigurationError("pause requires execution ownership")
        checkpoint = self._snapshot()
        self.policy.check(checkpoint.data)
        try:
            await disk(self.recorder.checkpoint, __import__("json").loads(checkpoint.data), status=status,
                       event=Event(kind, state_after_ref=checkpoint.ref(self.recorder.episode)))
        except Exception:
            self._failed = True
            raise TraceStorageError("continuation checkpoint failed") from None

    @asynccontextmanager
    async def execution(self):
        """A runtime may hold this context across model + tool + model steps."""
        if self._owner is not None:
            raise ConfigurationError("gateway already executing")
        with self.recorder.store.execution_guard(self.recorder.episode):
            ep = await disk(self.recorder.store.read_episode, self.recorder.episode)
            if ep.revision != self.recorder.revision or ep.status != "running":
                raise ConfigurationError("episode terminal, recovered, or gateway revision stale")
            self._owner = asyncio.current_task()
            try:
                yield self
            finally:
                self._owner = None

    async def _record(self, kind, *, payload=None, checkpoint=None, data=None, **ids):
        event = Event(kind, payload_ref=payload.ref(self.recorder.episode) if payload else None,
                      state_after_ref=checkpoint.ref(self.recorder.episode) if checkpoint else None,
                      data=data or {}, **ids)
        try:
            self.policy.check(asdict(event))
            for item in (payload, checkpoint):
                if item is not None:
                    self.policy.check(item.data)
            await disk(self.recorder.append_event, event, payload=payload, checkpoint=checkpoint)
        except asyncio.CancelledError:
            self._failed = True
            self.capture, self.external_outcome = "partial", "unknown"
            raise
        except Exception:
            self._failed = True
            self.capture, self.external_outcome = "partial", "unknown"
            raise TraceStorageError("trace_storage_error; stop external actions") from None
        return event.event_id

    def _snapshot(self, state=None):
        return Payload.json({"gateway_version": "gateway_v1", "role": self.role,
                             "elapsed_seconds": self.clock() - self.started_at,
                             "provider_state": (state or self.state).to_document(),
                             "application_state": copy.deepcopy(self.application_state),
                             "budget": asdict(self.budget), "retry": asdict(self.retry),
                             "requests_used": self.requests_used, "tokens_reserved": self.tokens_reserved,
                             "tokens_observed": self.tokens_observed, "tokens_accounted": self.tokens_accounted,
                             "usage_unknown": self.usage_unknown,
                             "capture": self.capture, "external_outcome": self.external_outcome})

    def _reserve(self, request):
        remaining = self.budget.max_elapsed_seconds - (self.clock() - self.started_at)
        if self.requests_used >= self.budget.max_requests or remaining <= 0:
            raise BudgetExceeded("request/time budget exhausted")
        reserve = 0
        if self.budget.max_total_tokens is not None:
            bound = request.input_token_upper_bound
            if type(bound) is not int or bound < 0 or request.max_output_tokens is None:
                raise ConfigurationError("token budget requires trusted input upper bound and output limit")
            reserve = bound + request.max_output_tokens
            if self.tokens_accounted + reserve > self.budget.max_total_tokens:
                raise BudgetExceeded("token reservation budget exhausted")
        self.requests_used += 1
        self.tokens_reserved += reserve
        self.tokens_accounted += reserve
        return min(self.budget.attempt_timeout_seconds, remaining), reserve

    def _delay(self, attempt, retry_after):
        if retry_after:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    seconds = (when - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    seconds = -1
            if math.isfinite(seconds) and seconds >= 0:
                return seconds  # Never retry earlier than Retry-After.
        return min(self.retry.max_delay_seconds,
                   self.retry.base_delay_seconds * 2 ** min(attempt - 1, 20) * (1 + self.jitter()))

    async def generate(self, request, *, stream=False, on_frame=None):
        if self._owner is None:
            async with self.execution():
                return await self.generate(request, stream=stream, on_frame=on_frame)
        if self._owner is not asyncio.current_task() or self._failed:
            raise ConfigurationError("gateway busy or failed; start a new episode")
        if request.route != self.route:
            raise ConfigurationError("route change requires a new episode")
        native = self.adapter.prepare(request, self.state)
        if stream:
            from .streaming import prepare_request
            native = prepare_request(native)
        self.policy.check(native.body)
        model_call_id = uuid4().hex
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                timeout, reservation = self._reserve(request)
            except BudgetExceeded:
                self._failed = True
                await self._record("budget.exhausted", checkpoint=self._snapshot(),
                                   model_call_id=model_call_id)
                raise
            attempt_id = uuid4().hex
            ids = {"attempt_id": attempt_id, "model_call_id": model_call_id}
            captured_request = False
            captured_response = None
            response_ref = None

            async def before_send(final):
                nonlocal captured_request
                if captured_request or final != native:
                    raise ProtocolError("transport changed/repeated a prepared request")
                await self._record("attempt.started", payload=Payload(final.body), checkpoint=self._snapshot(),
                                   data={"method": final.method, "path": final.path, "role": self.role,
                                         "attempt": attempt, "timeout_seconds": timeout,
                                         "input_token_upper_bound": request.input_token_upper_bound,
                                         "max_output_tokens": request.max_output_tokens}, **ids)
                captured_request = True

            async def after_receive(response):
                nonlocal captured_response, response_ref
                if not captured_request or captured_response is not None:
                    raise ProtocolError("invalid transport capture order")
                try:
                    self.policy.check(response.body)
                    self.policy.check(response.headers)
                except Exception:
                    self._failed = True
                    self.capture, self.external_outcome = "partial", "unknown"
                    raise TraceStorageError("credential-bearing response rejected before persistence") from None
                if response.encoded_body is not None:
                    await self._record("attempt.encoded_response", payload=Payload(response.encoded_body),
                                       data={"content_encoding": response.headers.get("content-encoding")}, **ids)
                payload = Payload(response.body)
                await self._record("attempt.response", payload=payload,
                                   data={"status": response.status, "headers": response.headers,
                                         "complete": response.complete, "error_code": response.error_code,
                                         "derived_stream_aggregate": stream}, **ids)
                captured_response, response_ref = copy.deepcopy(response), payload.ref(self.recorder.episode)

            frame_count = 0
            async def capture_frame(frame):
                nonlocal frame_count
                if not captured_request or captured_response is not None or frame.sequence != frame_count:
                    raise ProtocolError("invalid stream capture order")
                await self._record("stream.frame", payload=Payload(frame.body),
                                   data={"sequence":frame.sequence, "received_at":frame.received_at,
                                         "complete_frame":frame.complete}, **ids)
                frame_count += 1

            response = None
            try:
                if stream:
                    response = await self.transport.exchange_stream(native, before_send=before_send,
                        after_receive=after_receive, on_frame=capture_frame, timeout=timeout, after_capture=on_frame)
                    await self._record("stream.eof", data={"frames":frame_count, "complete":response.complete}, **ids)
                else:
                    response = await self.transport.exchange(native, before_send=before_send,
                                                              after_receive=after_receive, timeout=timeout)
                if not captured_request or captured_response is None or response != captured_response:
                    raise ProtocolError("transport bypassed or changed mandatory capture")
            except TraceStorageError:
                self._failed = True
                raise
            except asyncio.CancelledError:
                self._failed = True
                self.capture, self.external_outcome = "partial", "unknown"
                self.usage_unknown = True
                await self._record("attempt.cancelled", checkpoint=self._snapshot(), **ids)
                raise
            except TransportError as error:
                self.capture, self.external_outcome = "partial", "unknown"
                self.usage_unknown = True
                await self._record("attempt.failed", checkpoint=self._snapshot(),
                                   data={"error_code": error.code, "external_outcome": "unknown"}, **ids)
                if (stream and frame_count) or not captured_request or not self.retry.retry_timeouts or attempt == self.retry.max_attempts:
                    self._failed = True
                    raise
            except Exception:
                self._failed, self.usage_unknown = True, True
                self.capture, self.external_outcome = "partial", "unknown"
                await self._record("attempt.failed", checkpoint=self._snapshot(),
                                   data={"error_code": "transport_contract_error"}, **ids)
                raise ProtocolError("transport contract violation") from None
            if response is not None and not response.complete:
                self.capture, self.external_outcome, self._failed = "partial", "unknown", True
                self.usage_unknown = True
                await self._record("attempt.partial", checkpoint=self._snapshot(), **ids)
                raise TransportError("partial_response")
            if response is not None and 200 <= response.status < 300:
                try:
                    output = self.adapter.normalize(response, native, response_ref=response_ref,
                                                    model_call_id=model_call_id)
                    next_state = self.adapter.advance(self.state, output)
                except (ProtocolError, ConfigurationError):
                    self._failed, self.usage_unknown = True, True
                    await self._record("attempt.parse_failed", data={"error_code": "provider_protocol_error"}, **ids)
                    raise
                if output.usage.total_tokens is None:
                    self.usage_unknown = True
                else:
                    self.tokens_observed += output.usage.total_tokens
                    self.tokens_accounted += max(0, output.usage.total_tokens - reservation)
                await self._record("model.completed", payload=Payload.json({
                    "text_blocks": [asdict(b) for b in output.text_blocks],
                    "thought_summaries": list(output.thought_summaries),
                    "tool_calls": [asdict(c) for c in output.tool_calls], "finish_status": output.finish_status,
                    "stop_reason": output.stop_reason, "usage": asdict(output.usage),
                    "native_response_ref": output.native_response_ref,
                    "selected_candidate": output.selected_candidate,
                    "config_hash": native.config_hash}), checkpoint=self._snapshot(next_state), **ids)
                self.state, self._last_output = next_state, output
                return output
            if response is not None:
                self.usage_unknown = True
                if response.status >= 500:
                    self.external_outcome = "unknown"  # A proxy/service error cannot prove generation never ran.
                await self._record("attempt.http_error", checkpoint=self._snapshot(),
                                   data={"status": response.status}, **ids)
                if response.status not in {429, 500, 502, 503, 504} or attempt == self.retry.max_attempts:
                    self._failed = True
                    raise ProviderHTTPError(response.status)
            delay = self._delay(attempt, response.headers.get("retry-after") if response else None)
            if self.clock() - self.started_at + delay >= self.budget.max_elapsed_seconds:
                self._failed = True
                await self._record("budget.exhausted", checkpoint=self._snapshot(), **ids)
                raise BudgetExceeded("retry delay exceeds remaining time budget")
            await self._record("attempt.retry_scheduled", data={"delay_seconds": delay}, **ids)
            try:
                await self.sleep(delay)
            except asyncio.CancelledError:
                self._failed = True
                self.capture, self.external_outcome = "partial", "unknown"
                await self._record("attempt.cancelled", checkpoint=self._snapshot(), **ids)
                raise
        raise AssertionError("unreachable")

    async def add_tool_results(self, results):
        if self._owner is None:
            async with self.execution():
                return await self.add_tool_results(results)
        if self._owner is not asyncio.current_task() or self._failed:
            raise ConfigurationError("gateway busy or failed")
        new = self.adapter.add_tool_results(self.state, results)
        await self._record("tool.results_attached", payload=Payload.json([asdict(r) for r in results]),
                           checkpoint=self._snapshot(new))
        self.state = new
        return self.state

    def register_authentication_values(self, values):
        self.policy.register_authentication_values(values)

    async def record_event(self, kind, *, payload=None, data=None, tool_call_id=None, model_call_id=None):
        if self._owner is not asyncio.current_task() or kind.startswith("episode."):
            raise ConfigurationError("application events require execution ownership")
        return await self._record(kind, payload=payload, data=data, checkpoint=self._snapshot(),
                                  tool_call_id=tool_call_id, model_call_id=model_call_id)

    def set_application_state(self, state):
        if self._owner is not asyncio.current_task():
            raise ConfigurationError("application state requires execution ownership")
        value = Payload.json(state)
        self.policy.check(value.data)
        self.application_state = copy.deepcopy(state)

    def remaining_seconds(self):
        return self.budget.max_elapsed_seconds - (self.clock() - self.started_at)

    def mark_external_unknown(self):
        self.capture, self.external_outcome = "partial", "unknown"

    async def finish(self, status="completed"):
        if self._owner is None:
            async with self.execution():
                return await self.finish(status)
        if self._owner is not asyncio.current_task():
            raise ConfigurationError("gateway busy")
        if status == "completed" and (self._failed or self.state.pending_calls or self._last_output is None
                                       or self._last_output.finish_status != "completed"):
            raise ConfigurationError("cannot mark unfinished/blocked/failed generation completed")
        try:
            await disk(self.recorder.finish_episode, status, capture=self.capture,
                       replay="unsupported", external_outcome=self.external_outcome)
        except Exception:
            self._failed = True
            raise TraceStorageError("trace_storage_error; stop external actions") from None
