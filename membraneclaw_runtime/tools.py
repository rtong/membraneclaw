"""Tool contracts, portable names, JSON argument validation and bounded invocation."""
from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Callable

from jsonschema import Draft202012Validator

from .providers.json_utils import strict_json
from .providers.types import ConfigurationError, ProtocolError, ToolCall, ToolDefinition, ToolResult
from .tracing.schema import Payload, canonical_json, digest


class ToolFailure(Exception):
    """A handler can declare a known failure that the model may correct."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable
    version: str
    read_only: bool = False
    parallel_safe: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedCall:
    call: ToolCall
    spec: ToolSpec | None
    arguments: dict | None
    validation_error: dict | None


@dataclass(frozen=True)
class ToolOutcome:
    result: ToolResult
    payload: Payload
    payload_kind: str
    known: bool = True
    error_code: str | None = None


def apply_defaults(schema, value):
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        for key, child in schema.get("properties", {}).items():
            if key not in value and "default" in child:
                value[key] = copy.deepcopy(child["default"])
            if key in value:
                value[key] = apply_defaults(child, value[key])
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key in set(value) - set(schema.get("properties", {})):
                value[key] = apply_defaults(additional, value[key])
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        value = [apply_defaults(schema["items"], child) for child in value]
    return value


class ToolRegistry:
    def __init__(self, tools=()):
        self.by_provider = {}
        self.original_to_provider = {}
        for tool in tools:
            if not isinstance(tool.name, str) or not tool.name or not isinstance(tool.version, str) or not tool.version:
                raise ConfigurationError("tool name and version required")
            if not callable(tool.handler):
                raise ConfigurationError("tool handler must be callable")
            if type(tool.read_only) is not bool or type(tool.parallel_safe) is not bool:
                raise ConfigurationError("tool safety flags must be boolean")
            Draft202012Validator.check_schema(tool.parameters)
            if tool.parameters.get("type") != "object":
                raise ConfigurationError("tool parameters must be an object")
            name = re.sub(r"[^A-Za-z0-9_]", "_", tool.name)
            if not re.match(r"[A-Za-z_]", name):
                name = "tool_" + name
            if len(name) > 64:
                name = name[:55] + "_" + digest(tool.name.encode())[:8]
            if name in self.by_provider or tool.name in self.original_to_provider:
                raise ConfigurationError("tool-name mapping collision")
            # Snapshot definitions: caller mutation cannot alter a running tool schema.
            frozen = ToolSpec(tool.name, tool.description, copy.deepcopy(tool.parameters), tool.handler,
                              tool.version, tool.read_only, tool.parallel_safe, copy.deepcopy(tool.metadata))
            self.by_provider[name] = frozen
            self.original_to_provider[tool.name] = name

    def definitions(self):
        return tuple(ToolDefinition(name, tool.description, copy.deepcopy(tool.parameters))
                     for name, tool in self.by_provider.items())

    def document(self):
        return {"original_to_provider": dict(self.original_to_provider),
                "provider_to_original": {p: t.name for p, t in self.by_provider.items()},
                "tools": [{"name": name, "original_name": t.name, "description": t.description,
                           "schema": copy.deepcopy(t.parameters), "schema_hash": digest(canonical_json(t.parameters)),
                           "version": t.version, "read_only": t.read_only, "parallel_safe": t.parallel_safe,
                           "metadata": copy.deepcopy(t.metadata)}
                          for name, t in self.by_provider.items()]}

    def prepare(self, call):
        spec = self.by_provider.get(call.name)
        if spec is None:
            return PreparedCall(call, None, None, {"code": "unknown_tool"})
        try:
            raw = strict_json(call.arguments_raw) if isinstance(call.arguments_raw, str) else copy.deepcopy(call.arguments_raw)
            if not isinstance(raw, dict):
                raise ValueError
            canonical_json(raw)
            arguments = apply_defaults(spec.parameters, raw)
            errors = sorted(Draft202012Validator(spec.parameters).iter_errors(arguments),
                            key=lambda e: str(list(e.absolute_path)))
            if errors:
                # Do not echo arbitrary values/secrets via validator exception strings.
                return PreparedCall(call, spec, None, {"code": "invalid_arguments", "issues": [
                    {"path": list(e.absolute_path), "rule": e.validator} for e in errors]})
            return PreparedCall(call, spec, arguments, None)
        except (ValueError, TypeError, ProtocolError):
            # Malformed provider arguments remain in tool.requested's raw payload.
            return PreparedCall(call, spec, None, {"code": "arguments_not_json_object"})


def outcome(call, raw, *, is_error=False, known=True, code=None):
    if isinstance(raw, bytes):
        payload, kind = Payload(raw), "bytes"
        normalized = {"encoding": "base64", "data": base64.b64encode(raw).decode("ascii")}
    elif isinstance(raw, str):
        payload, kind = Payload(raw.encode()), "utf8"
        normalized = {"output": raw}
    else:
        payload, kind = Payload.json(raw), "json"
        normalized = raw if isinstance(raw, dict) else {"output": raw}
    return ToolOutcome(ToolResult(call.id, copy.deepcopy(normalized), is_error), payload, kind, known, code)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout_seconds=60):
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ConfigurationError("tool timeout must be finite and positive")
        self.registry, self.timeout_seconds = registry, timeout_seconds

    async def invoke(self, prepared, *, timeout_seconds=None, on_cancelled_outcome=None):
        call, spec = prepared.call, prepared.spec
        if prepared.validation_error:
            return outcome(call, {"error": prepared.validation_error}, is_error=True)

        async def run():
            if inspect.iscoroutinefunction(spec.handler):
                return await spec.handler(copy.deepcopy(prepared.arguments))
            return await asyncio.to_thread(spec.handler, copy.deepcopy(prepared.arguments))

        effective_timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if type(effective_timeout) not in (int, float) or not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise ConfigurationError("tool timeout must be finite and positive")
        task = asyncio.create_task(run())
        try:
            value = await asyncio.wait_for(asyncio.shield(task), effective_timeout)
            # Structured MCP errors are known tool results, not transport failures.
            return outcome(call, value, is_error=isinstance(value, dict) and value.get("isError") is True)
        except (TimeoutError, asyncio.CancelledError) as error:
            if inspect.iscoroutinefunction(spec.handler):
                task.cancel()
            late, late_exception = None, None
            # A synchronous action cannot be cancelled. Keep its receipt and keep
            # execution ownership until the worker has actually stopped.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                late = task.result()
            except (Exception, asyncio.CancelledError) as late_error:
                late_exception = type(late_error).__name__
            if isinstance(late, bytes):
                late = {"encoding":"base64", "data":base64.b64encode(late).decode("ascii")}
            cancelled = isinstance(error, asyncio.CancelledError)
            reason = "cancelled" if cancelled else "timeout"
            result = outcome(call, {"error": {"code":"tool_outcome_unknown", "reason":reason},
                                    "late_result":late, "late_exception_type":late_exception},
                             is_error=True, known=False, code=reason)
            if cancelled:
                if on_cancelled_outcome is not None:
                    await on_cancelled_outcome(result)
                raise error
            return result
        except ToolFailure as error:
            return outcome(call, {"error": {"code": "tool_failed", "detail": str(error)}}, is_error=True)
        except Exception as error:
            # The tool may have performed an action before raising. Do not retry it.
            return outcome(call, {"error": {"code": "tool_outcome_unknown", "exception_type": type(error).__name__}},
                           is_error=True, known=False, code="handler_exception")
