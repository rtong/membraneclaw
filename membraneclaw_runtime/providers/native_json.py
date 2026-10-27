"""Shared validation and immutable native history for stateless JSON protocols."""
import copy
import math
import re
from dataclasses import replace

from jsonschema import Draft202012Validator
from ..tracing.schema import canonical_json, digest
from .json_utils import strict_json
from .types import ConfigurationError, ModelRoute, NativeRequest, ProtocolError, ProviderState


class NativeJSONAdapter:
    def initial_state(self, model):
        self.capabilities(model)
        return ProviderState(ModelRoute(self.provider, model, self.api_family), self.adapter_version)

    def _state(self, state):
        if (state.schema_version != 1 or state.adapter_version != self.adapter_version
                or state.route != self.initial_state(state.route.model).route):
            raise ConfigurationError("incompatible provider state")

    def _prepare(self, request, state):
        self._state(state)
        if state.route != request.route or state.pending_calls:
            raise ConfigurationError("route changed or pending tools unresolved")
        if not isinstance(request.system_instruction, str):
            raise ConfigurationError("system instruction must be text")
        if request.temperature is not None and (type(request.temperature) not in (int, float)
                or not math.isfinite(request.temperature) or not 0 <= request.temperature <= self.temperature_max):
            raise ConfigurationError("unsupported temperature")
        if request.max_output_tokens is not None and (type(request.max_output_tokens) is not int
                or not 1 <= request.max_output_tokens <= self.output_limit):
            raise ConfigurationError("unsupported output limit")
        options = request.provider_options
        if not isinstance(options, dict) or set(options) - {self.provider}:
            raise ConfigurationError("provider options require matching namespace")
        options = options.get(self.provider, {})
        if not isinstance(options, dict):
            raise ConfigurationError("invalid provider options")
        names = set()
        for t in request.tools:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", t.name) or t.name in names:
                raise ConfigurationError("invalid or duplicate tool name")
            names.add(t.name)
            try:
                Draft202012Validator.check_schema(t.parameters)
            except Exception:
                raise ConfigurationError("invalid JSON schema") from None
            if t.parameters.get("type") != "object" or not isinstance(t.description, str):
                raise ConfigurationError("tool requires object parameters and text description")
        for b in request.content:
            if b.type != "text" or not isinstance(b.text, str):
                raise ConfigurationError("only text input is supported")
        config = {"system": request.system_instruction, "tools": [vars(t) for t in request.tools],
                  "temperature": request.temperature, "max_output_tokens": request.max_output_tokens,
                  "options": options}
        config_hash = digest(canonical_json(config))
        if state.config_hash is not None and state.config_hash != config_hash:
            raise ConfigurationError("episode configuration changed")
        return copy.deepcopy(options), config_hash

    def advance(self, state, output, tool_results=()):
        self._state(state)
        if state.pending_calls or state.route != output.native_request.route:
            raise ConfigurationError("cannot advance pending or incompatible state")
        if state.config_hash is not None and state.config_hash != output.native_request.config_hash:
            raise ConfigurationError("output configuration mismatch")
        body = strict_json(output.native_request.body)
        history = body[self.history_key]
        if history[:len(state.native_history)] != list(state.native_history):
            raise ConfigurationError("native history does not extend recorded state")
        history = copy.deepcopy(history) + self._output_history(output.native_response)
        new = replace(state, native_history=tuple(history), config_hash=output.native_request.config_hash,
                      pending_calls=output.tool_calls if output.finish_status == "tool_calls" else ())
        return self.add_tool_results(new, tool_results) if tool_results else new

    def _results(self, state, results):
        self._state(state)
        by_id = {r.tool_call_id: r for r in results}
        if not state.pending_calls or len(by_id) != len(results) or set(by_id) != {c.id for c in state.pending_calls}:
            raise ConfigurationError("exactly one result per pending call required")
        for r in results:
            if not isinstance(r.result, dict) or type(r.is_error) is not bool:
                raise ConfigurationError("invalid tool result")
        return [(c, by_id[c.id]) for c in state.pending_calls]


def token_count(data, key):
    v = data.get(key)
    if v is not None and (type(v) is not int or v < 0):
        raise ProtocolError("invalid usage count")
    return v
