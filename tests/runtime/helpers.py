"""Synthetic provider protocol. No production-provider inheritance or network."""
import copy
import json
from dataclasses import asdict, replace

from membraneclaw_runtime.providers.types import (Capabilities, ContentBlock, ModelOutput, ModelRoute,
    NativeRequest, NativeResponse, ProviderState, ToolCall, Usage)
from membraneclaw_runtime.tracing.schema import canonical_json, digest


class FakeAdapter:
    adapter_version = "synthetic_v1"
    def __init__(self, provider="synthetic"):
        self.provider, self.api_family = provider, "synthetic_" + provider

    def capabilities(self, model):
        return Capabilities()

    def initial_state(self, model):
        return ProviderState(ModelRoute(self.provider, model, self.api_family), self.adapter_version)

    def prepare(self, request, state):
        value = {"marker": self.provider, "history": list(state.native_history),
                 "new_user": [asdict(c) for c in request.content], "tools": [asdict(t) for t in request.tools]}
        return NativeRequest(state.route, "POST", "/synthetic", canonical_json(value), "synthetic_config")

    def normalize(self, response, request, *, response_ref, model_call_id):
        value = json.loads(response.body)
        calls = tuple(ToolCall(f"{model_call_id}_{n}", item.get("provider_id"), item["name"],
                               copy.deepcopy(item["args"]), copy.deepcopy(item["args"]), n)
                      for n, item in enumerate(value.get("calls", [])))
        text = (ContentBlock.text_block(value["text"]),) if "text" in value else ()
        status = value.get("status", "tool_calls" if calls else "completed")
        return ModelOutput(text, (), calls, status, "synthetic_stop", Usage(total_tokens=1, unavailable_reason=None),
                           value, request, response_ref, 0, model_call_id)

    def advance(self, state, output):
        request = json.loads(output.native_request.body)
        history = copy.deepcopy(request["history"])
        if request["new_user"]:
            history.append({"user": request["new_user"]})
        history.append({"assistant": output.native_response})
        return replace(state, native_history=tuple(history), pending_calls=copy.deepcopy(output.tool_calls))

    def add_tool_results(self, state, results):
        assert {r.tool_call_id for r in results} == {c.id for c in state.pending_calls}
        return replace(state, native_history=state.native_history + ({"results": [asdict(r) for r in results]},),
                       pending_calls=())


class FakeTransport:
    authentication_values = (b"synthetic-model-key",)
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    async def exchange(self, native, *, before_send, after_receive, timeout):
        await before_send(native)
        self.requests.append(native)
        value = self.responses.pop(0)
        if callable(value):
            value = value(native)
        response = NativeResponse(200, {}, canonical_json(value))
        await after_receive(response)
        return response
