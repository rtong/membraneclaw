"""Gemini Developer API GenerateContent v1beta, adapter schema gemini_v1."""
from __future__ import annotations

import copy
import math
import re
from dataclasses import replace

from ..tracing.schema import canonical_json, digest
from .types import (Capabilities, ConfigurationError, ContentBlock, ModelOutput,
                    ModelRoute, NativeRequest, ProtocolError, ProviderState, ToolCall, Usage)


from .json_utils import strict_json


def validate_schema(schema):
    """Conservative subset, preserved unchanged in parametersJsonSchema."""
    allowed = {"type", "description", "properties", "required", "items", "enum",
               "additionalProperties", "minimum", "maximum", "minItems", "maxItems", "default", "title"}
    if not isinstance(schema, dict) or set(schema) - allowed:
        raise ConfigurationError("unsupported tool schema keyword")
    kind = schema.get("type")
    if kind not in {"object", "array", "string", "number", "integer", "boolean", "null"}:
        raise ConfigurationError("tool schema requires a supported type")
    if "description" in schema and not isinstance(schema["description"], str):
        raise ConfigurationError("description must be text")
    if "properties" in schema:
        if kind != "object" or not isinstance(schema["properties"], dict):
            raise ConfigurationError("invalid tool properties")
        for child in schema["properties"].values():
            validate_schema(child)
    if "items" in schema:
        if kind != "array":
            raise ConfigurationError("items require an array")
        validate_schema(schema["items"])
    if "required" in schema:
        required = schema["required"]
        if kind != "object" or not isinstance(required, list) or not all(isinstance(k, str) for k in required):
            raise ConfigurationError("invalid required properties")
        if len(set(required)) != len(required) or set(required) - set(schema.get("properties", {})):
            raise ConfigurationError("required property not declared")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ConfigurationError("enum must be a nonempty array")
    if "additionalProperties" in schema:
        additional = schema["additionalProperties"]
        if kind != "object":
            raise ConfigurationError("additionalProperties requires an object")
        if isinstance(additional, dict):
            validate_schema(additional)
        elif type(additional) is not bool:
            raise ConfigurationError("additionalProperties requires boolean or schema")
    for key in ("minimum", "maximum"):
        if key in schema and (kind not in {"number", "integer"} or type(schema[key]) not in (int, float)
                              or not math.isfinite(schema[key])):
            raise ConfigurationError("numeric bound requires a finite number")
    for key in ("minItems", "maxItems"):
        if key in schema and (kind != "array" or type(schema[key]) is not int or schema[key] < 0):
            raise ConfigurationError("array bound requires a nonnegative integer")
    canonical_json(schema)


class GeminiAdapter:
    provider = "gemini"
    api_family = "generate_content_v1beta"
    adapter_version = "gemini_v1"
    MODELS = {"gemini-2.5-flash"}  # Reviewed profile, not a latest-model claim.

    def capabilities(self, model):
        if model not in self.MODELS:
            raise ConfigurationError("model has no reviewed Gemini capability profile")
        return Capabilities(reasoning_options=("thinkingBudget", "includeThoughts"))

    def initial_state(self, model):
        self.capabilities(model)
        return ProviderState(ModelRoute(self.provider, model, self.api_family), self.adapter_version)

    def _state(self, state):
        if (state.schema_version != 1 or state.adapter_version != self.adapter_version
                or state.route.provider != self.provider or state.route.api_family != self.api_family):
            raise ConfigurationError("incompatible provider state")
        self.capabilities(state.route.model)

    def prepare(self, request, state):
        self._state(state)
        if request.route != state.route:
            raise ConfigurationError("provider/API/model cannot change inside an episode")
        if state.pending_calls:
            raise ConfigurationError("matching tool results required before the next model call")
        if not isinstance(request.system_instruction, str):
            raise ConfigurationError("system instruction must be text")
        if request.temperature is not None and (type(request.temperature) not in (int, float)
                or not math.isfinite(request.temperature) or not 0 <= request.temperature <= 2):
            raise ConfigurationError("Gemini temperature must be finite and in 0..2")
        if request.max_output_tokens is not None and (type(request.max_output_tokens) is not int
                                                     or not 1 <= request.max_output_tokens <= 65536):
            raise ConfigurationError("max output tokens must be an integer in 1..65536")
        options = request.provider_options
        if not isinstance(options, dict) or set(options) - {"gemini"}:
            raise ConfigurationError("provider options require the gemini namespace")
        gemini = options.get("gemini", {})
        if not isinstance(gemini, dict) or set(gemini) - {"thinkingConfig"}:
            raise ConfigurationError("unsupported Gemini option")
        thinking = gemini.get("thinkingConfig", {})
        if not isinstance(thinking, dict) or set(thinking) - {"thinkingBudget", "includeThoughts"}:
            raise ConfigurationError("unsupported Gemini reasoning option")
        if "thinkingBudget" in thinking:
            budget = thinking["thinkingBudget"]
            if type(budget) is not int or not (budget == -1 or 0 <= budget <= 24576):
                raise ConfigurationError("invalid thinkingBudget for gemini-2.5-flash")
        if "includeThoughts" in thinking and type(thinking["includeThoughts"]) is not bool:
            raise ConfigurationError("includeThoughts must be boolean")
        declarations, names = [], set()
        for tool in request.tools:
            if (not isinstance(tool.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", tool.name)
                    or tool.name in names):
                raise ConfigurationError("invalid or duplicate portable tool name")
            names.add(tool.name)
            validate_schema(tool.parameters)
            if tool.parameters["type"] != "object" or not isinstance(tool.description, str):
                raise ConfigurationError("tool needs object parameters and text description")
            declarations.append({"name": tool.name, "description": tool.description,
                                 "parametersJsonSchema": copy.deepcopy(tool.parameters)})
        config = {"system_instruction": request.system_instruction, "tools": declarations,
                  "temperature": request.temperature, "max_output_tokens": request.max_output_tokens,
                  "provider_options": copy.deepcopy(options)}
        config_hash = digest(canonical_json(config))
        if state.config_hash is not None and state.config_hash != config_hash:
            raise ConfigurationError("episode configuration changed; start a child episode")
        contents = copy.deepcopy(list(state.native_history))
        if request.content:
            parts = []
            for block in request.content:
                if block.type != "text" or not isinstance(block.text, str):
                    raise ConfigurationError("this adapter accepts text input only")
                parts.append({"text": block.text})
            contents.append({"role": "user", "parts": parts})
        if not contents or contents[-1].get("role") != "user":
            raise ConfigurationError("new user input or completed tool results required")
        body = {"contents": contents}
        if request.system_instruction:
            body["systemInstruction"] = {"parts": [{"text": request.system_instruction}]}
        if declarations:
            body["tools"] = [{"functionDeclarations": declarations}]
        generation = {k: v for k, v in {"temperature": request.temperature,
                      "maxOutputTokens": request.max_output_tokens}.items() if v is not None}
        if thinking:
            generation["thinkingConfig"] = copy.deepcopy(thinking)
        if generation:
            body["generationConfig"] = generation
        return NativeRequest(request.route, "POST", f"/v1beta/models/{request.route.model}:generateContent",
                             canonical_json(body), config_hash)

    def normalize(self, response, request, *, response_ref, model_call_id):
        raw = strict_json(response.body)
        if not isinstance(raw, dict) or "error" in raw:
            raise ProtocolError("unexpected GenerateContent response")
        usage_data = raw.get("usageMetadata")
        if usage_data is None:
            usage = Usage()
        else:
            if not isinstance(usage_data, dict):
                raise ProtocolError("invalid usage metadata")
            def count(key):
                value = usage_data.get(key)
                if value is not None and (type(value) is not int or value < 0):
                    raise ProtocolError("invalid usage token count")
                return value
            usage = Usage(count("promptTokenCount"), count("candidatesTokenCount"), count("totalTokenCount"),
                          count("cachedContentTokenCount"), count("thoughtsTokenCount"),
                          copy.deepcopy(usage_data), None if usage_data.get("totalTokenCount") is not None else "total_not_provided")
        candidates = raw.get("candidates", [])
        if not isinstance(candidates, list):
            raise ProtocolError("invalid candidates")
        text, thoughts, calls = [], [], []
        selected, reason, status = None, None, "unknown"
        feedback = raw.get("promptFeedback", {})
        if not isinstance(feedback, dict):
            raise ProtocolError("invalid prompt feedback")
        if not candidates:
            if feedback.get("blockReason"):
                reason, status = feedback["blockReason"], "blocked"
        else:
            selected = 0  # Select first array entry; retain ALL candidates in raw.
            candidate = candidates[0]
            if not isinstance(candidate, dict):
                raise ProtocolError("invalid candidate")
            reason = candidate.get("finishReason")
            if reason is not None and not isinstance(reason, str):
                raise ProtocolError("invalid finish reason")
            content = candidate.get("content", {})
            if not isinstance(content, dict) or not isinstance(content.get("parts", []), list):
                raise ProtocolError("invalid candidate content")
            allowed_tools = {f["name"] for tool in strict_json(request.body).get("tools", [])
                             for f in tool.get("functionDeclarations", [])}
            provider_ids = set()
            unsupported_part = False
            for position, part in enumerate(content.get("parts", [])):
                if not isinstance(part, dict):
                    raise ProtocolError("invalid content part")
                if "text" in part:
                    if not isinstance(part["text"], str):
                        raise ProtocolError("invalid text part")
                    if part.get("thought") is True:
                        thoughts.append(part["text"])
                    else:
                        text.append(ContentBlock.text_block(part["text"]))
                if "functionCall" in part:
                    call = part["functionCall"]
                    if not isinstance(call, dict) or call.get("name") not in allowed_tools:
                        raise ProtocolError("unknown or invalid tool call")
                    args, provider_id = call.get("args", {}), call.get("id")
                    if not isinstance(args, dict) or (provider_id is not None and
                            (not isinstance(provider_id, str) or not provider_id or provider_id in provider_ids)):
                        raise ProtocolError("invalid tool arguments or duplicate provider call ID")
                    if provider_id is not None:
                        provider_ids.add(provider_id)
                    calls.append(ToolCall(f"{model_call_id}_part_{position}", provider_id, call["name"],
                                          copy.deepcopy(args), copy.deepcopy(args), position))
                if not ("text" in part or "functionCall" in part):
                    unsupported_part = True
            blocked = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
            if reason in blocked or feedback.get("blockReason"):
                status = "blocked"
            elif reason == "MAX_TOKENS":
                status = "incomplete"
            elif reason == "STOP" and not unsupported_part:
                status = "tool_calls" if calls else ("completed" if text else "unknown")
            elif reason and reason != "FINISH_REASON_UNSPECIFIED":
                status = "incomplete"
        return ModelOutput(tuple(text), tuple(thoughts), tuple(calls), status, reason, usage,
                           copy.deepcopy(raw), request, response_ref, selected, model_call_id)

    def advance(self, state, output, tool_results=()):
        self._state(state)
        if state.pending_calls or state.route != output.native_request.route:
            raise ConfigurationError("cannot advance incompatible or pending state")
        if state.config_hash is not None and state.config_hash != output.native_request.config_hash:
            raise ConfigurationError("output configuration does not match episode")
        history = copy.deepcopy(strict_json(output.native_request.body)["contents"])
        if history[:len(state.native_history)] != list(state.native_history):
            raise ConfigurationError("native request does not extend the recorded state")
        if output.selected_candidate is not None:
            content = output.native_response["candidates"][output.selected_candidate].get("content")
            if content is not None:
                if content.get("role", "model") != "model":
                    raise ProtocolError("candidate role is not model")
                content = copy.deepcopy(content)
                content.setdefault("role", "model")
                history.append(content)  # Signatures, part ordering, unknown fields unchanged.
        new = ProviderState(state.route, self.adapter_version, tuple(history),
                            copy.deepcopy(output.tool_calls) if output.finish_status == "tool_calls" else (),
                            output.native_request.config_hash)
        return self.add_tool_results(new, tool_results) if tool_results else new

    def add_tool_results(self, state, tool_results):
        self._state(state)
        results = {result.tool_call_id: result for result in tool_results}
        if (not state.pending_calls or len(results) != len(tool_results)
                or set(results) != {c.id for c in state.pending_calls}):
            raise ConfigurationError("provide exactly one result per pending call")
        parts = []
        for call in state.pending_calls:
            result = results[call.id]
            if not isinstance(result.result, dict) or type(result.is_error) is not bool:
                raise ConfigurationError("tool result needs a JSON object and boolean error flag")
            response = {"name": call.name, "response": {
                "error" if result.is_error else "result": copy.deepcopy(result.result)}}
            if call.provider_call_id is not None:
                response["id"] = call.provider_call_id
            parts.append({"functionResponse": response})
        canonical_json(parts)
        return replace(state, native_history=state.native_history + ({"role": "user", "parts": parts},), pending_calls=())
