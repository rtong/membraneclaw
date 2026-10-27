"""Responses API; explicit local history retains every native output item."""
import copy
from dataclasses import replace
from .native_json import NativeJSONAdapter, token_count
from .json_utils import strict_json
from .types import (Capabilities, ConfigurationError, ContentBlock, ModelOutput,
                    NativeRequest, ProtocolError, ToolCall, Usage)
from ..tracing.schema import canonical_json


class OpenAIAdapter(NativeJSONAdapter):
    provider, api_family, adapter_version = "openai", "responses_v1", "openai_v1"
    history_key, temperature_max, output_limit = "input", 2, 32768
    MODELS = {"gpt-4.1-2025-04-14", "o4-mini-2025-04-16"}

    def capabilities(self, model):
        if model not in self.MODELS:
            raise ConfigurationError("model has no reviewed OpenAI profile")
        return Capabilities(reasoning_options=("effort", "summary") if model.startswith("o4") else ())

    def prepare(self, request, state):
        options, config_hash = self._prepare(request, state)
        if set(options) - {"reasoning"}:
            raise ConfigurationError("unsupported OpenAI option")
        reasoning = options.get("reasoning")
        if reasoning is not None:
            if not request.route.model.startswith("o4") or not isinstance(reasoning, dict) or set(reasoning) - {"effort", "summary"}:
                raise ConfigurationError("unsupported reasoning configuration")
            if reasoning.get("effort", "medium") not in {"low", "medium", "high"} or reasoning.get("summary", "auto") not in {"auto", "concise", "detailed"}:
                raise ConfigurationError("invalid reasoning option")
        if request.route.model.startswith("o4") and request.temperature is not None:
            raise ConfigurationError("temperature is not supported for this reasoning profile")
        history = copy.deepcopy(list(state.native_history))
        if request.content:
            history.append({"role": "user", "content": [{"type": "input_text", "text": b.text} for b in request.content]})
        if not history:
            raise ConfigurationError("input required")
        body = {"model": request.route.model, "input": history, "store": False,
                "include": ["reasoning.encrypted_content"]}
        if request.system_instruction:
            body["instructions"] = request.system_instruction
        if request.tools:
            body["tools"] = [{"type": "function", "name": t.name, "description": t.description,
                              "parameters": copy.deepcopy(t.parameters), "strict": False} for t in request.tools]
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body["max_output_tokens"] = request.max_output_tokens
        if reasoning is not None:
            body["reasoning"] = reasoning
        return NativeRequest(request.route, "POST", "/v1/responses", canonical_json(body), config_hash)

    def normalize(self, response, request, *, response_ref, model_call_id):
        raw = strict_json(response.body)
        if not isinstance(raw, dict) or "error" in raw and raw["error"] is not None or not isinstance(raw.get("output"), list):
            raise ProtocolError("invalid Responses body")
        allowed = {t["name"] for t in strict_json(request.body).get("tools", [])}
        texts, thoughts, calls, ids = [], [], [], set()
        refused, unsupported = False, False
        for i, item in enumerate(raw["output"]):
            if not isinstance(item, dict):
                raise ProtocolError("invalid output item")
            kind = item.get("type")
            if kind == "function_call":
                cid, args = item.get("call_id"), item.get("arguments")
                if not isinstance(cid, str) or not cid or cid in ids or item.get("name") not in allowed or not isinstance(args, str):
                    raise ProtocolError("invalid function call")
                ids.add(cid)
                try:
                    parsed = strict_json(args.encode())
                    if not isinstance(parsed, dict):
                        parsed = {}
                except ProtocolError:
                    parsed = {}  # Runtime records raw arguments and returns a validation error.
                calls.append(ToolCall(f"{model_call_id}_item_{i}", cid, item["name"], args, parsed, i))
            elif kind == "message":
                for b in item.get("content", []):
                    if b.get("type") == "output_text" and isinstance(b.get("text"), str):
                        texts.append(ContentBlock.text_block(b["text"]))
                    elif b.get("type") == "refusal":
                        refused = True
                    else:
                        unsupported = True
            elif kind == "reasoning":
                thoughts.extend(b["text"] for b in item.get("summary", []) if isinstance(b.get("text"), str))
            else:
                unsupported = True
        reason = raw.get("status")
        status = "blocked" if refused else "incomplete" if reason == "incomplete" else "unknown"
        if reason == "completed" and not refused and not unsupported:
            status = "tool_calls" if calls else "completed" if texts else "unknown"
        u = raw.get("usage")
        usage = Usage()
        if u is not None:
            if not isinstance(u, dict):
                raise ProtocolError("invalid usage")
            usage = Usage(token_count(u, "input_tokens"), token_count(u, "output_tokens"), token_count(u, "total_tokens"),
                          token_count(u.get("input_tokens_details") or {}, "cached_tokens"),
                          token_count(u.get("output_tokens_details") or {}, "reasoning_tokens"), copy.deepcopy(u),
                          None if u.get("total_tokens") is not None else "total_not_provided")
        return ModelOutput(tuple(texts), tuple(thoughts), tuple(calls), status, reason, usage,
                           copy.deepcopy(raw), request, response_ref, None, model_call_id)

    def _output_history(self, raw):
        return copy.deepcopy(raw["output"])

    def add_tool_results(self, state, tool_results):
        items = [{"type": "function_call_output", "call_id": c.provider_call_id,
                  "output": canonical_json({"error" if r.is_error else "result": r.result}).decode()}
                 for c, r in self._results(state, tool_results)]
        return replace(state, native_history=state.native_history + tuple(items), pending_calls=())
