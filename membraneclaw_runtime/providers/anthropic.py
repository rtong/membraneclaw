"""Messages API with unmodified assistant blocks and ID-matched tool results."""
import copy
from dataclasses import replace
from .native_json import NativeJSONAdapter, token_count
from .json_utils import strict_json
from .types import (Capabilities, ConfigurationError, ContentBlock, ModelOutput,
                    NativeRequest, ProtocolError, ToolCall, Usage)
from ..tracing.schema import canonical_json


class AnthropicAdapter(NativeJSONAdapter):
    provider, api_family, adapter_version = "anthropic", "messages_v1", "anthropic_v1"
    history_key, temperature_max, output_limit = "messages", 1, 64000
    MODELS = {"claude-sonnet-4-20250514"}

    def capabilities(self, model):
        if model not in self.MODELS:
            raise ConfigurationError("model has no reviewed Anthropic profile")
        return Capabilities(reasoning_options=("thinking",))

    def prepare(self, request, state):
        options, config_hash = self._prepare(request, state)
        if set(options) - {"thinking", "tool_choice"}:
            raise ConfigurationError("unsupported Anthropic option")
        thinking = options.get("thinking")
        maximum = request.max_output_tokens or 4096
        if thinking is not None:
            if not isinstance(thinking, dict) or set(thinking) != {"type", "budget_tokens"} or thinking.get("type") != "enabled":
                raise ConfigurationError("unsupported thinking configuration")
            budget = thinking["budget_tokens"]
            if type(budget) is not int or not 1024 <= budget < maximum or request.temperature is not None:
                raise ConfigurationError("thinking needs 1024 <= budget < max_tokens and default temperature")
        choice = options.get("tool_choice")
        if choice is not None:
            if not request.tools or not isinstance(choice, dict) or set(choice) - {"type", "name", "disable_parallel_tool_use"}:
                raise ConfigurationError("invalid tool choice")
            if choice.get("type") not in {"auto", "any", "tool", "none"}:
                raise ConfigurationError("invalid tool choice type")
            if choice.get("type") == "tool" and choice.get("name") not in {t.name for t in request.tools}:
                raise ConfigurationError("unknown forced tool")
            if "name" in choice and choice.get("type") != "tool":
                raise ConfigurationError("tool name requires forced tool choice")
            if "disable_parallel_tool_use" in choice and type(choice["disable_parallel_tool_use"]) is not bool:
                raise ConfigurationError("invalid parallel flag")
            if thinking is not None and choice.get("type") not in {"auto", "none"}:
                raise ConfigurationError("forced tool choice incompatible with thinking")
        history = copy.deepcopy(list(state.native_history))
        if request.content:
            history.append({"role": "user", "content": [{"type": "text", "text": b.text} for b in request.content]})
        if not history or history[-1].get("role") != "user":
            raise ConfigurationError("new user input or tool results required")
        body = {"model": request.route.model, "messages": history, "max_tokens": maximum}
        if request.system_instruction: body["system"] = request.system_instruction
        if request.tools:
            body["tools"] = [{"name": t.name, "description": t.description,
                              "input_schema": copy.deepcopy(t.parameters)} for t in request.tools]
        if request.temperature is not None: body["temperature"] = request.temperature
        if thinking is not None: body["thinking"] = thinking
        if choice is not None: body["tool_choice"] = choice
        return NativeRequest(request.route, "POST", "/v1/messages", canonical_json(body), config_hash)

    def normalize(self, response, request, *, response_ref, model_call_id):
        raw = strict_json(response.body)
        if not isinstance(raw, dict) or raw.get("type") != "message" or raw.get("role") != "assistant" or not isinstance(raw.get("content"), list):
            raise ProtocolError("invalid Messages body")
        allowed = {t["name"] for t in strict_json(request.body).get("tools", [])}
        texts, thoughts, calls, ids = [], [], [], set()
        unsupported = False
        for i, b in enumerate(raw["content"]):
            if not isinstance(b, dict): raise ProtocolError("invalid content block")
            kind = b.get("type")
            if kind == "text":
                if not isinstance(b.get("text"), str): raise ProtocolError("invalid text")
                texts.append(ContentBlock.text_block(b["text"]))
            elif kind == "thinking":
                if not isinstance(b.get("thinking"), str) or not isinstance(b.get("signature"), str):
                    raise ProtocolError("incomplete thinking block")
                thoughts.append(b["thinking"])
            elif kind == "redacted_thinking":
                if not isinstance(b.get("data"), str): raise ProtocolError("invalid redacted thinking")
            elif kind == "tool_use":
                cid, args = b.get("id"), b.get("input")
                if not isinstance(cid, str) or not cid or cid in ids or b.get("name") not in allowed or not isinstance(args, dict):
                    raise ProtocolError("invalid tool use")
                ids.add(cid)
                calls.append(ToolCall(f"{model_call_id}_block_{i}", cid, b["name"], copy.deepcopy(args), copy.deepcopy(args), i))
            else:
                unsupported = True  # Server tools must never become local executable calls.
        reason = raw.get("stop_reason")
        status = "blocked" if reason == "refusal" else "incomplete" if reason in {"max_tokens", "pause_turn", "model_context_window_exceeded"} else "unknown"
        if not unsupported:
            if reason == "tool_use" and calls: status = "tool_calls"
            elif reason in {"end_turn", "stop_sequence"} and not calls and texts: status = "completed"
        u = raw.get("usage")
        usage = Usage()
        if u is not None:
            if not isinstance(u, dict): raise ProtocolError("invalid usage")
            inp, out = token_count(u, "input_tokens"), token_count(u, "output_tokens")
            cached, created = token_count(u, "cache_read_input_tokens"), token_count(u, "cache_creation_input_tokens")
            # Messages input_tokens excludes cache read/write. Normalize inclusive input explicitly.
            inclusive = inp + (cached or 0) + (created or 0) if inp is not None else None
            total = inclusive + out if inclusive is not None and out is not None else None
            usage = Usage(inclusive, out, total, cached, None, copy.deepcopy(u), None if total is not None else "total_not_available")
        return ModelOutput(tuple(texts), tuple(thoughts), tuple(calls), status, reason, usage,
                           copy.deepcopy(raw), request, response_ref, None, model_call_id)

    def _output_history(self, raw):
        return [{"role": "assistant", "content": copy.deepcopy(raw["content"])}]

    def add_tool_results(self, state, tool_results):
        blocks = [{"type": "tool_result", "tool_use_id": c.provider_call_id,
                   "content": canonical_json(r.result).decode(), "is_error": r.is_error}
                  for c, r in self._results(state, tool_results)]
        return replace(state, native_history=state.native_history + ({"role": "user", "content": blocks},), pending_calls=())
