"""Compatibility shims for qwen-agent 0.0.34 against vLLM's OpenAI server."""
import qwen_agent.llm.base as _base
import qwen_agent.llm.oai as _oai

_BaseFunctionCall = _oai.FunctionCall
_orig_conv = _base.BaseChatModel._conv_qwen_agent_messages_to_oai


class _LenientFunctionCall(_BaseFunctionCall):
    # vLLM's first streaming tool_call delta carries name/arguments as None, but
    # qwen_agent.llm.oai builds a FunctionCall from them directly and its pydantic
    # model requires str. Coerce so the deltas can accumulate normally.
    def __init__(self, name=None, arguments=None, **kwargs):
        super().__init__(name=name or "", arguments=arguments or "", **kwargs)


def _conv_with_tool_call_id(messages):
    # qwen_agent labels the tool result with `id`, but the OpenAI schema (and the
    # Qwen chat template) key results off `tool_call_id`. Without it the model
    # never sees the result and returns an empty follow-up turn.
    converted = _orig_conv(messages)
    for msg in converted:
        if msg.get("role") == "tool":
            msg["tool_call_id"] = msg.pop("id", "1")
    return converted


def apply() -> None:
    _oai.FunctionCall = _LenientFunctionCall
    _base.BaseChatModel._conv_qwen_agent_messages_to_oai = staticmethod(_conv_with_tool_call_id)
