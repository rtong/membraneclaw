"""Bridge client-declared (Open WebUI) tools into the Qwen-Agent loop.

Open WebUI owns the code for its tools, so we cannot execute them here. Instead we
declare them to the agent as stubs; when the agent decides to call one, the stub
returns a sentinel, we abandon the run, and hand the call back to the client as a
normal OpenAI `tool_calls` response. The client executes it and replays the
conversation with a `tool` message, which we convert back into qwen-agent's
function-message form so the agent resumes where it left off.
"""
import json
import logging
from typing import List, Optional, Tuple

from qwen_agent.tools.base import BaseTool

logger = logging.getLogger("membraneclaw")

# Returned by a stub tool to mean "stop, this one belongs to the client".
DEFER_SENTINEL = "__membraneclaw_defer_to_client__"

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}


def normalize_schema(params) -> dict:
    """Coerce a client parameters block into what qwen-agent will accept.

    qwen_agent.tools.base.is_tool_schema asserts the top level is *exactly*
    {type, properties, required}. Open WebUI omits `required` when every argument
    has a default, and Pydantic-derived schemas add `title`/`additionalProperties`
    /`$defs` — all of which fail that equality check even though they are valid
    JSON Schema. Keep the three expected keys and drop the rest.
    """
    if not isinstance(params, dict):
        return dict(_EMPTY_SCHEMA)
    properties = params.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = params.get("required")
    if not isinstance(required, list):
        required = []
    # `required` must be a subset of `properties`, or the assert trips.
    required = [r for r in required if r in properties]
    return {"type": "object", "properties": properties, "required": required}


class DeferredTool(BaseTool):
    """Declaration-only tool: advertises the schema, never does the work."""

    def __init__(self, spec: dict):
        fn = spec.get("function") or {}
        self.name = fn.get("name") or "unnamed_tool"
        self.description = (fn.get("description") or "").strip()
        self.parameters = normalize_schema(fn.get("parameters"))
        super().__init__()

    def call(self, params, **kwargs) -> str:
        return DEFER_SENTINEL


def client_tools(
    tools: Optional[List[dict]], allow: Optional[set] = None
) -> List[DeferredTool]:
    """allow=None accepts every client tool; otherwise a case-insensitive substring
    match against each pattern. Substring rather than exact because Open WebUI
    renames a tool to `{tool_id}_{name}` when its name collides with a built-in, so
    an exact allowlist silently drops the very tool you meant to keep."""
    out = []
    for spec in tools or []:
        if (spec or {}).get("type") != "function":
            continue
        name = (spec.get("function") or {}).get("name") or ""
        if allow is not None and not any(p.lower() in name.lower() for p in allow):
            continue
        try:
            out.append(DeferredTool(spec))
        except Exception as exc:
            # A malformed schema shouldn't take the whole request down, but a
            # silently dropped tool looks identical to "the model ignored it".
            logger.warning("skipping client tool %r: %s", name, exc)
    return out


def to_agent_messages(messages: List[dict]) -> List[dict]:
    """OpenAI wire format -> qwen-agent messages, preserving tool round-trips."""
    out: List[dict] = []
    call_names: dict = {}  # tool_call_id -> function name

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        content = content or ""

        if role == "assistant" and msg.get("tool_calls"):
            if content:
                out.append({"role": "assistant", "content": content})
            for call in msg["tool_calls"]:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                call_id = call.get("id") or "1"
                call_names[call_id] = name
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                out.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": name, "arguments": args},
                    "extra": {"function_id": call_id},
                })
        elif role == "tool":
            call_id = msg.get("tool_call_id") or "1"
            out.append({
                "role": "function",
                "name": msg.get("name") or call_names.get(call_id, "tool"),
                "content": content,
                "extra": {"function_id": call_id},
            })
        else:
            out.append({"role": role, "content": content})
    return out


def find_deferred(chunk: List[dict]) -> Optional[Tuple[int, int, dict]]:
    """Locate a stub tool's sentinel result in an agent snapshot.

    Returns (call_index, result_index, tool_call_payload). Reading the call from
    the snapshot that already contains its *result* guarantees the streamed
    arguments are complete.
    """
    for i, msg in enumerate(chunk):
        if msg.get("role") != "function" or msg.get("content") != DEFER_SENTINEL:
            continue
        call_id = (msg.get("extra") or {}).get("function_id")
        for j in range(i - 1, -1, -1):
            prev = chunk[j]
            fc = prev.get("function_call")
            if not fc:
                continue
            prev_id = (prev.get("extra") or {}).get("function_id")
            if call_id is None or prev_id == call_id or j == i - 1:
                return j, i, {
                    "id": prev_id or call_id or "call_1",
                    "type": "function",
                    "function": {"name": fc["name"], "arguments": fc["arguments"]},
                }
    return None
