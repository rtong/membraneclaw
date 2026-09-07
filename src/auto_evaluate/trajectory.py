from __future__ import annotations

import json
import re
from typing import Any


_TRANSCRIPT_TOKEN = re.compile(
    r"(?P<call>`?🔧\s+(?P<tool>[^\s(`]+)\((?P<arguments>.*?)\)`?)"
    r"|(?P<result>`?↳\s+(?P<observation>.*?)`\s*(?=\r?\n|$))",
    re.DOTALL,
)
_RETRIEVAL_HINTS = ("search", "retriev", "knowledge", "rag", "file", "spreadsheet")


def _parse_json_or_text(value: str) -> Any:
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def _preview(value: Any, limit: int = 1600) -> Any:
    if isinstance(value, dict) and set(value) == {"raw_text"}:
        raw = value["raw_text"]
        return {"raw_text": raw if len(raw) <= limit else raw[:limit] + "…[truncated]"}
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) <= limit:
        return value
    return {"preview": rendered[:limit] + "…[truncated]"}


def _event_type(tool_name: str) -> str:
    return (
        "retrieval_interaction"
        if any(hint in tool_name.lower() for hint in _RETRIEVAL_HINTS)
        else "tool_interaction"
    )


def _transcript_interactions(response_text: str) -> list[dict[str, Any]]:
    """Pair visible calls/results, including batched call-call/result-result transcripts."""
    pending: list[dict[str, str]] = []
    completed: list[dict[str, Any]] = []
    last_pending_key: tuple[str, str] | None = None

    for match in _TRANSCRIPT_TOKEN.finditer(response_text):
        if match.group("call") is not None:
            tool_name = match.group("tool")
            arguments_text = match.group("arguments").strip()
            key = (tool_name, arguments_text)
            # OpenWebUI can render the first call in a parallel batch twice:
            # once as the initiating call and once in the batch list. Collapse
            # only adjacent duplicates that have not received a result yet.
            if key == last_pending_key:
                continue
            pending.append({"tool_name": tool_name, "arguments_text": arguments_text})
            last_pending_key = key
            continue

        observation_text = match.group("observation")
        if pending:
            call = pending.pop(0)
            last_pending_key = (
                (pending[-1]["tool_name"], pending[-1]["arguments_text"])
                if pending
                else None
            )
            tool_name = call["tool_name"]
            arguments = _parse_json_or_text(call["arguments_text"])
        else:
            tool_name = "unmatched_tool_result"
            arguments = {}
            last_pending_key = None
        observation = _parse_json_or_text(observation_text)
        rendered_observation = json.dumps(observation, ensure_ascii=False).lower()
        status = (
            "error"
            if any(token in rendered_observation for token in ('"error"', "tool error", "exception"))
            else "success"
        )
        completed.append(
            {
                "event_type": _event_type(tool_name),
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": _preview(observation),
                "status": status,
                "evidence_source": "visible_response_transcript",
            }
        )

    for call in pending:
        completed.append(
            {
                "event_type": _event_type(call["tool_name"]),
                "tool_name": call["tool_name"],
                "arguments": _parse_json_or_text(call["arguments_text"]),
                "observation": {"missing": "no observable result"},
                "status": "missing_result",
                "evidence_source": "visible_response_transcript",
            }
        )

    for index, row in enumerate(completed, start=1):
        row["event_id"] = f"T{index:03d}"
        row["sequence"] = index
    return completed


def extract_observable_trajectory(
    response_text: str,
    *,
    raw_response: dict[str, Any] | None = None,
    tools_enabled: bool | None = None,
    rag_enabled: bool | None = None,
) -> dict[str, Any]:
    """Extract only observable actions/results; never expose hidden chain-of-thought."""
    interactions = _transcript_interactions(response_text)

    interactions.append(
        {
            "event_id": "FINAL_RESPONSE",
            "sequence": len(interactions) + 1,
            "event_type": "final_response",
            "status": "observed",
            "content_preview": response_text[-1000:],
            "evidence_source": "assistant_content",
        }
    )
    native_events = list((raw_response or {}).get("trajectory_events") or [])
    if raw_response and not native_events:
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            for field in ("tool_calls", "tool_results", "sources", "citations"):
                if isinstance(message, dict) and message.get(field) not in (None, [], {}):
                    native_events.append({"field": field, "payload": message[field]})
    structured_hint = bool(
        native_events
        or (
            raw_response
            and any(key in raw_response for key in ("tool_calls", "tool_results", "trajectory"))
        )
    )
    if len(interactions) == 1 and interactions[0].get("event_type") == "final_response" and native_events:
        final_event = interactions.pop()
        for index, native in enumerate(native_events, start=1):
            field = str(native.get("field", "api_event"))
            event_type = native.get("event_type") or (
                "retrieval_interaction"
                if field in {"sources", "citations", "retrieval_results"}
                else "tool_interaction"
            )
            interactions.append(
                {
                    "event_id": f"T{index:03d}",
                    "sequence": index,
                    "event_type": event_type,
                    "tool_name": native.get("tool_name") or field,
                    "arguments": native.get("arguments") or {},
                    "observation": _preview(
                        native.get("observation", native.get("payload"))
                    ),
                    "status": native.get("status") or "observed",
                    "evidence_source": "api_structured_event",
                    "metadata": native.get("metadata") or {},
                }
            )
        final_event["sequence"] = len(interactions) + 1
        interactions.append(final_event)
    if interactions[:-1]:
        source = "api_structured_and_transcript" if structured_hint else "visible_response_transcript"
        completeness = "observable_transcript"
    else:
        source = "final_response_only"
        completeness = "insufficient_for_tool_trajectory" if tools_enabled else "not_applicable_no_tools"
    return {
        "schema_version": "1.0",
        "source": source,
        "completeness": completeness,
        "tools_enabled": tools_enabled,
        "rag_enabled": rag_enabled,
        "events": interactions,
        "summary": {
            "tool_interactions": sum(row["event_type"] == "tool_interaction" for row in interactions),
            "retrieval_interactions": sum(row["event_type"] == "retrieval_interaction" for row in interactions),
            "tool_errors": sum(
                row.get("status") in {"error", "missing_result"} for row in interactions
            ),
            "native_event_count": len(native_events),
        },
    }
