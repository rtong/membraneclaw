"""OpenAI-compatible front end for the Qwen-Agent harness.

BetterGPT (or any OpenAI client) points its base URL here instead of at vLLM, so
chats run through the agent loop — custom tools, RAG, and the compat shims —
rather than hitting the raw model.
"""
import json
import logging
import os
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import bridge, config
from agent.core import build_agent

AGENT_MODEL_ID = os.environ.get("AGENT_MODEL_ID", "qwen3.5-9b-agent")
# Optional shared secret. Unset means no auth, which is fine while bound to loopback.
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")
# Comma-separated docs attached to every request, so RAG works from a chat UI
# that has no way to upload files to us.
AGENT_FILES = [p for p in os.environ.get("AGENT_FILES", "").split(",") if p.strip()]
# Which client-declared tools to bridge. Empty/"*" means all. Open WebUI ships ~26
# built-in tools (knowledge, notes, tasks, calendar) alongside your own; a 9B model
# stops calling any of them somewhere past ~18 declared tools, so narrowing the list
# is often what makes a custom tool work at all.
_allow = os.environ.get("AGENT_CLIENT_TOOLS", "").strip()
CLIENT_TOOL_ALLOW = (
    None if _allow in ("", "*") else {t.strip() for t in _allow.split(",") if t.strip()}
)
# Declared-tool count past which tool selection gets unreliable on this model.
TOOL_COUNT_WARN = int(os.environ.get("AGENT_TOOL_COUNT_WARN", "18"))

logger = logging.getLogger("membraneclaw")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - membraneclaw - %(message)s"))
    logger.addHandler(_handler)

logger.info(
    "config: client_tool_allow=%s files=%s model=%s",
    CLIENT_TOOL_ALLOW, AGENT_FILES, AGENT_MODEL_ID,
)

app = FastAPI(title="MembraneClaw Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def allow_private_network(request: Request, call_next):
    # Chrome's Private Network Access preflight needs this to let an HTTPS page
    # (bettergpt.chat) reach a loopback server. This middleware wraps CORSMiddleware,
    # so preflights must still fall through to it for the Allow-Origin headers.
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


def _check_auth(request: Request) -> None:
    if not AGENT_API_KEY:
        return
    header = request.headers.get("authorization", "")
    if header.removeprefix("Bearer ").strip() != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _render(chunk: List[dict], skip: frozenset = frozenset()) -> str:
    """Flatten an agent snapshot to display text, holding back unsettled messages.

    Tool arguments stream in token by token, so a tool line is only rendered once
    a later message proves it is complete. That keeps the output monotonic, which
    the prefix-diff in _stream relies on.
    """
    parts = []
    for i, msg in enumerate(chunk):
        if i in skip:
            continue
        settled = i < len(chunk) - 1
        role = msg.get("role")
        if role == "assistant" and msg.get("function_call"):
            if settled:
                fc = msg["function_call"]
                parts.append(f"\n`🔧 {fc['name']}({fc['arguments']})`\n")
        elif role == "function":
            if settled:
                result = str(msg.get("content", ""))
                if len(result) > 500:
                    result = result[:500] + " …"
                parts.append(f"`↳ {result}`\n\n")
        elif role == "assistant" and msg.get("content"):
            parts.append(msg["content"])
    return "".join(parts)


_agent_cache: dict = {}


def _get_agent(tools: Optional[List[dict]] = None):
    # The Assistant keeps no per-conversation state (history is passed to run()),
    # so it is safe to share — and reusing it avoids re-indexing AGENT_FILES per
    # request. Keyed by the client's tool set, since that changes the tool list.
    bridged = bridge.client_tools(tools, CLIENT_TOOL_ALLOW)
    key = json.dumps(sorted(t.name for t in bridged))
    if key not in _agent_cache:
        _agent_cache[key] = build_agent(files=AGENT_FILES or None, extra_tools=bridged)
    return _agent_cache[key]


def _run_agent(messages: List[dict], tools: Optional[List[dict]] = None):
    """Yield (text, tool_call) pairs. A non-None tool_call ends the turn and is
    handed to the client to execute."""
    agent = _get_agent(tools)
    for chunk in agent.run(messages=bridge.to_agent_messages(messages)):
        deferred = bridge.find_deferred(chunk)
        if deferred:
            call_idx, result_idx, payload = deferred
            # Abandoning the generator here stops the agent loop; the client owns
            # this call now.
            yield _render(chunk, skip=frozenset({call_idx, result_idx})), payload
            return
        yield _render(chunk), None


def _chunk_payload(rid: str, model: str, delta: dict, finish: Optional[str] = None) -> str:
    return "data: " + json.dumps({
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


def _stream(messages: List[dict], model: str, tools: Optional[List[dict]] = None):
    rid = "chatcmpl-" + uuid.uuid4().hex[:24]
    yield _chunk_payload(rid, model, {"role": "assistant", "content": ""})
    sent = ""
    try:
        for text, tool_call in _run_agent(messages, tools):
            if tool_call is not None:
                if text.startswith(sent) and text != sent:
                    yield _chunk_payload(rid, model, {"content": text[len(sent):]})
                yield _chunk_payload(rid, model, {"tool_calls": [{
                    "index": 0,
                    "id": tool_call["id"],
                    "type": "function",
                    "function": tool_call["function"],
                }]})
                yield _chunk_payload(rid, model, {}, finish="tool_calls")
                yield "data: [DONE]\n\n"
                return
            if text == sent:
                continue
            if text.startswith(sent):
                delta, sent = text[len(sent):], text
            else:
                # Shouldn't happen, but never replay a shrinking buffer.
                common = 0
                for a, b in zip(sent, text):
                    if a != b:
                        break
                    common += 1
                delta, sent = text[common:], text
            if delta:
                yield _chunk_payload(rid, model, {"content": delta})
    except Exception as exc:  # surface failures in the chat instead of a dead stream
        yield _chunk_payload(rid, model, {"content": f"\n\n**[agent error]** {exc}"})
    yield _chunk_payload(rid, model, {}, finish="stop")
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": AGENT_MODEL_ID, "object": "model", "owned_by": "membraneclaw"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": config.BASE_URL, "model": config.MODEL}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    model = body.get("model") or AGENT_MODEL_ID

    tools = body.get("tools")
    bridged = [t.name for t in bridge.client_tools(tools, CLIENT_TOOL_ALLOW)]
    declared = len(tools or [])
    logger.info(
        "request: roles=%s stream=%s client_tools=%d bridged=%s declared=%s",
        [m.get("role") for m in messages], bool(body.get("stream")), declared, bridged,
        [(t.get("function") or {}).get("name") for t in tools or []],
    )
    if tools and not bridged:
        logger.warning("no tools bridged; first spec = %s", json.dumps(tools[0])[:300])
    if len(bridged) > TOOL_COUNT_WARN:
        logger.warning(
            "%d tools declared; this model tends to stop calling tools past ~%d. "
            "Narrow them with AGENT_CLIENT_TOOLS.", len(bridged), TOOL_COUNT_WARN,
        )

    if body.get("stream"):
        return StreamingResponse(
            _stream(messages, model, tools),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, tool_call = "", None
    for text, tool_call in _run_agent(messages, tools):
        if tool_call is not None:
            break

    message = {"role": "assistant", "content": text}
    if tool_call is not None:
        message["tool_calls"] = [tool_call]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_call else "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("AGENT_HOST", "127.0.0.1"),
        port=int(os.environ.get("AGENT_PORT", "8001")),
    )
