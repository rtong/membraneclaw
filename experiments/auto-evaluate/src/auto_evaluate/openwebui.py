from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OpenWebUIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    content: str
    raw: dict[str, Any]
    latency_ms: int


class OpenWebUIClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise OpenWebUIError(f"OpenWebUI HTTP {exc.code}: {detail}") from exc
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise OpenWebUIError(f"OpenWebUI connection failed: {exc}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OpenWebUIError(f"OpenWebUI returned non-JSON: {body[:1000]}") from exc

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/api/models")

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        generation: dict[str, Any],
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": generation.get("temperature", 0.0),
            "top_p": generation.get("top_p", 1.0),
            "max_tokens": generation.get("max_tokens", 6000),
        }
        start = time.perf_counter()
        raw = self._request("POST", "/api/chat/completions", payload)
        latency_ms = round((time.perf_counter() - start) * 1000)
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenWebUIError(f"Unexpected chat response shape: {json.dumps(raw)[:2000]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise OpenWebUIError("OpenWebUI returned an empty assistant response")
        return ChatResult(content=content, raw=raw, latency_ms=latency_ms)

    def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        generation: dict[str, Any],
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": generation.get("temperature", 0.0),
            "top_p": generation.get("top_p", 1.0),
            "max_tokens": generation.get("max_tokens", 6000),
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat/completions",
            data=data,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        start = time.perf_counter()
        parts: list[str] = []
        event_count = 0
        finish_reason = None
        usage = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    item = line[5:].strip()
                    if item == "[DONE]":
                        break
                    try:
                        event = json.loads(item)
                    except json.JSONDecodeError as exc:
                        raise OpenWebUIError(
                            f"OpenWebUI returned invalid SSE data: {item[:1000]}"
                        ) from exc
                    event_count += 1
                    if isinstance(event, dict) and event.get("error"):
                        raise OpenWebUIError(
                            f"OpenWebUI stream error: {json.dumps(event['error'], ensure_ascii=False)[:2000]}"
                        )
                    if not isinstance(event, dict):
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        parts.append(content)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise OpenWebUIError(f"OpenWebUI HTTP {exc.code}: {detail}") from exc
        except OpenWebUIError:
            raise
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise OpenWebUIError(f"OpenWebUI streaming connection failed: {exc}") from exc

        latency_ms = round((time.perf_counter() - start) * 1000)
        content = "".join(parts).strip()
        if not content:
            raise OpenWebUIError(
                f"OpenWebUI stream ended without assistant content ({event_count} SSE events)"
            )
        if "[agent error]" in content.lower():
            raise OpenWebUIError(f"Upstream agent reported an error: {content[-2000:]}")
        return ChatResult(
            content=content,
            raw={
                "object": "chat.completion.stream",
                "model": model,
                "event_count": event_count,
                "finish_reason": finish_reason,
                "usage": usage,
            },
            latency_ms=latency_ms,
        )
