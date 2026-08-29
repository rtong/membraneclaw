from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .codex_automation import (
    _REQUEST_B64_PREFIX,
    _RESULT_B64_PREFIX,
    build_repair_prompt,
    parse_json_object,
    validate_task_output,
)
from .trajectory import extract_observable_trajectory


class CodexWorkerError(RuntimeError):
    """Worker failure with structured diagnostics safe to persist in run artifacts."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _thread_id(thread: Any, result: Any) -> str | None:
    value = getattr(thread, "id", None) or getattr(result, "thread_id", None)
    return str(value) if value is not None else None


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", by_alias=True)
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _normalized_mcp_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize direct MCP and nested ``codex_apps`` calls to one logical tool ID."""
    provider_server = str(payload.get("server") or "mcp")
    raw_tool = str(payload.get("tool") or "unknown")
    logical_server = provider_server.lower()
    tool_name = f"{provider_server}.{raw_tool}"
    if provider_server.lower() == "codex_apps" and "." in raw_tool:
        logical_server = raw_tool.split(".", 1)[0].lower()
        tool_name = raw_tool
    elif raw_tool.lower().startswith(provider_server.lower() + "."):
        tool_name = raw_tool

    error = payload.get("error")
    result = payload.get("result")
    raw_status = str(payload.get("status") or "").lower()
    successful = (
        raw_status in {"completed", "success", "succeeded"}
        and error is None
        and result is not None
    )
    return {
        "field": "mcp_tool_call",
        "event_type": "tool_interaction",
        "tool_name": tool_name,
        "arguments": payload.get("arguments") or {},
        "observation": result if successful else (error if error is not None else result),
        "status": "success" if successful else "error",
        "metadata": {
            "server": logical_server,
            "provider_server": provider_server,
            "tool": raw_tool,
            "duration_ms": payload.get("durationMs") or payload.get("duration_ms"),
            "plugin_id": payload.get("pluginId") or payload.get("plugin_id"),
            "raw_status": payload.get("status"),
        },
    }


def _codex_tool_events(result: Any) -> list[dict[str, Any]]:
    """Keep observable tool calls/results, never hidden reasoning items."""
    events: list[dict[str, Any]] = []
    for wrapped in getattr(result, "items", None) or []:
        item = getattr(wrapped, "root", wrapped)
        payload = _model_dump(item)
        item_type = payload.get("type") or getattr(item, "type", None)
        if item_type in {"mcpToolCall", "mcp_tool_call"}:
            events.append(_normalized_mcp_event(payload))
        elif item_type in {"dynamicToolCall", "dynamic_tool_call"}:
            namespace = str(payload.get("namespace") or "dynamic")
            tool = str(payload.get("tool") or "unknown")
            events.append(
                {
                    "field": "dynamic_tool_call",
                    "event_type": "tool_interaction",
                    "tool_name": f"{namespace}.{tool}",
                    "arguments": payload.get("arguments") or {},
                    "observation": payload.get("contentItems"),
                    "status": "success" if payload.get("success") is not False else "error",
                    "metadata": {"namespace": namespace, "tool": tool},
                }
            )
    return events


def _parse_codex_cli_jsonl(stdout: str) -> dict[str, Any]:
    """Extract the final response and completed observable calls from Codex JSONL."""
    thread_id: str | None = None
    final_response: str | None = None
    completed_tool_events: dict[str, dict[str, Any]] = {}
    malformed_lines: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(line[-500:])
            continue
        record_type = record.get("type")
        if record_type == "thread.started":
            value = record.get("thread_id")
            thread_id = str(value) if value else thread_id
            continue
        if record_type != "item.completed":
            continue
        item = record.get("item") or {}
        item_type = item.get("type")
        if item_type in {"mcpToolCall", "mcp_tool_call"}:
            event_id = str(item.get("id") or f"tool-{len(completed_tool_events) + 1}")
            completed_tool_events[event_id] = _normalized_mcp_event(item)
        elif item_type in {"agentMessage", "agent_message"}:
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                final_response = text
    return {
        "thread_id": thread_id,
        "final_response": final_response,
        "tool_events": list(completed_tool_events.values()),
        "malformed_lines": malformed_lines,
    }


def _codex_cli_path() -> str:
    configured = os.environ.get("CODEX_CLI_PATH", "").strip()
    if configured and Path(configured).exists():
        return configured
    discovered = shutil.which("codex")
    if discovered:
        return discovered
    try:
        from codex_cli_bin import bundled_codex_path

        bundled = bundled_codex_path()
        if bundled.exists():
            return str(bundled)
    except (ImportError, OSError):
        pass
    raise CodexWorkerError(
        "Codex CLI was not found via CODEX_CLI_PATH, PATH, or the bundled SDK runtime"
    )


def _best_effort_cli_archive(codex_cli: str, thread_id: str | None) -> None:
    if not thread_id:
        return
    try:
        subprocess.run(
            [codex_cli, "archive", thread_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except Exception:
        pass


def _run_cli_tool_teacher(request: dict[str, Any]) -> dict[str, Any]:
    """Run a required-tools Teacher through the headless auto-approval CLI path."""
    stage = request["stage"]
    task = request["task"]
    prompt = request["prompt"]
    retries = int(request.get("retries", 2))
    model = request["model"]
    timeout_seconds = int(request.get("timeout_seconds", 3600))
    codex_cli = _codex_cli_path()
    started = time.perf_counter()
    last_errors: list[str] = []
    last_diagnostics: dict[str, Any] = {}

    for attempt in range(1, retries + 2):
        completed = subprocess.run(
            [
                codex_cli,
                "exec",
                "--json",
                "--approve-for-me",
                "--skip-git-repo-check",
                "-m",
                model,
                "-",
            ],
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        parsed = _parse_codex_cli_jsonl(completed.stdout)
        thread_id = parsed["thread_id"]
        tool_events = parsed["tool_events"]
        last_diagnostics = {
            "executor": "codex_cli_auto_approve",
            "thread_id": thread_id,
            "attempt": attempt,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-4000:],
            "malformed_stdout_lines": parsed["malformed_lines"][-5:],
            "observable_tool_events": tool_events,
        }
        try:
            if completed.returncode != 0:
                raise ValueError(
                    f"Codex CLI exited with {completed.returncode}: "
                    + (
                        completed.stderr.strip()
                        or completed.stdout.strip()
                        or "no output"
                    )[-2000:]
                )
            response = parsed["final_response"]
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Codex CLI returned no completed agent_message")
            output = parse_json_object(response)
            answer_text = output.get("response_text")
            if not isinstance(answer_text, str):
                answer_text = response
            output["trajectory"] = extract_observable_trajectory(
                answer_text,
                raw_response={"trajectory_events": tool_events},
                tools_enabled=True,
                rag_enabled=None,
            )
            last_errors = validate_task_output(stage, task, output)
            if not last_errors:
                return {
                    "ok": True,
                    "output": output,
                    "thread_id": thread_id,
                    "attempts": attempt,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "diagnostics": last_diagnostics,
                }
        except ValueError as exc:
            last_errors = [str(exc)]
        finally:
            _best_effort_cli_archive(codex_cli, thread_id)
        if attempt <= retries:
            prompt = build_repair_prompt(stage, task, last_errors).replace(
                "in this same conversation", "in this fresh isolated retry"
            )

    last_diagnostics["last_errors"] = last_errors
    raise CodexWorkerError(
        "Codex output remained invalid after repair: " + "; ".join(last_errors),
        last_diagnostics,
    )


def _codex_session_tool_events(
    thread_id: str | None,
    *,
    seen_call_ids: set[str],
    sessions_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Read completed nested app calls from Codex's observable session audit log."""
    if not thread_id:
        return []
    root = sessions_root or (Path.home() / ".codex" / "sessions")
    if not root.exists():
        return []
    files = list(root.rglob(f"*{thread_id}.jsonl"))
    if not files:
        return []
    events: list[dict[str, Any]] = []
    for line in files[0].read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = record.get("payload") or {}
        if payload.get("type") != "mcp_tool_call_end":
            continue
        call_id = str(payload.get("call_id") or "")
        if call_id and call_id in seen_call_ids:
            continue
        invocation = payload.get("invocation") or {}
        tool = str(invocation.get("tool") or payload.get("action_name") or "unknown")
        app_name = str(payload.get("app_name") or tool.split(".", 1)[0] or "mcp")
        app_id = app_name.lower()
        result = payload.get("result")
        failed = isinstance(result, dict) and "Err" in result
        events.append(
            {
                "field": "codex_app_tool_call",
                "event_type": "tool_interaction",
                "tool_name": tool if "." in tool else f"{app_id}.{tool}",
                "arguments": invocation.get("arguments") or {},
                "observation": result,
                "status": "error" if failed else "success",
                "metadata": {
                    "server": app_id,
                    "provider_server": invocation.get("server"),
                    "app_name": app_name,
                    "action_name": payload.get("action_name"),
                    "call_id": call_id or None,
                    "duration": payload.get("duration"),
                },
            }
        )
        if call_id:
            seen_call_ids.add(call_id)
    return events

def _best_effort_archive(codex: Any, thread_id: str | None) -> None:
    """Archive a finished teacher/judge thread so it leaves the active UI list.

    Best-effort only: results are already persisted in the run directory, so a
    failure to archive must never fail the finished task.
    """
    if not thread_id:
        return
    try:
        codex.thread_archive(thread_id)
    except Exception:
        pass


def run(request: dict[str, Any]) -> dict[str, Any]:
    tool_policy = (request.get("task") or {}).get("tool_policy") or {}
    tool_teacher_executor = os.environ.get(
        "AE_CODEX_TOOL_TEACHER_EXECUTOR", "cli_auto_approve"
    ).strip().lower()
    if (
        request.get("stage") == "teacher"
        and tool_policy.get("require_observable_call")
        and tool_teacher_executor != "sdk"
    ):
        return _run_cli_tool_teacher(request)

    try:
        from openai_codex import Codex, Sandbox
    except ImportError as exc:
        raise RuntimeError(
            "The active Miniforge environment does not contain openai-codex. "
            "Run: python -m pip install -e ."
        ) from exc

    stage = request["stage"]
    task = request["task"]
    prompt = request["prompt"]
    retries = int(request.get("retries", 2))
    model = request["model"]
    started = time.perf_counter()
    last_errors: list[str] = []
    observable_tool_events: list[dict[str, Any]] = []
    seen_session_call_ids: set[str] = set()
    with Codex() as codex:
        sandbox = getattr(Sandbox, "read_only", Sandbox.workspace_write)
        ephemeral = os.environ.get("AE_CODEX_EPHEMERAL_THREADS", "1").strip().lower() not in {"0", "false", "no"}
        thread = codex.thread_start(model=model, sandbox=sandbox, ephemeral=ephemeral)
        thread_id = _thread_id(thread, None)
        result = None
        for attempt in range(1, retries + 2):
            result = thread.run(prompt)
            observable_tool_events.extend(_codex_tool_events(result))
            observable_tool_events.extend(
                _codex_session_tool_events(
                    _thread_id(thread, result),
                    seen_call_ids=seen_session_call_ids,
                )
            )
            if stage == "judge" and observable_tool_events:
                _best_effort_archive(codex, thread_id)
                raise RuntimeError(
                    "Judge isolation violation: observable tool calls are forbidden"
                )
            response = getattr(result, "final_response", None)
            if not isinstance(response, str) or not response.strip():
                last_errors = ["Codex returned an empty final_response"]
            else:
                try:
                    output = parse_json_object(response)
                    if stage == "teacher":
                        answer_text = output.get("response_text")
                        if not isinstance(answer_text, str):
                            answer_text = response
                        output["trajectory"] = extract_observable_trajectory(
                            answer_text,
                            raw_response={"trajectory_events": observable_tool_events},
                            tools_enabled=not bool((task.get("tool_policy") or {}).get("forbid_observable_calls")),
                            rag_enabled=None,
                        )
                    last_errors = validate_task_output(stage, task, output)
                except ValueError as exc:
                    last_errors = [str(exc)]
                if not last_errors:
                    _best_effort_archive(codex, thread_id)
                    return {
                        "ok": True,
                        "output": output,
                        "thread_id": _thread_id(thread, result),
                        "attempts": attempt,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                    }
            if attempt <= retries:
                prompt = build_repair_prompt(stage, task, last_errors)
        _best_effort_archive(codex, thread_id)
    raise CodexWorkerError(
        "Codex output remained invalid after repair: " + "; ".join(last_errors),
        {
            "executor": "openai_codex_sdk",
            "thread_id": thread_id,
            "attempt": retries + 1,
            "last_errors": last_errors,
            "observable_tool_events": observable_tool_events,
        },
    )


def main() -> int:
    try:
        request_payload = sys.stdin.read()
        if request_payload.startswith(_REQUEST_B64_PREFIX):
            encoded = request_payload[len(_REQUEST_B64_PREFIX) :]
            request_bytes = base64.b64decode(encoded, validate=True)
            request = json.loads(request_bytes.decode("utf-8"))
        else:
            request = json.loads(request_payload)
        result = run(request)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(diagnostics, dict) and diagnostics:
            result["diagnostics"] = diagnostics
    result_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
    encoded = base64.b64encode(result_bytes).decode("ascii")
    print(_RESULT_B64_PREFIX + encoded, flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
