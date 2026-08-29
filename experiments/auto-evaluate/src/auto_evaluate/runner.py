from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .benchmark import iter_benchmarks
from .io_utils import read_json, sha256_tree, stable_hash, utc_now, write_json
from .openwebui import OpenWebUIAgentError, OpenWebUIClient, OpenWebUIError
from .trajectory import extract_observable_trajectory


SCORE_POINTS_BEGIN = "[SCORE_POINTS_BEGIN]"
SCORE_POINTS_END = "[SCORE_POINTS_END]"
_REQUIRED_SCORE_POINT_KEYS = {
    "task_type",
    "decision_variables",
    "fixed_inputs",
    "tool_calls",
    "constraint_checks",
    "final_answer",
}

_RAG_ROUTE_ACTIONS = {"use_rag", "skip_rag"}
_RAG_ROUTE_REASON_CODES = {
    "MISSING_TOOL_CONTRACT",
    "MISSING_PARAMETER_MAPPING",
    "MISSING_DOMAIN_KNOWLEDGE",
    "FULLY_SPECIFIED_NUMERIC_TASK",
    "SIMULATION_EVIDENCE_DOMINATES",
}


def parse_rag_route(content: str) -> dict[str, Any]:
    """Parse and validate the compact decision emitted by the router model."""
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("RAG router did not return a JSON object")
    try:
        route = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"RAG router returned invalid JSON: {exc}") from exc
    if not isinstance(route, dict):
        raise ValueError("RAG router output must be a JSON object")
    action = route.get("action")
    if action not in _RAG_ROUTE_ACTIONS:
        raise ValueError(f"RAG router returned unsupported action: {action!r}")
    reason_code = route.get("reason_code")
    if reason_code not in _RAG_ROUTE_REASON_CODES:
        raise ValueError(f"RAG router returned unsupported reason_code: {reason_code!r}")
    confidence = route.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
    ):
        raise ValueError("RAG router confidence must be between 0 and 1")
    retrieval_need = route.get("retrieval_need")
    if action == "use_rag" and not isinstance(retrieval_need, str):
        raise ValueError("use_rag requires a targeted retrieval_need string")
    if action == "skip_rag" and retrieval_need is not None:
        raise ValueError("skip_rag requires retrieval_need=null")
    return {
        "action": action,
        "reason_code": reason_code,
        "confidence": float(confidence),
        "retrieval_need": retrieval_need.strip() if isinstance(retrieval_need, str) else None,
    }


def response_completion_error(
    content: str,
    raw_response: dict[str, Any] | None,
    system_prompt: str,
) -> str | None:
    """Return a stable error message when a response violates its completion contract."""
    raw = raw_response or {}
    finish_reason = raw.get("finish_reason")
    if finish_reason is None:
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
    if finish_reason == "length":
        return "Assistant response incomplete: finish_reason='length'"

    trailer_required = SCORE_POINTS_BEGIN in system_prompt and SCORE_POINTS_END in system_prompt
    if not trailer_required:
        return None
    start = content.find(SCORE_POINTS_BEGIN)
    end = content.find(SCORE_POINTS_END)
    if start < 0 or end < 0 or end <= start:
        return "Assistant response incomplete: required score-points trailer is missing"
    payload = content[start + len(SCORE_POINTS_BEGIN) : end].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return "Assistant response incomplete: score-points trailer is not valid JSON"
    if not isinstance(parsed, dict) or not _REQUIRED_SCORE_POINT_KEYS.issubset(parsed):
        return "Assistant response incomplete: score-points trailer is missing required keys"
    return None


def required_tool_call_error(
    trajectory: dict[str, Any],
    *,
    required: bool,
) -> str | None:
    """Reject a tool-required answer unless a successful call/result is observable."""
    if not required:
        return None
    for event in trajectory.get("events") or []:
        if (
            event.get("event_type") == "tool_interaction"
            and event.get("status") == "success"
        ):
            return None
    return (
        "Required observable tool call is missing: the tool-enabled solver returned "
        "an answer without a successful tool interaction and result"
    )


def load_systems(config_path: Path, selected_system_ids: list[str] | None = None) -> dict[str, Any]:
    config = read_json(config_path)
    systems = config.get("systems") or []
    ids = [item.get("id") for item in systems]
    if not systems or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("systems.json must define systems with unique non-empty IDs")
    if selected_system_ids is not None:
        missing = sorted(set(selected_system_ids) - set(ids))
        if missing:
            raise ValueError(f"Evaluation profile references unknown system IDs: {missing}")
        by_id = {item["id"]: item for item in systems}
        systems = [by_id[system_id] for system_id in selected_system_ids]
        config["systems"] = systems
    for system in systems:
        env_name = system.get("model_env")
        model_id = os.environ.get(env_name or "", "").strip()
        if not model_id:
            raise ValueError(f"Missing model ID environment variable: {env_name}")
        system["model_id"] = model_id
        rag_version_env = system.get("rag_version_env")
        if system.get("rag_enabled"):
            system["rag_version"] = os.environ.get(rag_version_env or "", "unversioned").strip() or "unversioned"
        require_tool = system.get("require_observable_tool_call", False)
        if not isinstance(require_tool, bool):
            raise ValueError(
                f"{system['id']}: require_observable_tool_call must be true or false"
            )
        if require_tool and not system.get("tools_enabled"):
            raise ValueError(
                f"{system['id']}: require_observable_tool_call requires tools_enabled=true"
            )
        skill_version = system.get("skill_version")
        if skill_version:
            skill_id, version = skill_version.split("@", 1)
            skill_dir = config_path.parents[1] / "skills" / skill_id / f"v{version}"
            if not skill_dir.exists():
                raise ValueError(f"Configured Skill directory does not exist: {skill_dir}")
            system["skill_artifact_sha256"] = sha256_tree(skill_dir)
        adaptive = system.get("adaptive_rag")
        if adaptive is not None:
            if not isinstance(adaptive, dict):
                raise ValueError(f"{system['id']}: adaptive_rag must be an object")
            for field in ("router_model_env", "no_rag_model_env", "rag_model_env"):
                env_name = adaptive.get(field)
                model_id = os.environ.get(env_name or "", "").strip()
                if not model_id:
                    raise ValueError(f"{system['id']}: missing model ID environment variable: {env_name}")
                adaptive[field.removesuffix("_env") + "_id"] = model_id
            router_skill_version = adaptive.get("router_skill_version")
            if not isinstance(router_skill_version, str) or "@" not in router_skill_version:
                raise ValueError(f"{system['id']}: adaptive_rag.router_skill_version is required")
            router_skill_id, router_version = router_skill_version.split("@", 1)
            router_dir = config_path.parents[1] / "skills" / router_skill_id / f"v{router_version}"
            router_artifact = router_dir / "SKILL.md"
            if not router_artifact.exists():
                raise ValueError(f"Configured Router Skill does not exist: {router_artifact}")
            adaptive["router_skill_sha256"] = sha256_tree(router_dir)
            adaptive["router_prompt"] = router_artifact.read_text(encoding="utf-8")
            fallback = adaptive.get("fallback_action", "skip_rag")
            if fallback not in _RAG_ROUTE_ACTIONS:
                raise ValueError(f"{system['id']}: invalid adaptive RAG fallback_action: {fallback}")

    system_ids = {system["id"] for system in systems}
    for system in systems:
        adaptive = system.get("adaptive_rag")
        if not adaptive:
            continue
        execution_mode = adaptive.get("execution_mode", "independent")
        if execution_mode not in {"independent", "policy_replay"}:
            raise ValueError(
                f"{system['id']}: adaptive_rag.execution_mode must be independent or policy_replay"
            )
        if execution_mode == "policy_replay":
            for field in ("no_rag_system_id", "rag_system_id"):
                arm_system_id = adaptive.get(field)
                if arm_system_id not in system_ids or arm_system_id == system["id"]:
                    raise ValueError(
                        f"{system['id']}: adaptive_rag.{field} must reference another selected system"
                    )

    recovery = config.get("context_recovery")
    if recovery is not None:
        if not isinstance(recovery, dict):
            raise ValueError("systems.json context_recovery must be an object")
        if recovery.get("enabled"):
            policy_version = recovery.get("policy_version")
            if not isinstance(policy_version, str) or not policy_version.strip():
                raise ValueError("context_recovery.policy_version is required")
            trigger_error_types = recovery.get("trigger_error_types")
            if not isinstance(trigger_error_types, list) or not all(
                isinstance(value, str) and value for value in trigger_error_types
            ):
                raise ValueError("context_recovery.trigger_error_types must be a string list")
            finalizer_model_env = recovery.get("finalizer_model_env")
            finalizer_model_id = os.environ.get(finalizer_model_env or "", "").strip()
            if not finalizer_model_id:
                raise ValueError(
                    "context_recovery: missing model ID environment variable: "
                    f"{finalizer_model_env}"
                )
            recovery["finalizer_model_id"] = finalizer_model_id
            max_chars = recovery.get("max_partial_response_chars", 12000)
            if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1000:
                raise ValueError(
                    "context_recovery.max_partial_response_chars must be an integer >= 1000"
                )
            max_prompt_chars = recovery.get("max_prompt_chars", 24000)
            if (
                not isinstance(max_prompt_chars, int)
                or isinstance(max_prompt_chars, bool)
                or max_prompt_chars < 4000
            ):
                raise ValueError(
                    "context_recovery.max_prompt_chars must be an integer >= 4000"
                )
            if not isinstance(recovery.get("system_prompt"), str) or not recovery[
                "system_prompt"
            ].strip():
                raise ValueError("context_recovery.system_prompt is required")
            if not isinstance(recovery.get("generation"), dict):
                raise ValueError("context_recovery.generation must be an object")
    return config


def _run_rag_router(
    *,
    client: OpenWebUIClient,
    benchmark: dict[str, Any],
    system: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    adaptive = system["adaptive_rag"]
    router_generation = {
        **generation,
        "stream": True,
        "max_tokens": 128,
        "temperature": 0.0,
        "enable_thinking": False,
        "max_retries": 0,
        **(adaptive.get("generation_overrides") or {}),
    }
    started = time.perf_counter()
    raw_content = ""
    try:
        chat_method = client.chat_stream if router_generation.get("stream") else client.chat
        result = chat_method(
            model=adaptive["router_model_id"],
            messages=[
                {"role": "system", "content": adaptive["router_prompt"]},
                {"role": "user", "content": benchmark["question_prompt"]},
            ],
            generation=router_generation,
        )
        raw_content = result.content
        route = parse_rag_route(result.content)
        return {
            **route,
            "status": "success",
            "router_model_id": adaptive["router_model_id"],
            "router_skill_version": adaptive["router_skill_version"],
            "latency_ms": result.latency_ms,
            "raw_response": result.raw,
        }
    except (OpenWebUIError, ValueError) as exc:
        latency_ms = round((time.perf_counter() - started) * 1000)
        return {
            "action": adaptive.get("fallback_action", "skip_rag"),
            "reason_code": "ROUTER_FALLBACK",
            "confidence": 0.0,
            "retrieval_need": None,
            "status": "fallback",
            "error": str(exc),
            "raw_content": raw_content,
            "router_model_id": adaptive["router_model_id"],
            "router_skill_version": adaptive["router_skill_version"],
            "latency_ms": latency_ms,
        }


def make_client(generation: dict[str, Any]) -> OpenWebUIClient:
    base_url = os.environ.get("OPENWEBUI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENWEBUI_API_KEY", "").strip()
    if not base_url:
        raise ValueError("OPENWEBUI_BASE_URL is missing; fill it in .env")
    if not api_key:
        raise ValueError("OPENWEBUI_API_KEY is missing; fill it in .env")
    return OpenWebUIClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=int(generation.get("timeout_seconds", 600)),
    )

def classify_execution_error(message: str) -> str:
    lowered = message.lower()
    if "maximum context length" in lowered or "context window" in lowered:
        return "context_window_exceeded"
    if "without assistant content" in lowered and "finish_reason='length'" in lowered:
        return "output_budget_exhausted"
    if "without assistant content" in lowered or "empty assistant response" in lowered:
        return "empty_assistant_response"
    if "assistant response incomplete" in lowered:
        return "incomplete_response"
    if "required observable tool call is missing" in lowered:
        return "required_tool_call_missing"
    if "connection failed" in lowered or "connection closed" in lowered:
        return "connection_failure"
    if "http 401" in lowered or "http 403" in lowered or "unauthorized" in lowered:
        return "authentication_failure"
    if "http 400" in lowered or "bad request" in lowered:
        return "invalid_request"
    return "upstream_execution_failure"


def retry_limit_for_error(error_type: str, configured_retries: int) -> int:
    if error_type in {
        "authentication_failure",
        "invalid_request",
        "context_window_exceeded",
        "output_budget_exhausted",
        "incomplete_response",
        "required_tool_call_missing",
    }:
        return 0
    return configured_retries


def retry_delay_for_error(
    error_type: str,
    attempt: int,
    generation: dict[str, Any] | None = None,
) -> float:
    settings = generation or {}
    if error_type == "connection_failure":
        base = max(0.0, float(settings.get("connection_retry_base_seconds", 10)))
        maximum = max(base, float(settings.get("connection_retry_max_seconds", 60)))
        return min(base * (2**attempt), maximum)
    return float(min(2**attempt, 8))


def _context_recovery_excerpt(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    marker = "\n\n[... middle of partial execution omitted for context safety ...]\n\n"
    if max_chars <= len(marker) + 2:
        return text[-max_chars:], True
    available_chars = max_chars - len(marker)
    head_chars = available_chars // 4
    tail_chars = available_chars - head_chars
    return text[:head_chars] + marker + text[-tail_chars:], True


def _context_recovery_eligible(
    *,
    recovery: dict[str, Any] | None,
    system: dict[str, Any],
    error_type: str,
    partial_response: str,
) -> bool:
    if not recovery or not recovery.get("enabled"):
        return False
    if error_type not in set(recovery.get("trigger_error_types") or []):
        return False
    if recovery.get("require_partial_response", True) and not partial_response.strip():
        return False
    return True


def _run_context_recovery(
    *,
    client: OpenWebUIClient,
    benchmark: dict[str, Any],
    recovery: dict[str, Any],
    partial_response: str,
    shared_system_prompt: str,
) -> dict[str, Any]:
    recovery_generation = dict(recovery["generation"])
    user_prefix = (
        "ORIGINAL QUESTION\n"
        "-----------------\n"
        f"{benchmark['question_prompt']}\n\n"
        "PARTIAL EXECUTION FROM THE SOLVER\n"
        "---------------------------------\n"
    )
    user_suffix = (
        "\n\nFINALIZATION TASK\n"
        "-----------------\n"
        "Return the final answer now. Use only observed evidence above, make no more "
        "tool calls, and explicitly identify any boundary or required value that remains "
        "unverified. Put the decision or conclusion first and do not narrate the solver's "
        "trial-and-error process."
    )
    max_prompt_chars = int(recovery.get("max_prompt_chars", 24000))
    fixed_prompt_chars = len(recovery["system_prompt"]) + len(user_prefix) + len(user_suffix)
    excerpt_budget = min(
        int(recovery.get("max_partial_response_chars", 12000)),
        max(0, max_prompt_chars - fixed_prompt_chars),
    )
    if excerpt_budget <= 0:
        raise OpenWebUIError(
            "Context recovery fixed prompt exceeds configured max_prompt_chars"
        )
    excerpt, truncated = _context_recovery_excerpt(partial_response, excerpt_budget)
    messages = [
        {"role": "system", "content": recovery["system_prompt"]},
        {
            "role": "user",
            "content": user_prefix + excerpt + user_suffix,
        },
    ]
    chat_method = client.chat_stream if recovery_generation.get("stream") else client.chat
    result = chat_method(
        model=recovery["finalizer_model_id"],
        messages=messages,
        generation=recovery_generation,
    )
    completion_error = response_completion_error(
        result.content,
        result.raw,
        shared_system_prompt,
    )
    if completion_error:
        raise OpenWebUIAgentError(
            completion_error,
            response_text=result.content,
            raw_response=result.raw,
            latency_ms=result.latency_ms,
        )
    return {
        "content": result.content,
        "raw_response": result.raw,
        "latency_ms": result.latency_ms,
        "generation": recovery_generation,
        "input_partial_response_chars": len(partial_response),
        "max_prompt_chars": max_prompt_chars,
        "prompt_chars": sum(len(message["content"]) for message in messages),
        "included_partial_response_chars": len(excerpt),
        "partial_response_truncated": truncated,
    }


def _record_path(run_dir: Path, case_id: str, system_id: str) -> Path:
    return run_dir / "responses" / f"{case_id}__{system_id}.json"


def _execute_work_item(
    *,
    client: OpenWebUIClient,
    run_dir: Path,
    shared_system_prompt: str,
    item: dict[str, Any],
) -> str:
    benchmark = item["benchmark"]
    system = item["system"]
    system_generation = item["generation"]
    context_recovery = item.get("context_recovery")
    messages = item["messages"]
    path = item["path"]
    label = item["label"]
    print(f"{label}: running", flush=True)
    routing = None
    solver_model_id = system["model_id"]
    actual_rag_enabled = bool(system.get("rag_enabled", False))
    routing_latency_ms = 0
    if system.get("adaptive_rag"):
        routing = _run_rag_router(
            client=client,
            benchmark=benchmark,
            system=system,
            generation=system_generation,
        )
        routing_latency_ms = int(routing.get("latency_ms", 0) or 0)
        adaptive = system["adaptive_rag"]
        if routing["action"] == "use_rag":
            solver_model_id = adaptive["rag_model_id"]
            actual_rag_enabled = True
        else:
            solver_model_id = adaptive["no_rag_model_id"]
            actual_rag_enabled = False
        print(
            f"{label}: route={routing['action']} "
            f"({routing['reason_code']}, confidence={routing['confidence']:.2f})",
            flush=True,
        )
    base_record = {
        "schema_version": "1.1",
        "run_id": run_dir.name,
        "case_id": benchmark["case_id"],
        "system_id": system["id"],
        "display_name": system["display_name"],
        "model_id": solver_model_id,
        "configured_model_id": system["model_id"],
        "tools_enabled": system.get("tools_enabled", False),
        "skill_version": system.get("skill_version"),
        "rag_enabled": actual_rag_enabled,
        "rag_available": bool(system.get("rag_enabled", False)),
        "rag_policy": system.get("rag_policy"),
        "routing": routing,
        "benchmark_view": benchmark.get("benchmark_view"),
        "request_hash": item["request_hash"],
        "started_at": utc_now(),
    }
    last_error = None
    partial_response = ""
    partial_raw_response: dict[str, Any] | None = None
    partial_latency_ms: int | None = None
    max_retries = int(system_generation.get("max_retries", 2))
    attempts_used = 0
    for attempt in range(max_retries + 1):
        attempts_used = attempt + 1
        try:
            chat_method = client.chat_stream if system_generation.get("stream") else client.chat
            result = chat_method(
                model=solver_model_id, messages=messages, generation=system_generation
            )
            trajectory = extract_observable_trajectory(
                result.content,
                raw_response=result.raw,
                tools_enabled=system.get("tools_enabled", False),
                rag_enabled=actual_rag_enabled,
            )
            completion_error = response_completion_error(
                result.content, result.raw, shared_system_prompt
            )
            completion_error = completion_error or required_tool_call_error(
                trajectory,
                required=bool(system.get("require_observable_tool_call", False)),
            )
            if completion_error:
                raise OpenWebUIAgentError(
                    completion_error,
                    response_text=result.content,
                    raw_response=result.raw,
                    latency_ms=result.latency_ms,
                )
            write_json(
                path,
                {
                    **base_record,
                    "status": "success",
                    "native_status": "success",
                    "completion_mode": "native",
                    "attempts": attempt + 1,
                    "completed_at": utc_now(),
                    "latency_ms": routing_latency_ms + result.latency_ms,
                    "solver_latency_ms": result.latency_ms,
                    "response_text": result.content,
                    "usage": result.raw.get("usage"),
                    "generation": system_generation,
                    "raw_response": result.raw,
                    "trajectory": trajectory,
                },
            )
            print(
                f"{label}: success in {result.latency_ms / 1000:.1f}s "
                f"(attempt {attempt + 1})",
                flush=True,
            )
            return "success"
        except OpenWebUIError as exc:
            last_error = str(exc)
            if isinstance(exc, OpenWebUIAgentError):
                partial_response = exc.response_text
                partial_raw_response = exc.raw_response
                partial_latency_ms = exc.latency_ms
            error_type = classify_execution_error(last_error)
            retry_limit = retry_limit_for_error(error_type, max_retries)
            if attempt < retry_limit:
                delay = retry_delay_for_error(error_type, attempt, system_generation)
                print(
                    f"{label}: attempt {attempt + 1} failed [{error_type}]; "
                    f"retrying in {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
            else:
                break
    error_type = classify_execution_error(last_error or "")
    native_trajectory = extract_observable_trajectory(
        partial_response,
        raw_response=partial_raw_response,
        tools_enabled=system.get("tools_enabled", False),
        rag_enabled=actual_rag_enabled,
    )
    recovery_record = None
    if _context_recovery_eligible(
        recovery=context_recovery,
        system=system,
        error_type=error_type,
        partial_response=partial_response,
    ):
        print(
            f"{label}: native solver failed [{error_type}]; "
            "running one tool-free context-reset finalizer",
            flush=True,
        )
        recovery_started_at = utc_now()
        try:
            recovered = _run_context_recovery(
                client=client,
                benchmark=benchmark,
                recovery=context_recovery,
                partial_response=partial_response,
                shared_system_prompt=shared_system_prompt,
            )
            recovery_record = {
                "status": "success",
                "policy_version": context_recovery["policy_version"],
                "started_at": recovery_started_at,
                "completed_at": utc_now(),
                "finalizer_model_id": context_recovery["finalizer_model_id"],
                "tools_enabled": False,
                "rag_enabled": False,
                "latency_ms": recovered["latency_ms"],
                "generation": recovered["generation"],
                "max_prompt_chars": recovered["max_prompt_chars"],
                "prompt_chars": recovered["prompt_chars"],
                "input_partial_response_chars": recovered[
                    "input_partial_response_chars"
                ],
                "included_partial_response_chars": recovered[
                    "included_partial_response_chars"
                ],
                "partial_response_truncated": recovered[
                    "partial_response_truncated"
                ],
            }
            total_latency_ms = routing_latency_ms + int(partial_latency_ms or 0) + int(
                recovered["latency_ms"]
            )
            write_json(
                path,
                {
                    **base_record,
                    "status": "success",
                    "native_status": "error",
                    "native_error": last_error,
                    "native_error_type": error_type,
                    "completion_mode": "context_reset_finalizer",
                    "attempts": attempts_used,
                    "completed_at": utc_now(),
                    "latency_ms": total_latency_ms,
                    "solver_latency_ms": partial_latency_ms,
                    "recovery_latency_ms": recovered["latency_ms"],
                    "response_text": recovered["content"],
                    "partial_response_text": partial_response,
                    "partial_response_available": True,
                    "usage": recovered["raw_response"].get("usage"),
                    "native_usage": (partial_raw_response or {}).get("usage"),
                    "generation": system_generation,
                    "raw_response": recovered["raw_response"],
                    "native_raw_response": partial_raw_response,
                    "trajectory": native_trajectory,
                    "recovery": recovery_record,
                },
            )
            print(
                f"{label}: recovered in {recovered['latency_ms'] / 1000:.1f}s "
                f"after native [{error_type}]",
                flush=True,
            )
            return "success"
        except OpenWebUIError as recovery_exc:
            recovery_record = {
                "status": "error",
                "policy_version": context_recovery["policy_version"],
                "started_at": recovery_started_at,
                "completed_at": utc_now(),
                "finalizer_model_id": context_recovery["finalizer_model_id"],
                "tools_enabled": False,
                "rag_enabled": False,
                "error": str(recovery_exc),
                "error_type": classify_execution_error(str(recovery_exc)),
            }
            print(
                f"{label}: context-reset finalizer failed "
                f"[{recovery_record['error_type']}]",
                flush=True,
            )
    write_json(
        path,
        {
            **base_record,
            "status": "error",
            "native_status": "error",
            "completion_mode": "failed",
            "attempts": attempts_used,
            "completed_at": utc_now(),
            "error": last_error,
            "error_type": error_type,
            "response_text": partial_response,
            "latency_ms": (
                routing_latency_ms + partial_latency_ms
                if partial_latency_ms is not None
                else routing_latency_ms or None
            ),
            "solver_latency_ms": partial_latency_ms,
            "usage": (partial_raw_response or {}).get("usage"),
            "generation": system_generation,
            "raw_response": partial_raw_response,
            "trajectory": native_trajectory,
            "partial_response_available": bool(partial_response),
            "recovery": recovery_record,
        },
    )
    print(f"{label}: error after {attempts_used} attempts [{error_type}]", flush=True)
    return "error"


def _execute_policy_replay_item(
    *,
    client: OpenWebUIClient,
    run_dir: Path,
    item: dict[str, Any],
) -> str:
    benchmark = item["benchmark"]
    system = item["system"]
    adaptive = system["adaptive_rag"]
    path = item["path"]
    label = item["label"]
    print(f"{label}: running router for local policy replay", flush=True)
    routing = _run_rag_router(
        client=client,
        benchmark=benchmark,
        system=system,
        generation=item["generation"],
    )
    selected_system_id = (
        adaptive["rag_system_id"]
        if routing["action"] == "use_rag"
        else adaptive["no_rag_system_id"]
    )
    source_path = _record_path(run_dir, benchmark["case_id"], selected_system_id)
    source = read_json(source_path) if source_path.exists() else None
    base_record = {
        "schema_version": "1.1",
        "run_id": run_dir.name,
        "case_id": benchmark["case_id"],
        "system_id": system["id"],
        "display_name": system["display_name"],
        "configured_model_id": system["model_id"],
        "tools_enabled": True,
        "rag_available": True,
        "rag_policy": system.get("rag_policy"),
        "routing": routing,
        "benchmark_view": benchmark.get("benchmark_view"),
        "request_hash": item["request_hash"],
        "started_at": utc_now(),
        "execution_mode": "policy_replay",
        "selected_arm_system_id": selected_system_id,
    }
    if not source or source.get("status") != "success":
        error = (
            f"Selected physical arm {selected_system_id!r} is missing or incomplete; "
            "policy replay cannot produce an answer"
        )
        write_json(
            path,
            {
                **base_record,
                "status": "error",
                "native_status": "not_executed",
                "completion_mode": "failed",
                "attempts": 1,
                "completed_at": utc_now(),
                "error": error,
                "error_type": "upstream_execution_failure",
                "response_text": "",
                "latency_ms": routing.get("latency_ms"),
                "routing_latency_ms": routing.get("latency_ms"),
                "trajectory": None,
            },
        )
        print(f"{label}: error [selected_arm_unavailable: {selected_system_id}]", flush=True)
        return "error"

    routing_latency_ms = int(routing.get("latency_ms", 0) or 0)
    source_latency_ms = int(source.get("latency_ms", 0) or 0)
    write_json(
        path,
        {
            **base_record,
            "status": "success",
            "native_status": "not_executed",
            "completion_mode": "policy_replay",
            "attempts": 1,
            "completed_at": utc_now(),
            "model_id": source.get("model_id"),
            "tools_enabled": source.get("tools_enabled", True),
            "rag_enabled": source.get("rag_enabled", routing["action"] == "use_rag"),
            "latency_ms": routing_latency_ms + source_latency_ms,
            "routing_latency_ms": routing_latency_ms,
            "solver_latency_ms": source.get("solver_latency_ms"),
            "selected_arm_latency_ms": source_latency_ms,
            "selected_arm_request_hash": source.get("request_hash"),
            "selected_arm_completion_mode": source.get("completion_mode"),
            "selected_arm_native_status": source.get("native_status"),
            "selected_arm_native_error_type": source.get("native_error_type"),
            "selected_arm_recovery": source.get("recovery"),
            "response_text": source.get("response_text") or "",
            "usage": source.get("usage"),
            "generation": source.get("generation"),
            "raw_response": source.get("raw_response"),
            "trajectory": source.get("trajectory"),
        },
    )
    print(
        f"{label}: route={routing['action']} ({routing['reason_code']}, "
        f"confidence={routing['confidence']:.2f}); replayed {selected_system_id} locally",
        flush=True,
    )
    return "success"


def summarize_run_completeness(
    *,
    benchmarks_dir: Path,
    run_dir: Path,
    system_ids: list[str],
) -> dict[str, Any]:
    cases = [benchmark["case_id"] for benchmark in iter_benchmarks(benchmarks_dir)]
    incomplete: list[dict[str, Any]] = []
    success = 0
    native_success = 0
    recovered_success = 0
    policy_replay_success = 0
    for case_id in cases:
        for system_id in system_ids:
            path = _record_path(run_dir, case_id, system_id)
            if not path.exists():
                incomplete.append(
                    {"case_id": case_id, "system_id": system_id, "status": "missing"}
                )
                continue
            record = read_json(path)
            if record.get("status") == "success":
                success += 1
                if record.get("completion_mode") == "context_reset_finalizer":
                    recovered_success += 1
                elif record.get("completion_mode") == "policy_replay":
                    policy_replay_success += 1
                elif record.get("completion_mode") == "native":
                    native_success += 1
            else:
                incomplete.append(
                    {
                        "case_id": case_id,
                        "system_id": system_id,
                        "status": record.get("status", "unknown"),
                        "error_type": record.get("error_type"),
                    }
                )
    expected = len(cases) * len(system_ids)
    return {
        "expected": expected,
        "success": success,
        "native_success": native_success,
        "recovered_success": recovered_success,
        "policy_replay_success": policy_replay_success,
        "incomplete": len(incomplete),
        "items": incomplete,
    }


def execute_run(
    *,
    benchmarks_dir: Path,
    systems_path: Path,
    run_dir: Path,
    force: bool = False,
    selected_system_ids: list[str] | None = None,
    evaluation_profile: str | None = None,
    system_concurrency: int = 2,
) -> dict[str, int]:
    if system_concurrency < 1 or system_concurrency > 8:
        raise ValueError("system concurrency must be between 1 and 8")
    config = load_systems(systems_path, selected_system_ids)
    generation = config["generation"]
    context_recovery = config.get("context_recovery")
    client = make_client(generation)
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = list(iter_benchmarks(benchmarks_dir))
    policy_replay_system_ids = [
        system["id"]
        for system in config["systems"]
        if (system.get("adaptive_rag") or {}).get("execution_mode") == "policy_replay"
    ]
    physical_system_ids = [
        system["id"]
        for system in config["systems"]
        if system["id"] not in policy_replay_system_ids
    ]

    manifest = {
        "schema_version": "1.3",
        "run_id": run_dir.name,
        "created_at": utc_now(),
        "benchmark_cases": [item["case_id"] for item in benchmarks],
        "systems": config["systems"],
        "generation": generation,
        "context_recovery": context_recovery,
        "system_concurrency": system_concurrency,
        "execution_plan": {
            "physical_system_ids": physical_system_ids,
            "policy_replay_system_ids": policy_replay_system_ids,
            "expected_physical_solver_requests": len(benchmarks)
            * len(physical_system_ids),
            "expected_router_requests": len(benchmarks)
            * len(policy_replay_system_ids),
        },
        "shared_system_prompt": config["shared_system_prompt"],
        "evaluation_profile": evaluation_profile,
        "openwebui_base_url": os.environ.get("OPENWEBUI_BASE_URL", "").rstrip("/"),
    }
    write_json(run_dir / "manifest.json", manifest)

    counts = {"success": 0, "error": 0, "skipped": 0}
    total = len(benchmarks) * len(config["systems"])
    all_work_items: list[dict[str, Any]] = []
    item_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    position = 0
    for benchmark in benchmarks:
        for system in config["systems"]:
            position += 1
            label = f"[{position}/{total}] {benchmark['case_id']} / {system['id']}"
            system_generation = {**generation, **(system.get("generation_overrides") or {})}
            messages = [
                {"role": "system", "content": config["shared_system_prompt"]},
                {"role": "user", "content": benchmark["question_prompt"]},
            ]
            request_fingerprint = stable_hash(
                {
                    "case_sha256": benchmark["source"]["sha256"],
                    "system": system,
                    "generation": system_generation,
                    "messages": messages,
                    "context_recovery": context_recovery,
                }
            )
            item = {
                "benchmark": benchmark,
                "system": system,
                "generation": system_generation,
                "messages": messages,
                "context_recovery": context_recovery,
                "path": _record_path(run_dir, benchmark["case_id"], system["id"]),
                "label": label,
                "request_hash": request_fingerprint,
            }
            all_work_items.append(item)
            item_by_key[(benchmark["case_id"], system["id"])] = item

    for item in all_work_items:
        adaptive = item["system"].get("adaptive_rag") or {}
        if adaptive.get("execution_mode") != "policy_replay":
            continue
        case_id = item["benchmark"]["case_id"]
        arm_hashes = {
            arm_id: item_by_key[(case_id, arm_id)]["request_hash"]
            for arm_id in (
                adaptive["no_rag_system_id"],
                adaptive["rag_system_id"],
            )
        }
        item["request_hash"] = stable_hash(
            {
                "router_request_hash": item["request_hash"],
                "physical_arm_request_hashes": arm_hashes,
            }
        )

    work_items: list[dict[str, Any]] = []
    for item in all_work_items:
        path = item["path"]
        if path.exists() and not force:
            previous = read_json(path)
            if (
                previous.get("status") == "success"
                and previous.get("request_hash") == item["request_hash"]
            ):
                counts["skipped"] += 1
                print(f"{item['label']}: skipped (matching successful response)", flush=True)
                continue
        work_items.append(item)

    def complete(item: dict[str, Any]) -> str:
        return _execute_work_item(
            client=client,
            run_dir=run_dir,
            shared_system_prompt=config["shared_system_prompt"],
            item=item,
        )

    def complete_replay(item: dict[str, Any]) -> str:
        return _execute_policy_replay_item(client=client, run_dir=run_dir, item=item)

    def is_policy_replay(item: dict[str, Any]) -> bool:
        return (
            (item["system"].get("adaptive_rag") or {}).get("execution_mode")
            == "policy_replay"
        )

    physical_items = [item for item in work_items if not is_policy_replay(item)]
    replay_items = [item for item in work_items if is_policy_replay(item)]

    def run_phase(items: list[dict[str, Any]], worker) -> None:
        if system_concurrency == 1:
            for item in items:
                counts[worker(item)] += 1
        elif items:
            with ThreadPoolExecutor(max_workers=system_concurrency) as executor:
                futures = {executor.submit(worker, item): item["label"] for item in items}
                for future in as_completed(futures):
                    counts[future.result()] += 1

    # Physical arms must finish before a virtual policy record can reuse the
    # selected answer. Router requests within the replay phase remain parallel.
    run_phase(physical_items, complete)
    run_phase(replay_items, complete_replay)
    return counts
