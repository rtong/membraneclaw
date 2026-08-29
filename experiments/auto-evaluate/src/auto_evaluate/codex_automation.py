from __future__ import annotations

import base64
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .io_utils import read_json, stable_hash, utc_now, write_json, write_jsonl
from .taxonomy import FAILURE_CODES


class CodexAutomationError(RuntimeError):
    """Raised when a Codex teacher or judge task cannot be completed safely."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


_REQUEST_B64_PREFIX = "__AE_CODEX_REQUEST_B64__"
_RESULT_PREFIX = "__AE_CODEX_RESULT__"
_RESULT_B64_PREFIX = "__AE_CODEX_RESULT_B64__"
_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "CODEX_API_KEY")


def require_chatgpt_auth_environment() -> None:
    configured = [name for name in _API_KEY_ENV_VARS if os.environ.get(name)]
    if configured:
        names = ", ".join(configured)
        raise CodexAutomationError(
            f"Refusing to run with Platform API credentials present ({names}). "
            "Remove them from this shell and authenticate Codex with the ChatGPT account login instead."
        )


def validate_stage_environment(stage: str, tasks: list[dict[str, Any]]) -> None:
    require_chatgpt_auth_environment()


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        value = fence.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(value):
            if character != "{":
                continue
            try:
                parsed, end = decoder.raw_decode(value[index:])
            except json.JSONDecodeError:
                continue
            if value[index + end :].strip().strip("`"):
                continue
            break
        else:
            raise ValueError(f"response is not one valid JSON object: {first_error}") from first_error
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    return parsed


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _tool_event_matches_server(event: dict[str, Any], server: str) -> bool:
    server_id = server.lower()
    metadata = event.get("metadata") or {}
    return (
        str(metadata.get("server") or "").lower() == server_id
        or str(event.get("tool_name") or "").lower().startswith(server_id + ".")
    )


def _has_nonempty_observation(event: dict[str, Any]) -> bool:
    observation = event.get("observation")
    if observation is None:
        return False
    if isinstance(observation, str):
        return bool(observation.strip())
    if isinstance(observation, (dict, list, tuple, set)):
        return bool(observation)
    return True


def validate_teacher_output(task: dict[str, Any], output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "task_id": task["task_id"],
        "case_id": task["case_id"],
        "system_id": task.get("expected_output", {}).get("system_id") or task.get("model") or "gpt-5.6-teacher",
    }
    for field, value in expected.items():
        if output.get(field) != value:
            errors.append(f"{field} must equal {value!r}")
    if not isinstance(output.get("response_text"), str) or not output["response_text"].strip():
        errors.append("response_text must be a non-empty string")
    tool_policy = task.get("tool_policy") or {}
    trajectory = output.get("trajectory") or {}
    tool_events = [
        event
        for event in trajectory.get("events", [])
        if isinstance(event, dict) and event.get("event_type") == "tool_interaction"
    ]
    if tool_policy.get("forbid_observable_calls") and tool_events:
        errors.append("tool-free teacher must not make observable tool calls")
    if tool_policy.get("require_observable_call"):
        server = str(tool_policy.get("mcp_server") or "configured MCP")
        matching_calls = [
            event
            for event in tool_events
            if isinstance(event, dict)
            and _tool_event_matches_server(event, server)
        ]
        unexpected_calls = [
            event
            for event in tool_events
            if isinstance(event, dict) and not _tool_event_matches_server(event, server)
        ]
        if unexpected_calls:
            names = sorted({str(event.get("tool_name") or "unknown") for event in unexpected_calls})
            errors.append(
                f"tools teacher may call only {server} tools; observed unexpected tools: {names}"
            )
        if not matching_calls:
            errors.append(
                f"teacher must make at least one observable {server} tool call before answering"
            )
        elif not any(
            event.get("status") == "success" and _has_nonempty_observation(event)
            for event in matching_calls
        ):
            errors.append(
                f"teacher must complete at least one successful {server} tool call "
                "with a non-empty observation before answering"
            )
    return errors


def validate_judge_output(task: dict[str, Any], output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("task_id", "case_id", "candidate_label"):
        if output.get(field) != task[field]:
            errors.append(f"{field} must equal {task[field]!r}")

    rubric_steps = {step["step_id"]: step for step in task["rubric"]["steps"]}
    steps = output.get("steps")
    if not isinstance(steps, list):
        errors.append("steps must be an array")
        steps = []
    seen: set[Any] = set()
    score_sum = 0.0
    for index, step in enumerate(steps):
        prefix = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{prefix} must be an object")
            continue
        step_id = step.get("step_id")
        if step_id not in rubric_steps:
            errors.append(f"{prefix}.step_id is unknown: {step_id!r}")
            continue
        if step_id in seen:
            errors.append(f"duplicate rubric step: {step_id!r}")
            continue
        seen.add(step_id)
        maximum = float(rubric_steps[step_id]["max_score"])
        if not _is_number(step.get("max_score")) or abs(float(step["max_score"]) - maximum) > 1e-6:
            errors.append(f"{prefix}.max_score must equal {maximum}")
        score = step.get("score")
        if not _is_number(score):
            errors.append(f"{prefix}.score must be a number")
        else:
            score_value = float(score)
            score_sum += score_value
            if score_value < 0 or score_value > maximum:
                errors.append(f"{prefix}.score {score_value} is outside 0..{maximum}")
        for field in ("evidence", "diagnosis"):
            if not isinstance(step.get(field), str) or not step[field].strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
        codes = step.get("failure_codes")
        if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
            errors.append(f"{prefix}.failure_codes must be an array of strings")
        else:
            invalid = sorted(set(codes) - set(FAILURE_CODES))
            if invalid:
                errors.append(f"{prefix}.failure_codes contains invalid values: {invalid}")

    missing = list(rubric_steps.keys() - seen)
    if missing:
        errors.append(f"steps is missing rubric step IDs: {sorted(missing, key=str)}")
    total = output.get("total_score")
    if not _is_number(total):
        errors.append("total_score must be a number")
    elif abs(float(total) - score_sum) > 1e-6:
        errors.append(f"total_score {float(total)} does not equal step sum {score_sum}")
    if not isinstance(output.get("overall_diagnosis"), str) or not output["overall_diagnosis"].strip():
        errors.append("overall_diagnosis must be a non-empty string")
    efficiency_rubric = task.get("tool_efficiency_rubric")
    if efficiency_rubric is not None:
        dimensions = efficiency_rubric.get("dimensions", [])
        dimensions_by_id = {item["dimension_id"]: item for item in dimensions}
        efficiency_rows = output.get("tool_efficiency_dimensions")
        if not isinstance(efficiency_rows, list):
            errors.append("tool_efficiency_dimensions must be an array")
            efficiency_rows = []
        efficiency_seen: set[Any] = set()
        efficiency_sum = 0.0
        for index, row in enumerate(efficiency_rows):
            prefix = f"tool_efficiency_dimensions[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{prefix} must be an object")
                continue
            dimension_id = row.get("dimension_id")
            if dimension_id not in dimensions_by_id:
                errors.append(f"{prefix}.dimension_id is unknown: {dimension_id!r}")
                continue
            if dimension_id in efficiency_seen:
                errors.append(f"duplicate tool efficiency dimension: {dimension_id!r}")
                continue
            efficiency_seen.add(dimension_id)
            maximum = float(dimensions_by_id[dimension_id]["max_score"])
            if not _is_number(row.get("max_score")) or abs(float(row["max_score"]) - maximum) > 1e-6:
                errors.append(f"{prefix}.max_score must equal {maximum}")
            score = row.get("score")
            if not _is_number(score):
                errors.append(f"{prefix}.score must be a number")
            else:
                score_value = float(score)
                efficiency_sum += score_value
                if score_value < 0 or score_value > maximum:
                    errors.append(f"{prefix}.score {score_value} is outside 0..{maximum}")
            for field in ("evidence", "diagnosis"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    errors.append(f"{prefix}.{field} must be a non-empty string")
        missing_dimensions = sorted(set(dimensions_by_id) - efficiency_seen)
        if missing_dimensions:
            errors.append(
                "tool_efficiency_dimensions is missing dimension IDs: "
                f"{missing_dimensions}"
            )
        efficiency_score = output.get("tool_efficiency_score")
        if not _is_number(efficiency_score):
            errors.append("tool_efficiency_score must be a number")
        elif abs(float(efficiency_score) - efficiency_sum) > 1e-6:
            errors.append(
                f"tool_efficiency_score {float(efficiency_score)} does not equal "
                f"dimension sum {efficiency_sum}"
            )
        if (
            not isinstance(output.get("tool_efficiency_overall_diagnosis"), str)
            or not output["tool_efficiency_overall_diagnosis"].strip()
        ):
            errors.append("tool_efficiency_overall_diagnosis must be a non-empty string")
    trajectory_expected = "trajectory_analysis" in task.get("expected_output", {})
    trajectory_analysis = output.get("trajectory_analysis")
    if trajectory_expected and trajectory_analysis is None:
        errors.append("trajectory_analysis is required by this judge task")
    if trajectory_analysis is not None:
        if not isinstance(trajectory_analysis, dict):
            errors.append("trajectory_analysis must be an object when present")
        else:
            observable = task.get("observable_trajectory") or {}
            observable_event_ids = {
                event.get("event_id")
                for event in observable.get("events", [])
                if isinstance(event, dict) and event.get("event_id")
            }
            if trajectory_analysis.get("trajectory_source") != observable.get("source"):
                errors.append("trajectory_analysis.trajectory_source must match observable_trajectory.source")
            path_classification = trajectory_analysis.get("path_classification")
            if path_classification not in {
                "golden_aligned", "valid_alternative", "invalid", "insufficient_trace"
            }:
                errors.append("trajectory_analysis.path_classification is invalid")
            if not isinstance(trajectory_analysis.get("summary"), str) or not trajectory_analysis["summary"].strip():
                errors.append("trajectory_analysis.summary must be a non-empty string")
            first_event = trajectory_analysis.get("first_error_event_id")
            if first_event is not None and first_event not in observable_event_ids:
                errors.append("trajectory_analysis.first_error_event_id must be an observable event ID or null")
            recovery_attempted = trajectory_analysis.get("recovery_attempted")
            recovery_succeeded = trajectory_analysis.get("recovery_succeeded")
            if not isinstance(recovery_attempted, bool):
                errors.append("trajectory_analysis.recovery_attempted must be boolean")
            if recovery_attempted and not isinstance(recovery_succeeded, bool):
                errors.append("trajectory_analysis.recovery_succeeded must be boolean when recovery was attempted")
            if recovery_attempted is False and recovery_succeeded is not None:
                errors.append("trajectory_analysis.recovery_succeeded must be null when no recovery was attempted")
            assessments = trajectory_analysis.get("event_assessments")
            if not isinstance(assessments, list):
                errors.append("trajectory_analysis.event_assessments must be an array")
                assessments = []
            assessment_seen: set[Any] = set()
            attributed_loss = 0.0
            for index, assessment in enumerate(assessments):
                prefix = f"trajectory_analysis.event_assessments[{index}]"
                if not isinstance(assessment, dict):
                    errors.append(f"{prefix} must be an object")
                    continue
                event_id = assessment.get("event_id")
                if event_id not in observable_event_ids:
                    errors.append(f"{prefix}.event_id must be an observable event ID")
                elif event_id in assessment_seen:
                    errors.append(f"duplicate trajectory event assessment: {event_id!r}")
                assessment_seen.add(event_id)
                if assessment.get("verdict") not in {
                    "correct", "incorrect", "redundant", "recovered", "insufficient_evidence"
                }:
                    errors.append(f"{prefix}.verdict is invalid")
                codes = assessment.get("failure_codes")
                if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
                    errors.append(f"{prefix}.failure_codes must be an array of strings")
                else:
                    invalid = sorted(set(codes) - set(FAILURE_CODES))
                    if invalid:
                        errors.append(f"{prefix}.failure_codes contains invalid values: {invalid}")
                primary_code = assessment.get("primary_failure_code")
                if primary_code is not None and primary_code not in FAILURE_CODES:
                    errors.append(f"{prefix}.primary_failure_code is invalid")
                if isinstance(codes, list) and primary_code is not None and primary_code not in codes:
                    errors.append(f"{prefix}.primary_failure_code must appear in failure_codes")
                for field in ("evidence", "diagnosis"):
                    if not isinstance(assessment.get(field), str) or not assessment[field].strip():
                        errors.append(f"{prefix}.{field} must be a non-empty string")
                affected_steps = assessment.get("affected_rubric_steps")
                if not isinstance(affected_steps, list) or any(
                    step_id not in rubric_steps for step_id in affected_steps
                ):
                    errors.append(f"{prefix}.affected_rubric_steps must contain rubric step IDs")
                loss = assessment.get("attributed_task_loss")
                if not _is_number(loss) or float(loss) < 0:
                    errors.append(f"{prefix}.attributed_task_loss must be a non-negative number")
                else:
                    loss_value = float(loss)
                    attributed_loss += loss_value
                    if loss_value > 0 and primary_code is None:
                        errors.append(f"{prefix}.primary_failure_code is required when attributed_task_loss > 0")
            if _is_number(total):
                expected_loss = max(0.0, 100.0 - float(total))
                if abs(attributed_loss - expected_loss) > 1e-6:
                    errors.append(
                        f"trajectory attributed task loss {attributed_loss} does not equal "
                        f"100 - total_score ({expected_loss})"
                    )
    suggestions = output.get("skill_improvement_suggestions")
    if not isinstance(suggestions, list) or any(not isinstance(item, str) for item in suggestions):
        errors.append("skill_improvement_suggestions must be an array of strings")
    causal = output.get("causal_analysis")
    expected_output = task.get("expected_output", {})
    if "causal_analysis" in expected_output and causal is None:
        errors.append("causal_analysis is required by this judge task")
    if causal is not None:
        if not isinstance(causal, dict):
            errors.append("causal_analysis must be an object when present")
        else:
            first_error = causal.get("first_error_step_id")
            if first_error is not None and first_error not in rubric_steps:
                errors.append("causal_analysis.first_error_step_id must be a rubric step ID or null")
            for field in ("root_cause", "minimal_fix", "counterfactual_outcome"):
                if not isinstance(causal.get(field), str) or not causal[field].strip():
                    errors.append(f"causal_analysis.{field} must be a non-empty string")
            strength = causal.get("evidence_strength")
            if strength not in {"direct", "inferred", "insufficient"}:
                errors.append(
                    "causal_analysis.evidence_strength must be direct, inferred, or insufficient"
                )
            propagation = causal.get("error_propagation")
            if not isinstance(propagation, list) or any(
                not isinstance(item, str) or not item.strip() for item in propagation
            ):
                errors.append("causal_analysis.error_propagation must be an array of strings")
            affected = causal.get("downstream_affected_steps")
            if not isinstance(affected, list) or any(item not in rubric_steps for item in affected):
                errors.append(
                    "causal_analysis.downstream_affected_steps must contain rubric step IDs"
                )
    research_tags = output.get("research_tags")
    if "research_tags" in expected_output and research_tags is None:
        errors.append("research_tags is required by this judge task")
    if research_tags is not None and (
        not isinstance(research_tags, list)
        or any(not isinstance(item, str) or not item.strip() for item in research_tags)
    ):
        errors.append("research_tags must be an array of non-empty strings when present")
    return errors


def validate_task_output(stage: str, task: dict[str, Any], output: dict[str, Any]) -> list[str]:
    if stage == "teacher":
        return validate_teacher_output(task, output)
    if stage == "judge":
        return validate_judge_output(task, output)
    return [f"unknown Codex stage: {stage}"]


def build_task_prompt(stage: str, task: dict[str, Any]) -> str:
    if stage == "teacher":
        tool_policy = task.get("tool_policy") or {}
        server = tool_policy.get("mcp_server", "watertap")
        role = (
            "You are the blind upper-reference teacher for an SWRO engineering benchmark. "
            "Solve only from the supplied question. Do not use a reference answer or rubric. "
            "Preserve units, state assumptions, show the calculation path, check every constraint, "
            "and never claim a simulation or tool call that did not occur."
        )
        if tool_policy.get("forbid_observable_calls"):
            isolation_rules = (
                "Isolation rules:\n"
                "- This is one independent task in a fresh conversation.\n"
                "- Use only the JSON task below as benchmark content; do not inspect local benchmark, "
                "reference-answer, rubric, response, rating, or report files.\n"
                "- Do not run shell commands, browse the web, or call any tool or MCP server.\n"
            )
        else:
            isolation_rules = (
                "Isolation rules:\n"
                "- This is one independent task in a fresh conversation.\n"
                "- Use only the JSON task below as benchmark content; do not inspect local benchmark, "
                "reference-answer, rubric, response, rating, or report files.\n"
                "- Do not run shell commands or browse the web.\n"
                f"- You are allowed to call the configured MCP server `{server}` and must do so; base reported numerical "
                "results on returned observations.\n"
            )
    elif stage == "judge":
        role = (
            "You are the anonymous rubric judge for one SWRO benchmark response. "
            "Use the supplied reference as a correctness anchor, not a mandatory trajectory. Award partial credit step by step, "
            "cite concise response evidence, and never infer candidate identity."
        )
        isolation_rules = (
            "Isolation rules:\n"
            "- This is one independent task in a fresh conversation.\n"
            "- Use only the JSON task below. Do not inspect files, run commands, browse, or call tools.\n"
        )
    else:
        raise ValueError(f"unknown Codex stage: {stage}")
    return (
        f"{role}\n\n"
        f"{isolation_rules}"
        "- Return exactly one JSON object matching expected_output.\n"
        "- Do not use Markdown fences or add text outside the JSON object.\n"
        "- Preserve all identifier fields exactly.\n\n"
        "Task JSON:\n"
        + json.dumps(task, ensure_ascii=False, indent=2)
    )

def build_repair_prompt(stage: str, task: dict[str, Any], errors: list[str]) -> str:
    tool_instruction = ""
    tool_policy = task.get("tool_policy") or {}
    if stage == "teacher" and tool_policy.get("require_observable_call"):
        server = tool_policy.get("mcp_server", "watertap")
        tool_instruction = (
            f"Before returning the corrected object, call the configured `{server}` MCP tools "
            "needed to solve the benchmark; a text-only claim is not sufficient.\n"
        )
    return (
        "Your previous response failed machine validation. Correct it in this same conversation.\n"
        "Validation errors:\n- "
        + "\n- ".join(errors)
        + "\n\n"
        + tool_instruction
        + "Return exactly one corrected JSON object and no other text. "
        "Use the original task identifiers and expected_output structure.\n\nOriginal task:\n"
        + json.dumps(task, ensure_ascii=False, indent=2)
        + f"\n\nStage: {stage}"
    )


Invoker = Callable[[dict[str, Any], Path, int], dict[str, Any]]


def invoke_codex_worker(request: dict[str, Any], project_root: Path, timeout_seconds: int) -> dict[str, Any]:
    src_dir = project_root / "src"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing if existing else "")
    request = {**request, "timeout_seconds": timeout_seconds}
    request_bytes = json.dumps(request, ensure_ascii=False).encode("utf-8")
    request_payload = _REQUEST_B64_PREFIX + base64.b64encode(request_bytes).decode("ascii")
    with tempfile.TemporaryDirectory(prefix="swro-ae-codex-") as temp_dir:
        completed = subprocess.run(
            [sys.executable, "-m", "auto_evaluate.codex_worker"],
            input=request_payload,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=temp_dir,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
    result_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith((_RESULT_B64_PREFIX, _RESULT_PREFIX))
        ),
        None,
    )
    if result_line is None:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no worker output"
        raise CodexAutomationError(f"Codex worker failed (exit {completed.returncode}): {detail[-2000:]}")
    if result_line.startswith(_RESULT_B64_PREFIX):
        encoded = result_line[len(_RESULT_B64_PREFIX) :]
        try:
            result_bytes = base64.b64decode(encoded, validate=True)
            result = json.loads(result_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodexAutomationError(
                f"Codex worker returned an invalid encoded result: {exc}"
            ) from exc
    else:
        result = json.loads(result_line[len(_RESULT_PREFIX) :])
    if not result.get("ok"):
        error = CodexAutomationError(
            result.get("error") or "Codex worker returned an unknown error",
            result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else None,
        )
        raise error
    return result


def _record_path(run_dir: Path, stage: str, task: dict[str, Any]) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", task["task_id"]).strip("-")[:96]
    suffix = stable_hash(task["task_id"])[0:12]
    return run_dir / "codex" / "records" / stage / f"{safe_id}--{suffix}.json"


def _write_audit(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    records_dir = run_dir / "codex" / "records"
    for path in sorted(records_dir.glob("*/*.json")) if records_dir.exists() else []:
        record = read_json(path)
        rows.append(
            {
                key: record.get(key)
                for key in (
                    "stage",
                    "task_id",
                    "status",
                    "model",
                    "thread_id",
                    "input_hash",
                    "attempts",
                    "latency_ms",
                    "started_at",
                    "completed_at",
                )
            }
        )
    write_jsonl(run_dir / "codex" / "audit.jsonl", rows)


def run_codex_tasks(
    *,
    stage: str,
    tasks: list[dict[str, Any]],
    run_dir: Path,
    project_root: Path,
    model: str,
    concurrency: int = 1,
    retries: int = 2,
    timeout_seconds: int = 3600,
    force: bool = False,
    invoker: Invoker = invoke_codex_worker,
) -> list[dict[str, Any]]:
    require_chatgpt_auth_environment()
    if concurrency < 1 or concurrency > 8:
        raise ValueError("codex concurrency must be between 1 and 8")
    if retries < 0:
        raise ValueError("codex retries cannot be negative")
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], Path, str, str]] = []
    for task in tasks:
        prompt = build_task_prompt(stage, task)
        input_hash = stable_hash(
            {"schema": "2.1", "stage": stage, "model": model, "task": task, "prompt": prompt}
        )
        record_path = _record_path(run_dir, stage, task)
        if not force and record_path.exists():
            record = read_json(record_path)
            output = record.get("output")
            if (
                record.get("status") == "success"
                and record.get("input_hash") == input_hash
                and isinstance(output, dict)
                and not validate_task_output(stage, task, output)
            ):
                results[task["task_id"]] = output
                print(f"[codex:{stage}] cache {task['task_id']}")
                continue
        pending.append((task, record_path, input_hash, prompt))

    validate_stage_environment(stage, [item[0] for item in pending])

    def execute(item: tuple[dict[str, Any], Path, str, str]) -> tuple[str, dict[str, Any]]:
        task, record_path, input_hash, prompt = item
        started_at = utc_now()
        started = time.perf_counter()
        request = {
            "schema_version": "2.0",
            "stage": stage,
            "model": model,
            "task": task,
            "prompt": prompt,
            "retries": retries,
        }
        try:
            worker_result = invoker(request, project_root, timeout_seconds)
            output = worker_result.get("output")
            if not isinstance(output, dict):
                raise CodexAutomationError("Codex worker did not return an output object")
            validation_errors = validate_task_output(stage, task, output)
            if validation_errors:
                raise CodexAutomationError("invalid Codex output: " + "; ".join(validation_errors))
            record = {
                "schema_version": "2.0",
                "stage": stage,
                "task_id": task["task_id"],
                "status": "success",
                "model": model,
                "thread_id": worker_result.get("thread_id"),
                "input_hash": input_hash,
                "attempts": worker_result.get("attempts"),
                "latency_ms": worker_result.get("latency_ms"),
                "started_at": started_at,
                "completed_at": utc_now(),
                "output": output,
            }
            diagnostics = worker_result.get("diagnostics")
            if isinstance(diagnostics, dict) and diagnostics:
                record["diagnostics"] = diagnostics
            write_json(record_path, record)
            return task["task_id"], output
        except Exception as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            write_json(
                record_path,
                {
                    "schema_version": "2.0",
                    "stage": stage,
                    "task_id": task["task_id"],
                    "status": "error",
                    "model": model,
                    "input_hash": input_hash,
                    "thread_id": diagnostics.get("thread_id") if isinstance(diagnostics, dict) else None,
                    "attempts": diagnostics.get("attempt") if isinstance(diagnostics, dict) else None,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "started_at": started_at,
                    "completed_at": utc_now(),
                    "error": str(exc),
                    **(
                        {"diagnostics": diagnostics}
                        if isinstance(diagnostics, dict) and diagnostics
                        else {}
                    ),
                },
            )
            raise

    failures: list[str] = []
    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {executor.submit(execute, item): item[0]["task_id"] for item in pending}
            for future in as_completed(future_map):
                task_id = future_map[future]
                try:
                    completed_task_id, output = future.result()
                    results[completed_task_id] = output
                    print(f"[codex:{stage}] done {completed_task_id}")
                except Exception as exc:
                    failures.append(f"{task_id}: {exc}")
                    print(f"[codex:{stage}] ERROR {task_id}: {exc}", file=sys.stderr)
    _write_audit(run_dir)
    if failures:
        raise CodexAutomationError(
            f"{len(failures)} {stage} task(s) failed; successful tasks are cached. " + " | ".join(failures[:3])
        )
    return [results[task["task_id"]] for task in tasks]
