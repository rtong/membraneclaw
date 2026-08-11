from __future__ import annotations

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
from .judge import FAILURE_CODES


class CodexAutomationError(RuntimeError):
    """Raised when a Codex teacher or judge task cannot be completed safely."""


_RESULT_PREFIX = "__AE_CODEX_RESULT__"
_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "CODEX_API_KEY")


def require_chatgpt_auth_environment() -> None:
    configured = [name for name in _API_KEY_ENV_VARS if os.environ.get(name)]
    if configured:
        names = ", ".join(configured)
        raise CodexAutomationError(
            f"Refusing to run with Platform API credentials present ({names}). "
            "Remove them from this shell and authenticate Codex with the ChatGPT account login instead."
        )


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


def validate_teacher_output(task: dict[str, Any], output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "task_id": task["task_id"],
        "case_id": task["case_id"],
        "system_id": "gpt-5.6-teacher",
    }
    for field, value in expected.items():
        if output.get(field) != value:
            errors.append(f"{field} must equal {value!r}")
    if not isinstance(output.get("response_text"), str) or not output["response_text"].strip():
        errors.append("response_text must be a non-empty string")
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
    suggestions = output.get("skill_improvement_suggestions")
    if not isinstance(suggestions, list) or any(not isinstance(item, str) for item in suggestions):
        errors.append("skill_improvement_suggestions must be an array of strings")
    return errors


def validate_task_output(stage: str, task: dict[str, Any], output: dict[str, Any]) -> list[str]:
    if stage == "teacher":
        return validate_teacher_output(task, output)
    if stage == "judge":
        return validate_judge_output(task, output)
    return [f"unknown Codex stage: {stage}"]


def build_task_prompt(stage: str, task: dict[str, Any]) -> str:
    if stage == "teacher":
        role = (
            "You are the blind upper-reference teacher for an SWRO engineering benchmark. "
            "Solve only from the supplied question. Do not use a reference answer or rubric. "
            "Preserve units, state assumptions, show the calculation path, check every constraint, "
            "and never claim a simulation or tool call that did not occur."
        )
    elif stage == "judge":
        role = (
            "You are the anonymous rubric judge for one SWRO benchmark response. "
            "Score only against the supplied reference and rubric, award partial credit step by step, "
            "cite concise response evidence, and never infer candidate identity."
        )
    else:
        raise ValueError(f"unknown Codex stage: {stage}")
    return (
        f"{role}\n\n"
        "Isolation rules:\n"
        "- This is one independent task in a fresh conversation.\n"
        "- Use only the JSON task below. Do not inspect files, run commands, browse, or call tools.\n"
        "- Return exactly one JSON object matching expected_output.\n"
        "- Do not use Markdown fences or add text outside the JSON object.\n"
        "- Preserve all identifier fields exactly.\n\n"
        "Task JSON:\n"
        + json.dumps(task, ensure_ascii=False, indent=2)
    )


def build_repair_prompt(stage: str, task: dict[str, Any], errors: list[str]) -> str:
    return (
        "Your previous response failed machine validation. Correct it in this same conversation.\n"
        "Validation errors:\n- "
        + "\n- ".join(errors)
        + "\n\nReturn exactly one corrected JSON object and no other text. "
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
    with tempfile.TemporaryDirectory(prefix="swro-ae-codex-") as temp_dir:
        completed = subprocess.run(
            [sys.executable, "-m", "auto_evaluate.codex_worker"],
            input=json.dumps(request, ensure_ascii=False),
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
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(_RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no worker output"
        raise CodexAutomationError(f"Codex worker failed (exit {completed.returncode}): {detail[-2000:]}")
    result = json.loads(result_line[len(_RESULT_PREFIX) :])
    if not result.get("ok"):
        raise CodexAutomationError(result.get("error") or "Codex worker returned an unknown error")
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
    if concurrency < 1 or concurrency > 4:
        raise ValueError("codex concurrency must be between 1 and 4")
    if retries < 0:
        raise ValueError("codex retries cannot be negative")

    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], Path, str, str]] = []
    for task in tasks:
        prompt = build_task_prompt(stage, task)
        input_hash = stable_hash({"schema": "2.0", "stage": stage, "model": model, "task": task})
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
            write_json(record_path, record)
            return task["task_id"], output
        except Exception as exc:
            write_json(
                record_path,
                {
                    "schema_version": "2.0",
                    "stage": stage,
                    "task_id": task["task_id"],
                    "status": "error",
                    "model": model,
                    "input_hash": input_hash,
                    "attempts": None,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "started_at": started_at,
                    "completed_at": utc_now(),
                    "error": str(exc),
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
