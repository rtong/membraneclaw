from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .benchmark import iter_benchmarks
from .io_utils import read_json, read_jsonl, utc_now, write_json, write_jsonl
from .taxonomy import FAILURE_CODES, failure_code_payload
from .trajectory import extract_observable_trajectory


SCORE_POINTS_BEGIN = "[SCORE_POINTS_BEGIN]"
SCORE_POINTS_END = "[SCORE_POINTS_END]"

# Upper bound on trajectory events sent to a judge. A runaway teacher/agent can
# emit hundreds of tool events; shipping all of them blows the Codex input limit
# (~1 MB). Keep the first N events plus the aggregate summary so the judge still
# sees the call-overload fact while the prompt stays bounded.
_MAX_TRAJECTORY_EVENTS = 80


def _limit_trajectory(trajectory: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trajectory or not isinstance(trajectory, dict):
        return trajectory
    events = trajectory.get("events") or []
    if len(events) <= _MAX_TRAJECTORY_EVENTS:
        return trajectory
    kept = events[:_MAX_TRAJECTORY_EVENTS]
    summary = dict(trajectory.get("summary") or {})
    summary["truncated_event_count"] = len(events)
    return {
        **trajectory,
        "events": kept,
        "summary": summary,
        "truncation_note": (
            f"trajectory truncated from {len(events)} to {_MAX_TRAJECTORY_EVENTS} events "
            "to stay within the judge input limit; aggregate counts preserved"
        ),
    }


def prepare_teacher_tasks(
    benchmarks_dir: Path,
    run_dir: Path,
    teacher_profile: dict[str, Any] | None = None,
) -> Path:
    profile = teacher_profile or {
        "id": "tools",
        "system_id": "gpt-5.6-teacher-tools",
        "display_name": "GPT-5.6 Teacher + Tools",
        "tools_enabled": True,
        "mcp_server": "watertap",
    }
    profile_id = str(profile["id"])
    system_id = str(profile["system_id"])
    tools_enabled = bool(profile.get("tools_enabled"))
    server = str(profile.get("mcp_server") or "watertap")
    rows = []
    for benchmark in iter_benchmarks(benchmarks_dir):
        if tools_enabled:
            instructions = (
                "Answer the benchmark blind. Do not use its reference answer or rubric. "
                f"Use the configured {server} MCP tools for every calculation or simulation "
                "needed by the question. State clearly which values are calculated, simulated, "
                "or estimated. Preserve units and check all constraints."
            )
        else:
            instructions = (
                "Answer the benchmark blind using only the supplied question and your general "
                "reasoning. Do not call any tool, MCP server, browser, or local file. Do not use "
                "its reference answer or rubric. Preserve units and check all constraints."
            )
        rows.append(
            {
                "task_id": f"teacher-{profile_id}::{benchmark['case_id']}",
                "case_id": benchmark["case_id"],
                "model": system_id,
                "display_name": profile.get("display_name", system_id),
                "teacher_profile": profile_id,
                "instructions": instructions,
                "question": benchmark["question_prompt"],
                "tool_policy": {
                    "mode": "required" if tools_enabled else "forbidden",
                    "required": tools_enabled,
                    "mcp_server": server if tools_enabled else None,
                    "require_observable_call": tools_enabled,
                    "forbid_observable_calls": not tools_enabled,
                    **({"require_successful_observation": True} if tools_enabled else {}),
                },
                "expected_output": {
                    "task_id": "same as input",
                    "case_id": benchmark["case_id"],
                    "system_id": system_id,
                    "response_text": "complete answer",
                },
            }
        )
    output = run_dir / f"teacher_{profile_id}_batch.jsonl"
    write_jsonl(output, rows)
    return output


def extract_score_points(response_text: str) -> dict[str, Any] | None:
    start = response_text.find(SCORE_POINTS_BEGIN)
    end = response_text.find(SCORE_POINTS_END)
    if start < 0 or end < 0 or end <= start:
        return None
    payload = response_text[start + len(SCORE_POINTS_BEGIN):end].strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {
            "status": "invalid_json",
            "raw_text": payload,
        }
    if not isinstance(data, dict):
        return {
            "status": "invalid_shape",
            "raw_text": payload,
        }
    return {
        "status": "ok",
        "data": data,
    }


def _load_candidates(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((run_dir / "responses").glob("*.json")):
        row = read_json(path)
        # A policy-replay response is byte-for-byte evidence from the selected
        # physical arm. Judging it again would duplicate both cost and scores;
        # reward analysis inherits the already judged arm score instead.
        if row.get("completion_mode") == "policy_replay":
            continue
        candidates.setdefault(row["case_id"], []).append(
            {
                "system_id": row["system_id"],
                "display_name": row.get("display_name", row["system_id"]),
                "response_text": row.get("response_text") or "",
                "execution_status": row.get("status", "unknown"),
                "execution_error_type": row.get("error_type"),
                "execution_error": row.get("error"),
                "completion_mode": row.get("completion_mode", "unknown"),
                "native_execution_status": row.get(
                    "native_status", row.get("status", "unknown")
                ),
                "native_execution_error_type": row.get("native_error_type"),
                "recovery": row.get("recovery"),
                "score_points": extract_score_points(row.get("response_text") or ""),
                "trajectory": row.get("trajectory")
                or extract_observable_trajectory(
                    row.get("response_text") or "",
                    raw_response=row.get("raw_response"),
                    tools_enabled=row.get("tools_enabled"),
                    rag_enabled=row.get("rag_enabled"),
                ),
            }
        )
    for row in read_jsonl(run_dir / "teacher_responses.jsonl"):
        if row.get("response_text"):
            system_id = row.get("system_id", "gpt-5.6-teacher")
            candidates.setdefault(row["case_id"], []).append(
                {
                    "system_id": system_id,
                    "display_name": row.get("display_name") or "GPT-5.6 Teacher",
                    "response_text": row["response_text"],
                    "score_points": extract_score_points(row["response_text"]),
                    "trajectory": row.get("trajectory")
                    or extract_observable_trajectory(row["response_text"], tools_enabled=None),
                }
            )
    return candidates


def _case_rng(case_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def prepare_judge_tasks(
    benchmarks_dir: Path,
    run_dir: Path,
    *,
    seed: int = 20260806,
) -> Path:
    candidates = _load_candidates(run_dir)
    tasks: list[dict[str, Any]] = []
    mapping: list[dict[str, str]] = []

    for benchmark in iter_benchmarks(benchmarks_dir):
        case_candidates = list(candidates.get(benchmark["case_id"], []))
        _case_rng(benchmark["case_id"], seed).shuffle(case_candidates)
        for index, candidate in enumerate(case_candidates):
            label = f"Response {chr(ord('A') + index)}"
            task_id = f"judge::{benchmark['case_id']}::{label.replace(' ', '-').lower()}"
            mapping.append(
                {
                    "task_id": task_id,
                    "case_id": benchmark["case_id"],
                    "candidate_label": label,
                    "system_id": candidate["system_id"],
                    "display_name": candidate["display_name"],
                }
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "case_id": benchmark["case_id"],
                    "candidate_label": label,
                    "instructions": (
                        "Treat candidate_execution_status as observed evaluation evidence. When it is "
                        "error, score any useful work visible before failure normally, but do not infer a "
                        "missing final answer or unobserved checks. Attribute losses caused by execution "
                        "failure to OUTPUT_OMISSION and, when supported, SEARCH_STRATEGY or OTHER. Do not "
                        "discard the candidate and do not award credit for work that was never completed. "
                        "When candidate_completion_mode is context_reset_finalizer, score the recovered "
                        "final answer normally against the rubric while treating the native execution "
                        "failure and recovery metadata as observed reliability evidence. Do not infer "
                        "unobserved tool results during recovery. "
                        "Score only against the supplied rubric. Do not infer the model identity. "
                        "Use candidate_score_points as a high-signal summary when present, but verify "
                        "claims against candidate_response whenever needed. If the structured score points "
                        "conflict with the full response, trust the verifiable response evidence and "
                        "penalize overclaim. Award partial credit step by step, cite concise evidence "
                        "from the response, and distinguish a wrong result from a correct result with "
                        "incomplete explanation. Use the failure code vocabulary. The sum of step scores "
                        "must equal total_score. Identify the earliest evidence-supported error, trace how "
                        "it propagates into later reasoning or the final engineering decision, and state the "
                        "smallest intervention that would break that chain. Distinguish direct evidence from "
                        "inference. Use evaluation loss to mean rubric points lost, never training loss. "
                        "Independently score tool-use efficiency against tool_efficiency_rubric. Do not treat "
                        "fewer calls as automatically better: an efficient response must obtain enough evidence "
                        "to support a correct answer. Judge only tool behavior visible in candidate_response, "
                        "and do not mix the tool-efficiency score into total_score. Examine the supplied "
                        "observable_trajectory event by event. The reference answer is a correctness anchor, "
                        "not a mandatory action sequence: explicitly recognize a different but valid path. "
                        "Never invent hidden reasoning or unobserved calls. Attribute the complete task loss "
                        "(100 - total_score) across causal trajectory events; do not merely split loss evenly "
                        "across labels. If only a final answer is observable, mark the trace insufficient."
                    ),
                    "question": benchmark["question_prompt"],
                    "reference_answer": benchmark["reference_answer"],
                    "rubric": benchmark["rubric"],
                    "tool_efficiency_rubric": benchmark["tool_efficiency_rubric"],
                    "candidate_execution_status": candidate.get("execution_status", "success"),
                    "candidate_execution_error_type": candidate.get("execution_error_type"),
                    "candidate_execution_error": candidate.get("execution_error"),
                    "candidate_completion_mode": candidate.get("completion_mode", "unknown"),
                    "candidate_native_execution_status": candidate.get(
                        "native_execution_status", candidate.get("execution_status", "success")
                    ),
                    "candidate_native_execution_error_type": candidate.get(
                        "native_execution_error_type"
                    ),
                    "candidate_recovery": candidate.get("recovery"),
                    "candidate_response": candidate["response_text"],
                    "candidate_score_points": candidate.get("score_points"),
                    "observable_trajectory": _limit_trajectory(candidate.get("trajectory")),
                    "failure_code_vocabulary": failure_code_payload(),
                    "expected_output": {
                        "task_id": task_id,
                        "case_id": benchmark["case_id"],
                        "candidate_label": label,
                        "total_score": "number from 0 to 100",
                        "steps": [
                            {
                                "step_id": "integer",
                                "score": "number",
                                "max_score": "number copied from rubric",
                                "evidence": "short response excerpt or missing",
                                "diagnosis": "why points were gained/lost",
                                "failure_codes": ["zero or more allowed codes"],
                            }
                        ],
                        "overall_diagnosis": "concise summary",
                        "tool_efficiency_score": "number from 0 to 100, separate from total_score",
                        "tool_efficiency_dimensions": [
                            {
                                "dimension_id": "string copied from tool_efficiency_rubric",
                                "score": "number",
                                "max_score": "number copied from tool_efficiency_rubric",
                                "evidence": "visible tool-use evidence or explicit absence",
                                "diagnosis": "why efficiency points were gained/lost",
                            }
                        ],
                        "tool_efficiency_overall_diagnosis": "concise tool-use efficiency summary",
                        "trajectory_analysis": {
                            "trajectory_source": "copied from observable_trajectory.source",
                            "path_classification": "golden_aligned, valid_alternative, invalid, or insufficient_trace",
                            "summary": "observable execution-path summary",
                            "first_error_event_id": "observable event ID or null",
                            "recovery_attempted": "boolean",
                            "recovery_succeeded": "boolean or null",
                            "event_assessments": [
                                {
                                    "event_id": "ID copied from observable_trajectory",
                                    "verdict": "correct, incorrect, redundant, recovered, or insufficient_evidence",
                                    "failure_codes": ["zero or more allowed codes"],
                                    "primary_failure_code": "one allowed code or null when attributed_task_loss is zero",
                                    "evidence": "observable event evidence",
                                    "diagnosis": "event-level assessment",
                                    "affected_rubric_steps": ["integer rubric step IDs"],
                                    "attributed_task_loss": "non-negative number",
                                }
                            ],
                        },
                        "causal_analysis": {
                            "first_error_step_id": "integer or null",
                            "root_cause": "earliest evidence-supported failure",
                            "error_propagation": ["ordered causal links to downstream decision loss"],
                            "downstream_affected_steps": ["integer step IDs"],
                            "minimal_fix": "smallest intervention that breaks the failure chain",
                            "counterfactual_outcome": "likely corrected outcome after the intervention",
                            "evidence_strength": "direct, inferred, or insufficient",
                        },
                        "research_tags": ["reusable paper-analysis tags"],
                        "skill_improvement_suggestions": [
                            "method-level suggestions; do not copy benchmark answer values into the skill"
                        ],
                    },
                }
            )

    output = run_dir / "judge_batch.jsonl"
    write_jsonl(output, tasks)
    write_json(
        run_dir / "judge_mapping.json",
        {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "seed": seed,
            "mapping": mapping,
        },
    )
    return output


def validate_ratings(run_dir: Path) -> list[str]:
    from .codex_automation import validate_judge_output

    tasks = {row["task_id"]: row for row in read_jsonl(run_dir / "judge_batch.jsonl")}
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    errors: list[str] = []
    if not tasks:
        return ["judge_batch.jsonl is empty or missing"]
    if not ratings:
        return ["ratings.jsonl is empty or missing"]
    seen: set[str] = set()
    for rating in ratings:
        task_id = rating.get("task_id")
        if task_id not in tasks:
            errors.append(f"unknown rating task_id: {task_id}")
            continue
        if task_id in seen:
            errors.append(f"duplicate rating task_id: {task_id}")
        seen.add(task_id)
        errors.extend(f"{task_id}: {error}" for error in validate_judge_output(tasks[task_id], rating))
    missing = set(tasks) - seen
    if missing:
        errors.append(f"missing ratings: {len(missing)}")
    return errors
