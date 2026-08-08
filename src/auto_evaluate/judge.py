from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .benchmark import iter_benchmarks
from .io_utils import read_json, read_jsonl, utc_now, write_json, write_jsonl


FAILURE_CODES = [
    "TASK_CLASSIFICATION",
    "PARAMETER_EXTRACTION",
    "UNIT_CONVERSION",
    "TOOL_NOT_CALLED",
    "TOOL_ARGUMENT",
    "SEARCH_STRATEGY",
    "CONSTRAINT_OMISSION",
    "NUMERICAL_REASONING",
    "OUTPUT_OMISSION",
    "OVERCLAIM",
    "ENGINEERING_JUDGMENT",
    "OTHER",
]

SCORE_POINTS_BEGIN = "[SCORE_POINTS_BEGIN]"
SCORE_POINTS_END = "[SCORE_POINTS_END]"


def prepare_teacher_tasks(benchmarks_dir: Path, run_dir: Path) -> Path:
    rows = []
    for benchmark in iter_benchmarks(benchmarks_dir):
        rows.append(
            {
                "task_id": f"teacher::{benchmark['case_id']}",
                "case_id": benchmark["case_id"],
                "model": "gpt-5.6-teacher",
                "instructions": (
                    "Answer the benchmark blind. Do not use its reference answer or rubric. "
                    "Use available WaterTAP tooling if configured; otherwise state clearly which "
                    "values are calculated, simulated, or estimated. Preserve units and check all constraints."
                ),
                "question": benchmark["question_prompt"],
                "expected_output": {
                    "task_id": "same as input",
                    "case_id": benchmark["case_id"],
                    "system_id": "gpt-5.6-teacher",
                    "response_text": "complete answer",
                },
            }
        )
    output = run_dir / "teacher_batch.jsonl"
    write_jsonl(output, rows)
    (run_dir / "TEACHER_INSTRUCTIONS.md").write_text(
        "# GPT-5.6 teacher batch\n\n"
        "Process each line of `teacher_batch.jsonl` independently and blind to the benchmark "
        "answer sheets. Write one JSON object per line to `teacher_responses.jsonl` using the "
        "declared `expected_output` fields. Do not wrap JSONL in a Markdown fence.\n",
        encoding="utf-8",
    )
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
        if row.get("status") != "success":
            continue
        candidates.setdefault(row["case_id"], []).append(
            {
                "system_id": row["system_id"],
                "display_name": row.get("display_name", row["system_id"]),
                "response_text": row["response_text"],
                "score_points": extract_score_points(row["response_text"]),
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
                        "Score only against the supplied rubric. Do not infer the model identity. "
                        "Use candidate_score_points as a high-signal summary when present, but verify "
                        "claims against candidate_response whenever needed. If the structured score points "
                        "conflict with the full response, trust the verifiable response evidence and "
                        "penalize overclaim. Award partial credit step by step, cite concise evidence "
                        "from the response, and distinguish a wrong result from a correct result with "
                        "incomplete explanation. Use the failure code vocabulary. The sum of step scores "
                        "must equal total_score."
                    ),
                    "question": benchmark["question_prompt"],
                    "reference_answer": benchmark["reference_answer"],
                    "rubric": benchmark["rubric"],
                    "candidate_response": candidate["response_text"],
                    "candidate_score_points": candidate.get("score_points"),
                    "failure_code_vocabulary": FAILURE_CODES,
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
    (run_dir / "JUDGE_INSTRUCTIONS.md").write_text(
        "# GPT-5.6 judge batch\n\n"
        "Process each line of `judge_batch.jsonl` independently. Write one JSON object per line "
        "to `ratings.jsonl` using the exact `expected_output` structure. Preserve `task_id`, "
        "`case_id`, and `candidate_label`. Do not inspect `judge_mapping.json` while scoring. "
        "Do not wrap JSONL in a Markdown fence.\n",
        encoding="utf-8",
    )
    return output


def validate_ratings(run_dir: Path) -> list[str]:
    tasks = {row["task_id"]: row for row in read_jsonl(run_dir / "judge_batch.jsonl")}
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    errors: list[str] = []
    seen: set[str] = set()
    for rating in ratings:
        task_id = rating.get("task_id")
        if task_id not in tasks:
            errors.append(f"unknown rating task_id: {task_id}")
            continue
        if task_id in seen:
            errors.append(f"duplicate rating task_id: {task_id}")
        seen.add(task_id)
        rubric_steps = {step["step_id"]: step for step in tasks[task_id]["rubric"]["steps"]}
        score_sum = 0.0
        for step in rating.get("steps", []):
            step_id = step.get("step_id")
            if step_id not in rubric_steps:
                errors.append(f"{task_id}: unknown step {step_id}")
                continue
            score = float(step.get("score", 0))
            maximum = float(rubric_steps[step_id]["max_score"])
            if score < 0 or score > maximum:
                errors.append(f"{task_id}: step {step_id} score {score} outside 0..{maximum}")
            score_sum += score
            invalid_codes = set(step.get("failure_codes", [])) - set(FAILURE_CODES)
            if invalid_codes:
                errors.append(f"{task_id}: invalid failure codes {sorted(invalid_codes)}")
        total = float(rating.get("total_score", 0))
        if abs(total - score_sum) > 1e-6:
            errors.append(f"{task_id}: total_score {total} != step sum {score_sum}")
    missing = set(tasks) - seen
    if ratings and missing:
        errors.append(f"missing ratings: {len(missing)}")
    return errors
