from __future__ import annotations

import copy
import itertools
import random
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .codex_automation import run_codex_tasks
from .io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    utc_now,
    write_json,
    write_jsonl,
)
from .judge import prepare_judge_tasks, validate_ratings


CALIBRATION_ID = "evaluation-skill-calibration@0.1.0"


def _required_string(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"judge calibration config requires non-empty {field}")
    return value.strip()


def _required_string_list(document: dict[str, Any], field: str) -> list[str]:
    value = document.get(field)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"judge calibration config requires non-empty string array {field}")
    normalized = [item.strip() for item in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"judge calibration config contains duplicate {field}")
    return normalized


def load_calibration_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("judge calibration config schema_version must be 1.0")
    _required_string(config, "selection_id")
    _required_string(config, "source_run")
    _required_string(config, "evaluation_skill")
    _required_string_list(config, "case_ids")
    _required_string_list(config, "system_ids")
    conditions = _required_string_list(config, "judge_conditions")
    if set(conditions) != {"j0", "j1"}:
        raise ValueError("judge_conditions must contain exactly j0 and j1")
    repeats = config.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 2:
        raise ValueError("judge calibration repeats must be an integer >= 2")
    skill_path = (root / config["evaluation_skill"]).resolve()
    if not skill_path.is_file():
        raise FileNotFoundError(f"Evaluation Skill not found: {skill_path}")
    return config


def _skill_payload(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith("---\n"):
        raise ValueError("Evaluation Skill must start with YAML frontmatter")
    pieces = text.split("---", 2)
    if len(pieces) != 3 or "name: swro-evaluation-judge" not in pieces[1]:
        raise ValueError("Evaluation Skill frontmatter must name swro-evaluation-judge")
    body = pieces[2].strip()
    forbidden = ("D1-", "D2-", "D3-", "D4-", "D5-", "D6-", "D7-", "C00", "C10", "C01", "C11")
    leaked = [token for token in forbidden if token in body]
    if leaked:
        raise ValueError(f"Evaluation Skill contains experiment-specific token(s): {leaked}")
    return {
        "skill_id": "swro-evaluation-judge@0.1.0",
        "sha256": sha256_file(path),
        "instructions": body,
    }


def _observable_checks(response: dict[str, Any]) -> dict[str, Any]:
    trajectory = response.get("trajectory") or {}
    events = trajectory.get("events") or []
    calls = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event_type") == "tool_interaction"
    ]
    successful = [event for event in calls if event.get("status") == "success"]
    failed = [event for event in calls if event.get("status") == "error"]
    return {
        "source_response_present": bool(response.get("response_text")),
        "observable_trajectory_present": isinstance(trajectory, dict) and bool(trajectory),
        "observable_tool_interactions": len(calls),
        "successful_observable_tool_interactions": len(successful),
        "failed_observable_tool_interactions": len(failed),
        "execution_status": response.get("status", "missing"),
        "native_execution_status": response.get("native_status", response.get("status", "missing")),
        "completion_mode": response.get("completion_mode", "unknown"),
        "scope_note": "These checks summarize observable presence and status only; they do not establish numerical correctness.",
    }


def prepare_calibration(
    *, root: Path, config_path: Path, run_id: str
) -> dict[str, Any]:
    config = load_calibration_config(root, config_path)
    run_dir = (root / "runs" / run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"Calibration run already exists and will not be overwritten: {run_dir}"
        )
    source_run = (root / "runs" / config["source_run"]).resolve()
    source_benchmarks = source_run / "benchmarks"
    source_responses = source_run / "responses"
    if not source_benchmarks.is_dir() or not source_responses.is_dir():
        raise FileNotFoundError(
            f"Source run lacks benchmark/response artifacts: {source_run}"
        )

    source_index = read_json(source_benchmarks / "index.json")
    index_by_case = {
        str(row.get("case_id")): row
        for row in source_index.get("cases", [])
        if isinstance(row, dict) and row.get("case_id")
    }
    case_ids = config["case_ids"]
    system_ids = config["system_ids"]
    missing_cases = [case_id for case_id in case_ids if case_id not in index_by_case]
    if missing_cases:
        raise ValueError(f"Source run is missing selected benchmark(s): {missing_cases}")

    benchmark_dir = run_dir / "benchmarks"
    response_dir = run_dir / "responses"
    benchmark_dir.mkdir(parents=True)
    response_dir.mkdir(parents=True)
    selected_index_rows = []
    for case_id in case_ids:
        row = copy.deepcopy(index_by_case[case_id])
        source = source_benchmarks / row["file"]
        if not source.is_file():
            raise FileNotFoundError(f"Selected benchmark file is missing: {source}")
        shutil.copy2(source, benchmark_dir / row["file"])
        selected_index_rows.append(row)
    write_json(
        benchmark_dir / "index.json",
        {
            "schema_version": "1.0",
            "generated_at": utc_now(),
            "source_run": source_run.name,
            "selection_id": config["selection_id"],
            "cases": selected_index_rows,
        },
    )

    response_hashes: dict[tuple[str, str], str] = {}
    responses: dict[tuple[str, str], dict[str, Any]] = {}
    for case_id, system_id in itertools.product(case_ids, system_ids):
        filename = f"{case_id}__{system_id}.json"
        source = source_responses / filename
        if not source.is_file():
            raise FileNotFoundError(f"Selected response is missing: {source}")
        destination = response_dir / filename
        shutil.copy2(source, destination)
        response_hashes[(case_id, system_id)] = sha256_file(destination)
        responses[(case_id, system_id)] = read_json(destination)

    base_batch_path = prepare_judge_tasks(
        benchmark_dir, run_dir, seed=int(config.get("seed", 20260907))
    )
    base_tasks = read_jsonl(base_batch_path)
    base_mapping = {
        row["task_id"]: row for row in read_json(run_dir / "judge_mapping.json")["mapping"]
    }
    skill = _skill_payload((root / config["evaluation_skill"]).resolve())
    seed = int(config.get("seed", 20260907))
    repeats = int(config["repeats"])

    candidate_ids: dict[tuple[str, str], str] = {}
    for case_id, system_id in itertools.product(case_ids, system_ids):
        candidate_ids[(case_id, system_id)] = "candidate-" + stable_hash(
            {"selection_id": config["selection_id"], "case_id": case_id, "system_id": system_id}
        )[:12]

    tasks: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    expert_tasks: list[dict[str, Any]] = []
    expert_mapping: list[dict[str, Any]] = []
    expert_template: list[dict[str, Any]] = []
    expert_seen: set[str] = set()
    for base_task in base_tasks:
        identity = base_mapping[base_task["task_id"]]
        key = (identity["case_id"], identity["system_id"])
        candidate_id = candidate_ids[key]
        checks = _observable_checks(responses[key])

        if candidate_id not in expert_seen:
            expert_seen.add(candidate_id)
            expert_task = {
                key_name: copy.deepcopy(base_task[key_name])
                for key_name in (
                    "case_id",
                    "question",
                    "reference_answer",
                    "rubric",
                    "tool_efficiency_rubric",
                    "candidate_execution_status",
                    "candidate_execution_error_type",
                    "candidate_execution_error",
                    "candidate_completion_mode",
                    "candidate_native_execution_status",
                    "candidate_native_execution_error_type",
                    "candidate_recovery",
                    "candidate_response",
                    "candidate_score_points",
                    "observable_trajectory",
                    "failure_code_vocabulary",
                )
                if key_name in base_task
            }
            expert_task.update(
                {
                    "candidate_id": candidate_id,
                    "precomputed_observation_checks": checks,
                    "expected_human_output": {
                        "candidate_id": candidate_id,
                        "total_score": "number from 0 to 100",
                        "failure_codes": ["zero or more allowed codes"],
                        "notes": "brief evidence-based rationale",
                    },
                }
            )
            expert_tasks.append(expert_task)
            expert_mapping.append(
                {
                    "candidate_id": candidate_id,
                    "case_id": key[0],
                    "system_id": key[1],
                    "display_name": identity.get("display_name"),
                    "source_response_sha256": response_hashes[key],
                }
            )
            expert_template.append(
                {
                    "candidate_id": candidate_id,
                    "total_score": None,
                    "failure_codes": [],
                    "notes": "",
                }
            )

        for condition in config["judge_conditions"]:
            for repeat_index in range(1, repeats + 1):
                task = copy.deepcopy(base_task)
                task_id = "judge-calibration::" + stable_hash(
                    {
                        "run_id": run_id,
                        "candidate_id": candidate_id,
                        "condition": condition,
                        "repeat": repeat_index,
                    }
                )[:20]
                task["task_id"] = task_id
                task["expected_output"]["task_id"] = task_id
                task["precomputed_observation_checks"] = checks
                if condition == "j1":
                    task["instructions"] += (
                        " Apply the frozen evaluation_skill procedure below before producing the same expected_output schema."
                    )
                    task["evaluation_skill"] = skill
                tasks.append(task)
                mapping.append(
                    {
                        "task_id": task_id,
                        "candidate_id": candidate_id,
                        "case_id": key[0],
                        "candidate_label": task["candidate_label"],
                        "system_id": key[1],
                        "display_name": identity.get("display_name"),
                        "judge_condition": condition,
                        "repeat_index": repeat_index,
                        "source_response_sha256": response_hashes[key],
                    }
                )

    task_to_mapping = {row["task_id"]: row for row in mapping}
    random.Random(seed).shuffle(tasks)
    mapping = [task_to_mapping[task["task_id"]] for task in tasks]
    expert_by_id = {row["candidate_id"]: row for row in expert_mapping}
    random.Random(seed + 1).shuffle(expert_tasks)
    expert_mapping = [expert_by_id[task["candidate_id"]] for task in expert_tasks]

    write_jsonl(run_dir / "judge_batch.jsonl", tasks)
    write_json(run_dir / "judge_mapping.json", {"schema_version": "1.0", "seed": seed, "mapping": mapping})
    write_jsonl(run_dir / "expert_review_batch.jsonl", expert_tasks)
    write_json(run_dir / "expert_review_mapping.json", {"schema_version": "1.0", "mapping": expert_mapping})
    write_jsonl(run_dir / "expert_ratings.template.jsonl", expert_template)
    shutil.copy2(config_path, run_dir / "selection.json")
    skill_snapshot = run_dir / "evaluation_skill" / "SKILL.md"
    skill_snapshot.parent.mkdir(parents=True)
    shutil.copy2((root / config["evaluation_skill"]).resolve(), skill_snapshot)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": CALIBRATION_ID,
        "run_id": run_id,
        "created_at": utc_now(),
        "source_run": source_run.name,
        "selection_id": config["selection_id"],
        "selection_sha256": sha256_file(config_path),
        "evaluation_skill_id": skill["skill_id"],
        "evaluation_skill_sha256": skill["sha256"],
        "conditions": config["judge_conditions"],
        "repeats": repeats,
        "candidate_count": len(expert_tasks),
        "judge_task_count": len(tasks),
        "source_responses": expert_mapping,
        "isolation": {
            "source_run_unchanged": True,
            "candidate_system_identity_excluded_from_judge_task": True,
            "solver_skill_forbidden": True,
            "j0_j1_share_candidates_and_output_schema": True,
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def run_calibration_judges(
    *,
    root: Path,
    run_id: str,
    model: str,
    concurrency: int,
    retries: int,
    timeout_seconds: int,
    force: bool = False,
) -> list[dict[str, Any]]:
    run_dir = (root / "runs" / run_id).resolve()
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Prepared calibration run not found: {run_dir}")
    tasks = read_jsonl(run_dir / "judge_batch.jsonl")
    outputs = run_codex_tasks(
        stage="judge",
        tasks=tasks,
        run_dir=run_dir,
        project_root=root,
        model=model,
        concurrency=concurrency,
        retries=retries,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    write_jsonl(run_dir / "ratings.jsonl", outputs)
    errors = validate_ratings(run_dir)
    if errors:
        raise ValueError("Calibration ratings failed validation: " + "; ".join(errors[:5]))
    return outputs


def _failure_codes(rating: dict[str, Any]) -> set[str]:
    return {
        str(code)
        for step in rating.get("steps", [])
        if isinstance(step, dict)
        for code in step.get("failure_codes", [])
        if isinstance(code, str)
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["rating"]["total_score"]) for row in rows]
    latencies = [
        float(row["record"]["latency_ms"])
        for row in rows
        if isinstance(row.get("record", {}).get("latency_ms"), (int, float))
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["identity"]["candidate_id"]].append(row)
    repeat_differences = []
    code_jaccards = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: item["identity"]["repeat_index"])
        for left, right in itertools.combinations(ordered, 2):
            repeat_differences.append(
                abs(float(left["rating"]["total_score"]) - float(right["rating"]["total_score"]))
            )
            left_codes = _failure_codes(left["rating"])
            right_codes = _failure_codes(right["rating"])
            union = left_codes | right_codes
            code_jaccards.append(len(left_codes & right_codes) / len(union) if union else 1.0)
    return {
        "ratings": len(rows),
        "candidates": len(grouped),
        "mean_score": mean(scores),
        "repeat_mean_absolute_difference": mean(repeat_differences),
        "repeat_median_absolute_difference": median(repeat_differences),
        "repeat_within_5_rate": mean(value <= 5 for value in repeat_differences),
        "repeat_within_10_rate": mean(value <= 10 for value in repeat_differences),
        "failure_code_repeat_jaccard": mean(code_jaccards),
        "mean_latency_ms": mean(latencies) if latencies else None,
    }


def _micro_code_metrics(predicted: list[set[str]], expected: list[set[str]]) -> dict[str, float]:
    tp = sum(len(p & e) for p, e in zip(predicted, expected))
    fp = sum(len(p - e) for p, e in zip(predicted, expected))
    fn = sum(len(e - p) for p, e in zip(predicted, expected))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def build_calibration_analysis(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    mapping_rows = read_json(run_dir / "judge_mapping.json").get("mapping", [])
    mapping = {row["task_id"]: row for row in mapping_rows}
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    record_by_task = {}
    records_dir = run_dir / "codex" / "records" / "judge"
    for path in records_dir.glob("*.json") if records_dir.exists() else []:
        record = read_json(path)
        if record.get("task_id"):
            record_by_task[record["task_id"]] = record
    rows = [
        {
            "identity": mapping[rating["task_id"]],
            "rating": rating,
            "record": record_by_task.get(rating["task_id"], {}),
        }
        for rating in ratings
        if rating.get("task_id") in mapping
    ]
    expected_count = int(manifest["judge_task_count"])
    if len(rows) != expected_count:
        raise ValueError(
            f"Calibration requires {expected_count} valid ratings; found {len(rows)}"
        )
    by_condition = {
        condition: [row for row in rows if row["identity"]["judge_condition"] == condition]
        for condition in manifest["conditions"]
    }
    summaries = {
        condition: _condition_summary(condition_rows)
        for condition, condition_rows in by_condition.items()
    }

    candidate_condition_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        identity = row["identity"]
        candidate_condition_scores[(identity["candidate_id"], identity["judge_condition"])].append(
            float(row["rating"]["total_score"])
        )
    candidate_shifts = []
    for candidate_id in sorted({key[0] for key in candidate_condition_scores}):
        j0 = mean(candidate_condition_scores[(candidate_id, "j0")])
        j1 = mean(candidate_condition_scores[(candidate_id, "j1")])
        identity = next(row["identity"] for row in rows if row["identity"]["candidate_id"] == candidate_id)
        candidate_shifts.append(
            {
                "candidate_id": candidate_id,
                "case_id": identity["case_id"],
                "system_id": identity["system_id"],
                "j0_mean_score": j0,
                "j1_mean_score": j1,
                "j1_minus_j0": j1 - j0,
            }
        )
    system_shifts = {}
    for system_id in sorted({row["system_id"] for row in candidate_shifts}):
        values = [row["j1_minus_j0"] for row in candidate_shifts if row["system_id"] == system_id]
        system_shifts[system_id] = mean(values)

    expert_path = run_dir / "expert_ratings.jsonl"
    expert_rows = read_jsonl(expert_path)
    expert = {
        row.get("candidate_id"): row
        for row in expert_rows
        if isinstance(row.get("total_score"), (int, float)) and not isinstance(row.get("total_score"), bool)
    }
    expert_analysis: dict[str, Any] = {
        "status": "complete" if len(expert) == manifest["candidate_count"] else "pending",
        "rated_candidates": len(expert),
        "required_candidates": manifest["candidate_count"],
    }
    if len(expert) == manifest["candidate_count"]:
        for condition in manifest["conditions"]:
            absolute_errors = []
            predicted_codes = []
            expert_codes = []
            for candidate_id, expert_rating in expert.items():
                predicted = mean(candidate_condition_scores[(candidate_id, condition)])
                absolute_errors.append(abs(predicted - float(expert_rating["total_score"])))
                condition_ratings = [
                    row["rating"]
                    for row in by_condition[condition]
                    if row["identity"]["candidate_id"] == candidate_id
                ]
                for rating in condition_ratings:
                    predicted_codes.append(_failure_codes(rating))
                    expert_codes.append({str(code) for code in expert_rating.get("failure_codes", [])})
            ranking_total = 0
            ranking_matches = 0
            for case_id in sorted({row["case_id"] for row in candidate_shifts}):
                case_candidates = [
                    row["candidate_id"]
                    for row in candidate_shifts
                    if row["case_id"] == case_id
                ]
                for left, right in itertools.combinations(case_candidates, 2):
                    expert_delta = float(expert[left]["total_score"]) - float(expert[right]["total_score"])
                    if abs(expert_delta) <= 1e-9:
                        continue
                    predicted_delta = mean(candidate_condition_scores[(left, condition)]) - mean(
                        candidate_condition_scores[(right, condition)]
                    )
                    ranking_total += 1
                    ranking_matches += (predicted_delta > 0) == (expert_delta > 0)
            expert_analysis[condition] = {
                "mean_absolute_error": mean(absolute_errors),
                "within_5_rate": mean(value <= 5 for value in absolute_errors),
                "within_10_rate": mean(value <= 10 for value in absolute_errors),
                "pairwise_ranking_agreement": (
                    ranking_matches / ranking_total if ranking_total else None
                ),
                "failure_codes_micro": _micro_code_metrics(predicted_codes, expert_codes),
            }

    analysis = {
        "schema_version": "1.0",
        "experiment_id": manifest["experiment_id"],
        "run_id": run_dir.name,
        "candidate_count": manifest["candidate_count"],
        "judge_task_count": expected_count,
        "condition_summaries": summaries,
        "j1_minus_j0": {
            "mean_score_shift": mean(row["j1_minus_j0"] for row in candidate_shifts),
            "candidate_shifts": candidate_shifts,
            "mean_shift_by_hidden_system": system_shifts,
            "system_shift_range": max(system_shifts.values()) - min(system_shifts.values()),
        },
        "expert_calibration": expert_analysis,
        "interpretation_boundary": (
            "Without complete independent expert ratings, this experiment measures Judge stability and score shift, not correctness."
        ),
    }
    return analysis


def write_calibration_analysis(run_dir: Path, analysis: dict[str, Any]) -> Path:
    write_json(run_dir / "calibration_analysis.json", analysis)
    summaries = analysis["condition_summaries"]
    expert = analysis["expert_calibration"]
    lines = [
        "# Evaluation Skill Judge校准结果",
        "",
        f"- 候选回答：{analysis['candidate_count']}",
        f"- Judge任务：{analysis['judge_task_count']}",
        f"- 专家校准：{expert['status']} ({expert['rated_candidates']}/{expert['required_candidates']})",
        "",
        "## 重复评分稳定性",
        "",
        "| 条件 | 平均分 | 重复评分平均绝对差 | ±5分一致率 | failure-code重复Jaccard | 平均时延秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("j0", "j1"):
        row = summaries[condition]
        lines.append(
            f"| {condition.upper()} | {row['mean_score']:.2f} | "
            f"{row['repeat_mean_absolute_difference']:.2f} | "
            f"{row['repeat_within_5_rate']:.1%} | {row['failure_code_repeat_jaccard']:.3f} | "
            f"{row['mean_latency_ms'] / 1000:.1f} |"
            if row["mean_latency_ms"] is not None
            else f"| {condition.upper()} | {row['mean_score']:.2f} | {row['repeat_mean_absolute_difference']:.2f} | {row['repeat_within_5_rate']:.1%} | {row['failure_code_repeat_jaccard']:.3f} | - |"
        )
    shift = analysis["j1_minus_j0"]
    lines.extend(
        [
            "",
            "## J1相对J0",
            "",
            f"- 平均分变化：{shift['mean_score_shift']:+.2f}",
            f"- 隐藏系统间变化范围：{shift['system_shift_range']:.2f}分",
        ]
    )
    if expert["status"] == "complete":
        lines.extend(["", "## 与专家评分比较", "", "| 条件 | MAE | ±5分 | ±10分 | 排序一致率 | failure-code F1 |", "|---|---:|---:|---:|---:|---:|"])
        for condition in ("j0", "j1"):
            row = expert[condition]
            lines.append(
                f"| {condition.upper()} | {row['mean_absolute_error']:.2f} | "
                f"{row['within_5_rate']:.1%} | {row['within_10_rate']:.1%} | "
                f"{row['pairwise_ranking_agreement']:.1%} | "
                f"{row['failure_codes_micro']['f1']:.3f} |"
            )
    else:
        lines.extend(
            [
                "",
                "## 当前结论边界",
                "",
                "尚无完整独立专家评分，因此当前只能比较重复稳定性和评分偏移，不能声称J1更正确。",
            ]
        )
    output = run_dir / "RESULTS_CN.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
