from __future__ import annotations

import copy
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .benchmark import iter_benchmarks, validate_benchmark
from .codex_automation import run_codex_tasks, validate_stage_environment
from .io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
    sha256_tree,
    utc_now,
    write_json,
    write_jsonl,
)
from .judge import prepare_judge_tasks, validate_ratings
from .runner import execute_run, summarize_run_completeness


PILOT_ID = "teacher-distilled-solver-skill-pilot@0.2.0"
PILOT_CONDITIONS = ("c00", "c10")
PRIMARY_OUTCOMES = {
    "best_of_3_task_score",
    "mean_case_best_of_3_task_score_difference",
}


def _required_string(config: dict[str, Any], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"solver Skill pilot config requires non-empty {field}")
    return value.strip()


def _required_string_list(config: dict[str, Any], field: str) -> list[str]:
    value = config.get(field)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"solver Skill pilot config requires non-empty string array {field}")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"solver Skill pilot config contains duplicate {field}")
    return normalized


def _skill_dir(root: Path, skill_version: str) -> Path:
    if "@" not in skill_version:
        raise ValueError("solver Skill version must use name@version")
    skill_id, version = skill_version.split("@", 1)
    path = (root / "skills" / skill_id / f"v{version}").resolve()
    if not (path / "SKILL.md").is_file():
        raise FileNotFoundError(f"Solver Skill artifact not found: {path}")
    return path


def load_solver_skill_pilot_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("solver Skill pilot config schema_version must be 1.0")
    _required_string(config, "selection_id")
    _required_string(config, "source_benchmarks_dir")
    _required_string(config, "base_systems_config")
    _required_string(config, "base_system_id")
    skill_version = _required_string(config, "skill_version")
    case_ids = _required_string_list(config, "case_ids")
    _required_string_list(config, "required_observable_tools")
    primary_outcome = config.get("primary_outcome")
    if primary_outcome not in PRIMARY_OUTCOMES:
        raise ValueError(
            "solver Skill pilot primary_outcome must be one of "
            f"{sorted(PRIMARY_OUTCOMES)}"
        )
    if primary_outcome == "mean_case_best_of_3_task_score_difference" and len(case_ids) < 2:
        raise ValueError("mean-case best-of-3 requires at least two selected cases")
    conditions = _required_string_list(config, "conditions")
    if tuple(conditions) != PILOT_CONDITIONS:
        raise ValueError("solver Skill pilot conditions must be ordered as c00, c10")
    repeats = config.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats != 3:
        raise ValueError("solver Skill best-of-3 experiments require exactly 3 repeats")
    max_attempts = config.get("max_collection_attempts", 4)
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_collection_attempts must be an integer >= 1")
    source_dir = (root / config["source_benchmarks_dir"]).resolve()
    if not (source_dir / "index.json").is_file():
        raise FileNotFoundError(f"Source benchmark index not found: {source_dir / 'index.json'}")
    systems_path = (root / config["base_systems_config"]).resolve()
    if not systems_path.is_file():
        raise FileNotFoundError(f"Base systems config not found: {systems_path}")
    skill_dir = _skill_dir(root, skill_version)
    expected_skill_sha256 = config.get("expected_skill_sha256")
    if expected_skill_sha256 is not None:
        if not isinstance(expected_skill_sha256, str) or not expected_skill_sha256.strip():
            raise ValueError("expected_skill_sha256 must be a non-empty string")
        actual_skill_sha256 = sha256_tree(skill_dir)
        if actual_skill_sha256 != expected_skill_sha256.strip():
            raise ValueError(
                "Frozen Solver Skill hash mismatch: "
                f"expected {expected_skill_sha256.strip()}, found {actual_skill_sha256}"
            )
    return config


def _clean_base_system(base: dict[str, Any]) -> dict[str, Any]:
    system = copy.deepcopy(base)
    for field in (
        "id",
        "display_name",
        "skill_version",
        "skill_delivery",
        "skill_prompt",
        "skill_artifact_sha256",
        "skill_path",
        "remote_skill_id",
        "rag_version",
        "rag_version_env",
        "rag_policy",
        "adaptive_rag",
    ):
        system.pop(field, None)
    system["tools_enabled"] = True
    system["require_observable_tool_call"] = True
    system["rag_enabled"] = False
    return system


def _pilot_systems(
    *,
    base: dict[str, Any],
    skill_version: str,
    repeats: int,
    required_observable_tools: list[str],
) -> list[dict[str, Any]]:
    systems: list[dict[str, Any]] = []
    for repeat_index in range(1, repeats + 1):
        for condition in PILOT_CONDITIONS:
            system = _clean_base_system(base)
            system.update(
                {
                    "id": f"{condition}-r{repeat_index}",
                    "display_name": (
                        f"C00 Tools (repeat {repeat_index})"
                        if condition == "c00"
                        else f"C10 Tools + Teacher-Distilled Skill (repeat {repeat_index})"
                    ),
                    "pilot_condition": condition,
                    "repeat_index": repeat_index,
                    "required_observable_tools": list(required_observable_tools),
                }
            )
            if condition == "c10":
                system.update(
                    {
                        "skill_version": skill_version,
                        "skill_delivery": "prompt",
                        "skill_path": "solver_skill",
                    }
                )
            systems.append(system)
    return systems


def prepare_solver_skill_pilot(
    *, root: Path, config_path: Path, run_id: str
) -> dict[str, Any]:
    config = load_solver_skill_pilot_config(root, config_path)
    run_dir = (root / "runs" / run_id).resolve()
    runs_root = (root / "runs").resolve()
    if not run_dir.is_relative_to(runs_root):
        raise ValueError(f"Invalid run ID outside runs directory: {run_id}")
    if run_dir.exists():
        raise FileExistsError(
            f"Solver Skill pilot run already exists and will not be overwritten: {run_dir}"
        )

    source_dir = (root / config["source_benchmarks_dir"]).resolve()
    source_index = read_json(source_dir / "index.json")
    index_by_case = {
        str(row.get("case_id")): row
        for row in source_index.get("cases", [])
        if isinstance(row, dict) and row.get("case_id")
    }
    case_ids = config["case_ids"]
    missing = [case_id for case_id in case_ids if case_id not in index_by_case]
    if missing:
        raise ValueError(f"Source benchmark set is missing selected case(s): {missing}")

    benchmark_dir = run_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    selected_rows = []
    for case_id in case_ids:
        row = copy.deepcopy(index_by_case[case_id])
        source_path = source_dir / row["file"]
        benchmark = read_json(source_path)
        validate_benchmark(benchmark)
        shutil.copy2(source_path, benchmark_dir / row["file"])
        selected_rows.append(row)
    write_json(
        benchmark_dir / "index.json",
        {
            "schema_version": source_index.get("schema_version", "1.0"),
            "generated_at": utc_now(),
            "selection": {
                "selection_id": config["selection_id"],
                "case_ids": case_ids,
                "source_benchmarks_dir": str(source_dir),
                "source_index_sha256": sha256_file(source_dir / "index.json"),
            },
            "cases": selected_rows,
        },
    )

    base_path = (root / config["base_systems_config"]).resolve()
    base_config = read_json(base_path)
    base_by_id = {
        row.get("id"): row
        for row in base_config.get("systems", [])
        if isinstance(row, dict) and row.get("id")
    }
    base_system_id = config["base_system_id"]
    if base_system_id not in base_by_id:
        raise ValueError(f"Unknown base system ID: {base_system_id}")
    base_system = base_by_id[base_system_id]
    if not base_system.get("tools_enabled") or base_system.get("rag_enabled"):
        raise ValueError("Base system must enable Tools and disable RAG")
    systems = _pilot_systems(
        base=base_system,
        skill_version=config["skill_version"],
        repeats=int(config["repeats"]),
        required_observable_tools=config["required_observable_tools"],
    )
    systems_snapshot = {
        "schema_version": "1.0",
        "shared_system_prompt": base_config["shared_system_prompt"],
        "generation": copy.deepcopy(base_config["generation"]),
        "context_recovery": copy.deepcopy(base_config.get("context_recovery")),
        "systems": systems,
    }
    if config.get("generation_overrides"):
        systems_snapshot["generation"].update(copy.deepcopy(config["generation_overrides"]))
    write_json(run_dir / "systems.json", systems_snapshot)

    system_ids = [system["id"] for system in systems]
    comparisons = [
        {
            "id": f"c10_minus_c00_r{repeat_index}",
            "label": f"Teacher-distilled Solver Skill effect, repeat {repeat_index}",
            "baseline_system": f"c00-r{repeat_index}",
            "candidate_system": f"c10-r{repeat_index}",
        }
        for repeat_index in range(1, int(config["repeats"]) + 1)
    ]
    write_json(
        run_dir / "evaluation_profile.json",
        {
            "schema_version": "1.0",
            "profile_id": "solver_skill_pilot_c00_c10",
            "benchmark_set": config.get("source_benchmark_set", "d1_d6"),
            "description": "Development-only C00/C10 test of one frozen, prompt-delivered Solver Skill",
            "system_ids": system_ids,
            "teachers": [],
            "comparisons": comparisons,
        },
    )

    skill_dir = _skill_dir(root, config["skill_version"])
    skill_snapshot = run_dir / "solver_skill"
    shutil.copytree(skill_dir, skill_snapshot)
    shutil.copy2(config_path, run_dir / "selection.json")
    pilot_manifest = {
        "schema_version": "1.0",
        "experiment_id": config.get("experiment_id", PILOT_ID),
        "study_phase": config.get("study_phase", "development"),
        "parent_run_id": config.get("parent_run_id"),
        "selection_rationale": config.get("selection_rationale"),
        "run_id": run_id,
        "created_at": utc_now(),
        "selection_id": config["selection_id"],
        "selection_sha256": sha256_file(config_path),
        "source_benchmark_set": config.get("source_benchmark_set", "d1_d6"),
        "case_ids": case_ids,
        "conditions": list(PILOT_CONDITIONS),
        "repeats": int(config["repeats"]),
        "primary_outcome": config["primary_outcome"],
        "required_observable_tools": config["required_observable_tools"],
        "system_ids": system_ids,
        "base_system_id": base_system_id,
        "base_systems_sha256": sha256_file(base_path),
        "solver_skill_version": config["skill_version"],
        "solver_skill_sha256": sha256_tree(skill_dir),
        "skill_delivery": "local_prompt",
        "max_collection_attempts": int(config.get("max_collection_attempts", 4)),
        "expected_solver_episodes": len(case_ids) * len(system_ids),
        "isolation": {
            "same_model_env": True,
            "same_tools": True,
            "rag_disabled": True,
            "only_c10_receives_solver_skill": True,
            "judge_is_frozen_and_shared": True,
            "teacher_not_called_during_solver_execution": True,
            "hard_executor_not_tested": True,
            "self_evolution_not_tested": True,
        },
    }
    write_json(run_dir / "pilot_manifest.json", pilot_manifest)
    return pilot_manifest


def _prepared_paths(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = (root / "runs" / run_id).resolve()
    manifest_path = run_dir / "pilot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared Solver Skill pilot not found: {run_dir}")
    return run_dir, read_json(manifest_path)


def run_solver_skill_systems(
    *, root: Path, run_id: str, concurrency: int
) -> dict[str, Any]:
    run_dir, pilot = _prepared_paths(root, run_id)
    benchmark_dir = run_dir / "benchmarks"
    systems_path = run_dir / "systems.json"
    counts = execute_run(
        benchmarks_dir=benchmark_dir,
        systems_path=systems_path,
        run_dir=run_dir,
        selected_system_ids=pilot["system_ids"],
        evaluation_profile="solver_skill_pilot_c00_c10",
        system_concurrency=concurrency,
        max_collection_attempts=int(pilot.get("max_collection_attempts", 4)),
    )
    completeness = summarize_run_completeness(
        benchmarks_dir=benchmark_dir,
        run_dir=run_dir,
        system_ids=pilot["system_ids"],
    )
    return {"counts": counts, "completeness": completeness}


def run_solver_skill_judges(
    *,
    root: Path,
    run_id: str,
    model: str,
    concurrency: int,
    retries: int,
    timeout_seconds: int,
    seed: int,
    force: bool = False,
) -> list[dict[str, Any]]:
    run_dir, pilot = _prepared_paths(root, run_id)
    completeness = summarize_run_completeness(
        benchmarks_dir=run_dir / "benchmarks",
        run_dir=run_dir,
        system_ids=pilot["system_ids"],
    )
    if completeness["incomplete"]:
        examples = ", ".join(
            f"{row['case_id']}/{row['system_id']}={row['status']}"
            for row in completeness["items"][:6]
        )
        raise ValueError(
            "Solver Skill pilot system stage is incomplete: "
            f"{completeness['success']}/{completeness['expected']}; {examples}"
        )
    batch_path = prepare_judge_tasks(
        run_dir / "benchmarks", run_dir, seed=seed
    )
    tasks = read_jsonl(batch_path)
    validate_stage_environment("judge", tasks)
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
        raise ValueError("Solver Skill pilot ratings failed validation: " + "; ".join(errors[:5]))
    return outputs


def _failure_codes(rating: dict[str, Any]) -> set[str]:
    return {
        str(code)
        for step in rating.get("steps", [])
        if isinstance(step, dict)
        for code in step.get("failure_codes", [])
        if isinstance(code, str)
    }


def _usage_total(response: dict[str, Any]) -> float | None:
    usage = response.get("usage") or {}
    for field in ("total_tokens", "total_token_count"):
        value = usage.get(field) if isinstance(usage, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def build_solver_skill_analysis(run_dir: Path) -> dict[str, Any]:
    pilot = read_json(run_dir / "pilot_manifest.json")
    mapping_rows = read_json(run_dir / "judge_mapping.json").get("mapping", [])
    mapping = {row["task_id"]: row for row in mapping_rows if row.get("task_id")}
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    responses = {
        (row.get("case_id"), row.get("system_id")): row
        for row in (
            read_json(path) for path in sorted((run_dir / "responses").glob("*.json"))
        )
    }
    systems = {
        row["id"]: row
        for row in read_json(run_dir / "systems.json").get("systems", [])
    }
    expected = len(pilot["case_ids"]) * len(pilot["system_ids"])
    if len(ratings) != expected:
        raise ValueError(f"Solver Skill pilot requires {expected} ratings; found {len(ratings)}")

    rows: list[dict[str, Any]] = []
    for rating in ratings:
        identity = mapping.get(rating.get("task_id"))
        if not identity:
            raise ValueError(f"Rating lacks Judge mapping: {rating.get('task_id')}")
        system_id = identity["system_id"]
        system = systems.get(system_id)
        if not system:
            raise ValueError(f"Unknown pilot system in Judge mapping: {system_id}")
        case_id = identity["case_id"]
        response = responses.get((case_id, system_id))
        if not response:
            raise ValueError(f"Missing response for {case_id}/{system_id}")
        condition = system["pilot_condition"]
        expected_hash = pilot["solver_skill_sha256"] if condition == "c10" else None
        if response.get("skill_artifact_sha256") != expected_hash:
            raise ValueError(f"Skill artifact mismatch for {case_id}/{system_id}")
        trajectory_summary = (response.get("trajectory") or {}).get("summary") or {}
        path_class = (rating.get("trajectory_analysis") or {}).get("path_classification")
        failure_codes = _failure_codes(rating)
        rows.append(
            {
                "case_id": case_id,
                "condition": condition,
                "repeat_index": int(system["repeat_index"]),
                "system_id": system_id,
                "score": float(rating["total_score"]),
                "tool_efficiency_score": (
                    float(rating["tool_efficiency_score"])
                    if isinstance(rating.get("tool_efficiency_score"), (int, float))
                    and not isinstance(rating.get("tool_efficiency_score"), bool)
                    else None
                ),
                "status": response.get("status", "missing"),
                "native_status": response.get("native_status", response.get("status", "missing")),
                "completion_mode": response.get("completion_mode", "missing"),
                "tool_calls": float(trajectory_summary.get("tool_interactions", 0) or 0),
                "tool_errors": float(trajectory_summary.get("tool_errors", 0) or 0),
                "latency_ms": (
                    float(response["latency_ms"])
                    if isinstance(response.get("latency_ms"), (int, float))
                    else None
                ),
                "total_tokens": _usage_total(response),
                "valid_path": path_class in {"golden_aligned", "valid_alternative"},
                "tool_argument_failure": "TOOL_ARGUMENT" in failure_codes,
                "failure_codes": sorted(failure_codes),
            }
        )

    if {response.get("model_id") for response in responses.values()} != {
        next(iter(responses.values())).get("model_id")
    }:
        raise ValueError("C00/C10 responses did not use one common model ID")
    if any(response.get("rag_enabled") for response in responses.values()):
        raise ValueError("Solver Skill pilot requires RAG disabled in every response")

    summaries: dict[str, Any] = {}
    for condition in PILOT_CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        best = max(selected, key=lambda row: (row["score"], -row["repeat_index"]))
        summaries[condition] = {
            "episodes": len(selected),
            "best_score": best["score"],
            "best_case_id": best["case_id"],
            "best_system_id": best["system_id"],
            "best_repeat_index": best["repeat_index"],
            "mean_score": mean(row["score"] for row in selected),
            "mean_tool_efficiency_score": _mean_optional(
                [row["tool_efficiency_score"] for row in selected]
            ),
            "final_completion_rate": mean(row["status"] == "success" for row in selected),
            "native_completion_rate": mean(row["native_status"] == "success" for row in selected),
            "valid_path_rate": mean(row["valid_path"] for row in selected),
            "tool_argument_failure_rate": mean(row["tool_argument_failure"] for row in selected),
            "mean_tool_calls": mean(row["tool_calls"] for row in selected),
            "mean_tool_errors": mean(row["tool_errors"] for row in selected),
            "mean_latency_ms": _mean_optional([row["latency_ms"] for row in selected]),
            "mean_total_tokens": _mean_optional([row["total_tokens"] for row in selected]),
        }

    lookup = {
        (row["case_id"], row["condition"], row["repeat_index"]): row
        for row in rows
    }
    paired_rows = []
    for case_id in pilot["case_ids"]:
        for repeat_index in range(1, int(pilot["repeats"]) + 1):
            c00 = lookup[(case_id, "c00", repeat_index)]
            c10 = lookup[(case_id, "c10", repeat_index)]
            paired_rows.append(
                {
                    "case_id": case_id,
                    "repeat_index": repeat_index,
                    "c00_score": c00["score"],
                    "c10_score": c10["score"],
                    "c10_minus_c00": c10["score"] - c00["score"],
                    "c00_tool_argument_failure": c00["tool_argument_failure"],
                    "c10_tool_argument_failure": c10["tool_argument_failure"],
                }
            )
    gains = [row["c10_minus_c00"] for row in paired_rows]
    case_effects = []
    case_best_of_3 = []
    for case_id in pilot["case_ids"]:
        case_gains = [row["c10_minus_c00"] for row in paired_rows if row["case_id"] == case_id]
        c00_rows = [
            row for row in rows if row["case_id"] == case_id and row["condition"] == "c00"
        ]
        c10_rows = [
            row for row in rows if row["case_id"] == case_id and row["condition"] == "c10"
        ]
        c00_case_best = max(c00_rows, key=lambda row: (row["score"], -row["repeat_index"]))
        c10_case_best = max(c10_rows, key=lambda row: (row["score"], -row["repeat_index"]))
        best_difference = c10_case_best["score"] - c00_case_best["score"]
        case_best_of_3.append(
            {
                "case_id": case_id,
                "c00_best_score": c00_case_best["score"],
                "c00_best_system_id": c00_case_best["system_id"],
                "c00_best_repeat_index": c00_case_best["repeat_index"],
                "c10_best_score": c10_case_best["score"],
                "c10_best_system_id": c10_case_best["system_id"],
                "c10_best_repeat_index": c10_case_best["repeat_index"],
                "c10_minus_c00": best_difference,
            }
        )
        case_effects.append(
            {
                "case_id": case_id,
                "repeats": len(case_gains),
                "mean_effect": mean(case_gains),
                "effect_sd": stdev(case_gains) if len(case_gains) > 1 else 0.0,
                "minimum_effect": min(case_gains),
                "maximum_effect": max(case_gains),
                "best_of_3_effect": best_difference,
            }
        )

    primary_metric = pilot.get("primary_outcome", "best_of_3_task_score")
    case_best_gains = [row["c10_minus_c00"] for row in case_best_of_3]
    primary_outcome = {
        "metric": primary_metric,
        "sampling_budget_per_condition_per_case": int(pilot["repeats"]),
        "case_count": len(case_best_of_3),
        "case_results": case_best_of_3,
        "c00_mean_case_best_score": mean(row["c00_best_score"] for row in case_best_of_3),
        "c10_mean_case_best_score": mean(row["c10_best_score"] for row in case_best_of_3),
        "mean_case_best_difference": mean(case_best_gains),
        "case_wins": sum(value > 0 for value in case_best_gains),
        "case_ties": sum(value == 0 for value in case_best_gains),
        "case_losses": sum(value < 0 for value in case_best_gains),
    }
    if len(case_best_of_3) == 1:
        only_case = case_best_of_3[0]
        primary_outcome.update(
            {
                "sampling_budget_per_condition": int(pilot["repeats"]),
                "c00_best_score": only_case["c00_best_score"],
                "c00_best_system_id": only_case["c00_best_system_id"],
                "c10_best_score": only_case["c10_best_score"],
                "c10_best_system_id": only_case["c10_best_system_id"],
                "c10_minus_c00": only_case["c10_minus_c00"],
            }
        )

    return {
        "schema_version": "1.0",
        "experiment_id": pilot["experiment_id"],
        "study_phase": pilot.get("study_phase", "development"),
        "parent_run_id": pilot.get("parent_run_id"),
        "run_id": run_dir.name,
        "case_ids": pilot["case_ids"],
        "repeats": pilot["repeats"],
        "solver_skill_version": pilot["solver_skill_version"],
        "solver_skill_sha256": pilot["solver_skill_sha256"],
        "condition_summaries": summaries,
        "primary_outcome": primary_outcome,
        "paired_effect": {
            "comparison": "c10_minus_c00",
            "mean_effect": mean(gains),
            "wins": sum(value > 0 for value in gains),
            "ties": sum(value == 0 for value in gains),
            "losses": sum(value < 0 for value in gains),
            "paired_episodes": paired_rows,
            "case_effects": case_effects,
        },
        "episode_results": rows,
        "interpretation_boundary": (
            "This pre-registered matching-family validation tests transfer of one frozen, "
            "prompt-delivered seed Skill to cases not used to write it. It does not test the "
            "hard executor, automatic Skill evolution, or held-out D7 generalization."
            if pilot.get("study_phase") == "matching_family_validation"
            else "This development-only pilot tests prompt-delivered seed Skill content. "
            "It does not test the hard executor, automatic Skill evolution, or held-out generalization."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: ";".join(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def write_solver_skill_analysis(run_dir: Path, analysis: dict[str, Any]) -> Path:
    write_json(run_dir / "solver_skill_analysis.json", analysis)
    _write_csv(run_dir / "solver_skill_episode_results.csv", analysis["episode_results"])
    _write_csv(
        run_dir / "solver_skill_paired_effects.csv",
        analysis["paired_effect"]["paired_episodes"],
    )
    _write_csv(
        run_dir / "solver_skill_case_best_of_3.csv",
        analysis["primary_outcome"]["case_results"],
    )
    summaries = analysis["condition_summaries"]
    primary = analysis["primary_outcome"]
    effect = analysis["paired_effect"]
    is_validation = (
        primary.get("metric") == "mean_case_best_of_3_task_score_difference"
    )
    lines = [
        (
            "# 教师蒸馏短Solver Skill同族验证"
            if is_validation
            else "# 教师蒸馏短Solver Skill开发试点"
        ),
        "",
        f"- Skill：`{analysis['solver_skill_version']}`",
        f"- Cases：{', '.join(f'`{case_id}`' for case_id in analysis['case_ids'])}",
        f"- 每条件每题重复：{analysis['repeats']}",
        "- RAG：关闭",
        "- Judge：两个条件共用同一冻结Judge",
        (
            "- 主指标：逐题计算两条件best-of-3差值，再对预注册题目取平均"
            if is_validation
            else "- 主指标：每个条件固定3次有效执行中的最高任务分（best-of-3）"
        ),
        "",
        "## 条件汇总",
        "",
        "| 条件 | Episode | 最高任务分 | 最佳Episode | 平均任务分（诊断） | 平均工具效率分 | 最终完成率 | 原生完成率 | 有效路径率 | Tool参数错误率 | 平均工具调用 | 平均时延秒 | 平均tokens |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in PILOT_CONDITIONS:
        row = summaries[condition]
        latency = "-" if row["mean_latency_ms"] is None else f"{row['mean_latency_ms'] / 1000:.1f}"
        tokens = "-" if row["mean_total_tokens"] is None else f"{row['mean_total_tokens']:.0f}"
        efficiency = (
            "-"
            if row["mean_tool_efficiency_score"] is None
            else f"{row['mean_tool_efficiency_score']:.2f}"
        )
        lines.append(
            f"| {condition.upper()} | {row['episodes']} | {row['best_score']:.2f} | "
            f"{row['best_case_id']}/{row['best_system_id']} | {row['mean_score']:.2f} | "
            f"{efficiency} | "
            f"{row['final_completion_rate']:.1%} | {row['native_completion_rate']:.1%} | "
            f"{row['valid_path_rate']:.1%} | {row['tool_argument_failure_rate']:.1%} | "
            f"{row['mean_tool_calls']:.1f} | {latency} | {tokens} |"
        )
    lines.extend(["", "## 主结果：逐题best-of-3", ""])
    if is_validation:
        lines.extend(
            [
                f"- C00逐题最高分均值：{primary['c00_mean_case_best_score']:.2f}。",
                f"- C10逐题最高分均值：{primary['c10_mean_case_best_score']:.2f}。",
                f"- 平均逐题增益：{primary['mean_case_best_difference']:+.2f}。",
                f"- Case胜/平/负：{primary['case_wins']}/{primary['case_ties']}/{primary['case_losses']}。",
            ]
        )
    else:
        lines.extend(
            [
                f"- C00最高分：{primary['c00_best_score']:.2f}（{primary['c00_best_system_id']}）。",
                f"- C10最高分：{primary['c10_best_score']:.2f}（{primary['c10_best_system_id']}）。",
                f"- C10相对C00：{primary['c10_minus_c00']:+.2f}。",
            ]
        )
    lines.extend(
        [
            "",
            "| Case | C00最高分 | C00 Episode | C10最高分 | C10 Episode | C10-C00 |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for row in primary["case_results"]:
        lines.append(
            f"| {row['case_id']} | {row['c00_best_score']:.2f} | {row['c00_best_system_id']} | "
            f"{row['c10_best_score']:.2f} | {row['c10_best_system_id']} | "
            f"{row['c10_minus_c00']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## 配对均值诊断",
            "",
            f"- 配对平均分变化：{effect['mean_effect']:+.2f}。",
            f"- Episode胜/平/负：{effect['wins']}/{effect['ties']}/{effect['losses']}。",
            "",
            "| Case | 平均变化 | 重复间标准差 | 最小 | 最大 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in effect["case_effects"]:
        lines.append(
            f"| {row['case_id']} | {row['mean_effect']:+.2f} | {row['effect_sd']:.2f} | "
            f"{row['minimum_effect']:+.2f} | {row['maximum_effect']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            (
                "这是对冻结Skill的同一任务族验证；验证题未用于编写Skill，但仍属于D6_6c。"
                if is_validation
                else "这是所选开发题上的种子Skill内容试点，只能判断是否值得继续实现C01/C11。"
            ),
            "它不证明硬执行器有效，不证明Skill已经自动进化，也不证明D7泛化。",
        ]
    )
    output = run_dir / "RESULTS_CN.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
