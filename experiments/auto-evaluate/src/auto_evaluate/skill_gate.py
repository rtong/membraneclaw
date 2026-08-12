from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl


DEFAULT_CONFIG = {
    "baseline_system": "environment",
    "candidate_system": "environment-skill",
    "minimum_mean_gain": 0.1,
    "require_each_case_gain": True,
    "forbidden_candidate_failure_codes": ["TOOL_ARGUMENT", "PARAMETER_EXTRACTION"],
}


def evaluate_skill_gate(
    run_dir: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare Environment-Skill with Environment inside one judged run."""
    settings = {**DEFAULT_CONFIG, **(config or {})}
    mapping_path = run_dir / "judge_mapping.json"
    ratings_path = run_dir / "ratings.jsonl"
    if not mapping_path.exists() or not ratings_path.exists():
        return {
            "status": "unavailable",
            "passed": False,
            "reason": "judge_mapping.json or ratings.jsonl is missing",
            "config": settings,
            "cases": [],
        }

    mapping = {
        row["task_id"]: row
        for row in read_json(mapping_path).get("mapping", [])
        if row.get("task_id")
    }
    scores: dict[str, dict[str, float]] = {}
    candidate_failures: Counter[str] = Counter()
    for rating in read_jsonl(ratings_path):
        identity = mapping.get(rating.get("task_id"), {})
        system_id = identity.get("system_id")
        case_id = rating.get("case_id") or identity.get("case_id")
        if not system_id or not case_id:
            continue
        scores.setdefault(case_id, {})[system_id] = float(rating.get("total_score", 0))
        if system_id == settings["candidate_system"]:
            for step in rating.get("steps", []):
                candidate_failures.update(step.get("failure_codes", []))

    baseline = settings["baseline_system"]
    candidate = settings["candidate_system"]
    manifest_path = run_dir / "manifest.json"
    manifest_cases = (
        read_json(manifest_path).get("benchmark_cases", []) if manifest_path.exists() else []
    )
    expected_cases = sorted(
        set(manifest_cases)
        or {
            case_id
            for case_id, values in scores.items()
            if baseline in values or candidate in values
        }
    )
    comparable = sorted(
        case_id
        for case_id, values in scores.items()
        if baseline in values and candidate in values
    )
    missing_scores = [
        {
            "case_id": case_id,
            "missing_systems": [
                system_id
                for system_id in (baseline, candidate)
                if system_id not in scores.get(case_id, {})
            ],
        }
        for case_id in expected_cases
        if baseline not in scores.get(case_id, {}) or candidate not in scores.get(case_id, {})
    ]
    if missing_scores:
        return {
            "status": "unavailable",
            "passed": False,
            "reason": "one or more benchmark cases are missing baseline or candidate ratings",
            "config": settings,
            "missing_scores": missing_scores,
            "cases": [],
        }
    if not comparable:
        return {
            "status": "unavailable",
            "passed": False,
            "reason": f"no cases contain both {baseline} and {candidate}",
            "config": settings,
            "cases": [],
        }

    case_rows = []
    for case_id in comparable:
        base_score = scores[case_id][baseline]
        candidate_score = scores[case_id][candidate]
        gain = candidate_score - base_score
        case_rows.append(
            {
                "case_id": case_id,
                "baseline_score": base_score,
                "candidate_score": candidate_score,
                "gain": gain,
                "passed": gain > 0,
            }
        )

    baseline_mean = sum(row["baseline_score"] for row in case_rows) / len(case_rows)
    candidate_mean = sum(row["candidate_score"] for row in case_rows) / len(case_rows)
    mean_gain = candidate_mean - baseline_mean
    required_codes = settings.get("forbidden_candidate_failure_codes", [])
    forbidden_found = {code: candidate_failures[code] for code in required_codes if candidate_failures[code]}
    each_case_passed = all(row["passed"] for row in case_rows)
    passed = mean_gain >= float(settings["minimum_mean_gain"])
    if settings.get("require_each_case_gain", True):
        passed = passed and each_case_passed
    passed = passed and not forbidden_found

    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "config": settings,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_gain": mean_gain,
        "each_case_passed": each_case_passed,
        "forbidden_failure_codes_found": forbidden_found,
        "cases": case_rows,
    }
