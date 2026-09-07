from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
from typing import Any

from .io_utils import read_json, read_jsonl
from .routing_metrics import binary_route_metrics


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 5000,
    seed: int = 20260824,
) -> list[float] | None:
    """Deterministic percentile bootstrap interval for paired case-level gains."""
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return [lower, upper]


def _load_mapping(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "judge_mapping.json"
    payload = read_json(path) if path.exists() else {}
    return {
        row["task_id"]: row
        for row in payload.get("mapping", [])
        if isinstance(row, dict) and row.get("task_id")
    }


def _rating_lookup(
    ratings: list[dict[str, Any]], mapping: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for rating in ratings:
        identity = mapping.get(rating.get("task_id"), {})
        case_id = rating.get("case_id") or identity.get("case_id")
        system_id = identity.get("system_id") or rating.get("system_id")
        if case_id and system_id:
            lookup[(str(case_id), str(system_id))] = rating
    return lookup


def _response_lookup(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    responses_dir = run_dir / "responses"
    if not responses_dir.exists():
        return lookup
    for path in sorted(responses_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        row = read_json(path)
        case_id = row.get("case_id")
        system_id = row.get("system_id")
        if case_id and system_id:
            lookup[(str(case_id), str(system_id))] = row
    return lookup


def _benchmark_metadata(run_dir: Path) -> tuple[dict[str, dict[int, str]], dict[str, dict[str, Any]]]:
    labels: dict[str, dict[int, str]] = {}
    views: dict[str, dict[str, Any]] = {}
    benchmarks_dir = run_dir / "benchmarks"
    if not benchmarks_dir.exists():
        return labels, views
    for path in sorted(benchmarks_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        benchmark = read_json(path)
        case_id = benchmark.get("case_id") or path.stem
        labels[str(case_id)] = {
            int(step["step_id"]): str(step.get("step_label") or f"S{step['step_id']}")
            for step in (benchmark.get("rubric") or {}).get("steps", [])
            if step.get("step_id") is not None
        }
        view = benchmark.get("benchmark_view")
        if isinstance(view, dict):
            views[str(case_id)] = view
    return labels, views


def _step_rewards(rating: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not rating:
        return {}
    return {
        int(step["step_id"]): {
            "score": float(step.get("score", 0)),
            "maximum": float(step.get("max_score", 0)),
            "failure_codes": list(step.get("failure_codes") or []),
        }
        for step in rating.get("steps", [])
        if step.get("step_id") is not None
    }


def _tool_calls(response: dict[str, Any] | None) -> int:
    summary = ((response or {}).get("trajectory") or {}).get("summary") or {}
    return int(summary.get("tool_interactions", 0) or 0)


def build_reward_analysis(run_dir: Path) -> dict[str, Any]:
    """Build paired, step-level rewards and adaptive-RAG counterfactual metrics."""
    profile_path = run_dir / "evaluation_profile.json"
    profile = read_json(profile_path) if profile_path.exists() else {}
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    mapping = _load_mapping(run_dir)
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    rating_by_key = _rating_lookup(ratings, mapping)
    response_by_key = _response_lookup(run_dir)
    step_labels, benchmark_views = _benchmark_metadata(run_dir)

    case_ids = list(manifest.get("benchmark_cases") or [])
    if not case_ids:
        case_ids = sorted({case_id for case_id, _ in set(rating_by_key) | set(response_by_key)})
    system_ids = list(profile.get("system_ids") or [])
    if not system_ids:
        system_ids = [row.get("id") for row in manifest.get("systems", []) if row.get("id")]
    comparisons = list(profile.get("comparisons") or [])

    system_acc: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scores": [],
            "efficiency": [],
            "expected": 0,
            "completed": 0,
            "native_completed": 0,
            "tool_calls": [],
            "latencies": [],
            "rag_activations": [],
            "statuses": Counter(),
            "completion_modes": Counter(),
            "errors": Counter(),
            "native_errors": Counter(),
        }
    )
    comparison_acc: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"gains": [], "step_gains": defaultdict(list), "completion_gains": []}
    )
    case_rows: list[dict[str, Any]] = []

    for case_id in case_ids:
        rewards: dict[str, Any] = {}
        for system_id in system_ids:
            rating = rating_by_key.get((case_id, system_id))
            response = response_by_key.get((case_id, system_id))
            score = float(rating["total_score"]) if rating and isinstance(rating.get("total_score"), (int, float)) else None
            efficiency = rating.get("tool_efficiency_score") if rating else None
            completed = bool(response and response.get("status") == "success")
            native_status = (response or {}).get(
                "native_status", (response or {}).get("status", "missing")
            )
            native_completed = native_status == "success"
            calls = _tool_calls(response)
            rewards[system_id] = {
                "task_reward": score,
                "tool_efficiency_reward": float(efficiency) if isinstance(efficiency, (int, float)) else None,
                "completed": completed,
                "status": (response or {}).get("status", "missing"),
                "error_type": (response or {}).get("error_type"),
                "completion_mode": (response or {}).get("completion_mode"),
                "native_status": native_status,
                "native_error_type": (response or {}).get("native_error_type"),
                "tool_calls": calls,
                "steps": _step_rewards(rating),
                "routing": (response or {}).get("routing"),
            }
            acc = system_acc[system_id]
            acc["expected"] += 1
            acc["completed"] += int(completed)
            acc["native_completed"] += int(native_completed)
            acc["tool_calls"].append(float(calls))
            acc["statuses"][(response or {}).get("status", "missing")] += 1
            acc["completion_modes"][(response or {}).get("completion_mode", "missing")] += 1
            if response and isinstance(response.get("latency_ms"), (int, float)):
                acc["latencies"].append(float(response["latency_ms"]))
            if response:
                acc["rag_activations"].append(float(bool(response.get("rag_enabled"))))
            if response and response.get("status") != "success":
                acc["errors"][response.get("error_type") or "unknown"] += 1
            if response and native_status == "error":
                acc["native_errors"][response.get("native_error_type") or response.get("error_type") or "unknown"] += 1
            if score is not None:
                acc["scores"].append(score)
            if isinstance(efficiency, (int, float)):
                acc["efficiency"].append(float(efficiency))

        paired: dict[str, Any] = {}
        for comparison in comparisons:
            comparison_id = comparison.get("id")
            baseline_id = comparison.get("baseline_system")
            candidate_id = comparison.get("candidate_system")
            if not comparison_id or not baseline_id or not candidate_id:
                continue
            baseline_rating = rating_by_key.get((case_id, baseline_id))
            candidate_rating = rating_by_key.get((case_id, candidate_id))
            baseline_score = baseline_rating.get("total_score") if baseline_rating else None
            candidate_score = candidate_rating.get("total_score") if candidate_rating else None
            gain = (
                float(candidate_score) - float(baseline_score)
                if isinstance(baseline_score, (int, float)) and isinstance(candidate_score, (int, float))
                else None
            )
            baseline_steps = _step_rewards(baseline_rating)
            candidate_steps = _step_rewards(candidate_rating)
            common_steps = sorted(set(baseline_steps) & set(candidate_steps))
            step_gains = {
                str(step_id): {
                    "label": step_labels.get(case_id, {}).get(step_id, f"S{step_id}"),
                    "gain": candidate_steps[step_id]["score"] - baseline_steps[step_id]["score"],
                }
                for step_id in common_steps
            }
            baseline_complete = bool(
                response_by_key.get((case_id, baseline_id), {}).get("status") == "success"
            )
            candidate_complete = bool(
                response_by_key.get((case_id, candidate_id), {}).get("status") == "success"
            )
            paired[comparison_id] = {
                "baseline_system": baseline_id,
                "candidate_system": candidate_id,
                "total_gain": gain,
                "completion_gain": int(candidate_complete) - int(baseline_complete),
                "step_gains": step_gains,
            }
            if gain is not None:
                comparison_acc[comparison_id]["gains"].append(gain)
            comparison_acc[comparison_id]["completion_gains"].append(
                int(candidate_complete) - int(baseline_complete)
            )
            for step_id, row in step_gains.items():
                comparison_acc[comparison_id]["step_gains"][step_id].append(row["gain"])
        case_rows.append({"case_id": case_id, "systems": rewards, "comparisons": paired})

    route_spec = profile.get("adaptive_rag_analysis") or {}
    adaptive_rows: list[dict[str, Any]] = []
    use_expected_route = False
    if route_spec:
        no_rag_id = route_spec.get("no_rag_system")
        rag_id = route_spec.get("always_rag_system")
        adaptive_id = route_spec.get("adaptive_system")
        minimum_gain = float(route_spec.get("minimum_gain_for_rag", 0))
        use_expected_route = bool(route_spec.get("use_benchmark_expected_route", False))
        default_expected_route = route_spec.get("default_expected_route")
        if default_expected_route not in {None, "skip_rag", "use_rag"}:
            raise ValueError(
                "adaptive_rag_analysis.default_expected_route must be skip_rag or use_rag"
            )
        default_rag_need = route_spec.get("default_rag_need")
        for case_id in case_ids:
            no_rating = rating_by_key.get((case_id, no_rag_id))
            rag_rating = rating_by_key.get((case_id, rag_id))
            adaptive_rating = rating_by_key.get((case_id, adaptive_id))
            response = response_by_key.get((case_id, adaptive_id), {})
            route = response.get("routing") or {}
            scores = [
                row.get("total_score") if row else None
                for row in (no_rating, rag_rating, adaptive_rating)
            ]
            no_score, rag_score, independent_adaptive_score = scores
            arm_scores_complete = all(
                isinstance(value, (int, float)) for value in (no_score, rag_score)
            )
            optimal_action = None
            regret = None
            correct = None
            policy_score = None
            execution_gap = None
            case_view = benchmark_views.get(case_id, {})
            expected_action = case_view.get("expected_route", default_expected_route)
            policy_correct = (
                route.get("action") == expected_action
                if expected_action in {"skip_rag", "use_rag"}
                else None
            )
            if arm_scores_complete:
                optimal_action = "use_rag" if float(rag_score) - float(no_score) > minimum_gain else "skip_rag"
                if route.get("action") == "skip_rag":
                    policy_score = float(no_score)
                elif route.get("action") == "use_rag":
                    policy_score = float(rag_score)
                if policy_score is not None:
                    regret = max(
                        0.0,
                        max(float(no_score), float(rag_score)) - policy_score,
                    )
                if isinstance(independent_adaptive_score, (int, float)) and policy_score is not None:
                    execution_gap = float(independent_adaptive_score) - policy_score
                correct = route.get("action") == optimal_action
            adaptive_rows.append(
                {
                    "case_id": case_id,
                    "route_action": route.get("action"),
                    "route_reason_code": route.get("reason_code"),
                    "route_confidence": route.get("confidence"),
                    "router_status": route.get("status"),
                    "score_if_skip_rag": float(no_score) if isinstance(no_score, (int, float)) else None,
                    "score_if_use_rag": float(rag_score) if isinstance(rag_score, (int, float)) else None,
                    # The policy score reuses the score of the physical arm selected by
                    # the Router.  A separately sampled adaptive run is an operational
                    # replicate, not a causal estimate of routing quality.
                    "adaptive_score": policy_score,
                    "policy_replay_score": policy_score,
                    "independent_adaptive_score": (
                        float(independent_adaptive_score)
                        if isinstance(independent_adaptive_score, (int, float))
                        else None
                    ),
                    "independent_execution_gap": execution_gap,
                    "optimal_action": optimal_action,
                    "routing_correct": correct,
                    "expected_action": expected_action,
                    "rag_need": case_view.get("rag_need", default_rag_need),
                    "policy_routing_correct": policy_correct,
                    "routing_regret": regret,
                }
            )

    comparable_routes = [row for row in adaptive_rows if row["routing_regret"] is not None]
    labeled_routes = [
        row for row in adaptive_rows if row["policy_routing_correct"] is not None
    ]
    comparison_summary = {
        comparison_id: {
            "n": len(acc["gains"]),
            "mean_total_gain": _mean(acc["gains"]),
            "mean_total_gain_ci95": _bootstrap_mean_ci(acc["gains"]),
            "mean_completion_gain": _mean(acc["completion_gains"]),
            "mean_step_gains": {
                step_id: _mean(values) for step_id, values in acc["step_gains"].items()
            },
        }
        for comparison_id, acc in comparison_acc.items()
    }
    interaction_effects = []
    for interaction in profile.get("interaction_effects", []) or []:
        with_id = interaction.get("with_context_comparison")
        without_id = interaction.get("without_context_comparison")
        with_gain = (comparison_summary.get(with_id) or {}).get("mean_total_gain")
        without_gain = (comparison_summary.get(without_id) or {}).get("mean_total_gain")
        value = (
            float(with_gain) - float(without_gain)
            if isinstance(with_gain, (int, float)) and isinstance(without_gain, (int, float))
            else None
        )
        interaction_effects.append(
            {
                "id": interaction.get("id"),
                "with_context_comparison": with_id,
                "without_context_comparison": without_id,
                "effect": value,
                "interpretation": (
                    "synergy" if value is not None and value > 0 else
                    "interference" if value is not None and value < 0 else
                    "neutral" if value == 0 else "unavailable"
                ),
            }
        )
    return {
        "schema_version": "1.1",
        "run_id": run_dir.name,
        "profile_id": profile.get("profile_id"),
        "reward_definition": {
            "primary": "end_to_end_task_score",
            "dense": "rubric_step_scores",
            "secondary": [
                "tool_efficiency_score",
                "completion_rate",
                "native_completion_rate",
                "recovery_rate",
                "tool_calls",
            ],
            "adaptive_rag": "offline selected-arm policy replay with paired counterfactual routing regret",
        },
        "systems": {
            system_id: {
                "n_expected": acc["expected"],
                "n_rated": len(acc["scores"]),
                "mean_task_reward": _mean(acc["scores"]),
                "mean_tool_efficiency_reward": _mean(acc["efficiency"]),
                "completion_rate": acc["completed"] / acc["expected"] if acc["expected"] else None,
                "native_completion_rate": (
                    acc["native_completed"] / acc["expected"] if acc["expected"] else None
                ),
                "recovery_rate": (
                    acc["completion_modes"].get("context_reset_finalizer", 0) / acc["expected"]
                    if acc["expected"]
                    else None
                ),
                "mean_tool_calls": _mean(acc["tool_calls"]),
                "mean_latency_ms": _mean(acc["latencies"]),
                "rag_activation_rate": _mean(acc["rag_activations"]),
                "status_counts": dict(acc["statuses"]),
                "completion_mode_counts": dict(acc["completion_modes"]),
                "error_counts": dict(acc["errors"]),
                "native_error_counts": dict(acc["native_errors"]),
            }
            for system_id, acc in system_acc.items()
        },
        "comparisons": comparison_summary,
        "interaction_effects": interaction_effects,
        "cases": case_rows,
        "adaptive_rag": {
            "config": route_spec,
            "n_comparable": len(comparable_routes),
            "routing_accuracy": (
                sum(bool(row["policy_routing_correct"]) for row in labeled_routes) / len(labeled_routes)
                if use_expected_route and labeled_routes
                else sum(bool(row["routing_correct"]) for row in comparable_routes) / len(comparable_routes)
                if comparable_routes
                else None
            ),
            "routing_accuracy_basis": (
                "benchmark_expected_route"
                if use_expected_route and labeled_routes
                else "paired_counterfactual_reward"
            ),
            "policy_routing_accuracy": (
                sum(bool(row["policy_routing_correct"]) for row in labeled_routes) / len(labeled_routes)
                if labeled_routes
                else None
            ),
            "policy_routing_classification": binary_route_metrics(
                adaptive_rows,
                expected_key="expected_action",
                predicted_key="route_action",
            ),
            "counterfactual_routing_accuracy": (
                sum(bool(row["routing_correct"]) for row in comparable_routes) / len(comparable_routes)
                if comparable_routes
                else None
            ),
            "mean_routing_regret": _mean(
                [float(row["routing_regret"]) for row in comparable_routes]
            ),
            "by_rag_need": {
                rag_need: {
                    "n": len(rows),
                    "policy_routing_accuracy": _mean(
                        [
                            float(row["policy_routing_correct"])
                            for row in rows
                            if row["policy_routing_correct"] is not None
                        ]
                    ),
                    "mean_score_if_skip_rag": _mean(
                        [
                            float(row["score_if_skip_rag"])
                            for row in rows
                            if row["score_if_skip_rag"] is not None
                        ]
                    ),
                    "mean_score_if_use_rag": _mean(
                        [
                            float(row["score_if_use_rag"])
                            for row in rows
                            if row["score_if_use_rag"] is not None
                        ]
                    ),
                    "mean_adaptive_score": _mean(
                        [
                            float(row["adaptive_score"])
                            for row in rows
                            if row["adaptive_score"] is not None
                        ]
                    ),
                    "mean_independent_adaptive_score": _mean(
                        [
                            float(row["independent_adaptive_score"])
                            for row in rows
                            if row["independent_adaptive_score"] is not None
                        ]
                    ),
                    "mean_independent_execution_gap": _mean(
                        [
                            float(row["independent_execution_gap"])
                            for row in rows
                            if row["independent_execution_gap"] is not None
                        ]
                    ),
                    "mean_rag_gain": _mean(
                        [
                            float(row["score_if_use_rag"])
                            - float(row["score_if_skip_rag"])
                            for row in rows
                            if row["score_if_use_rag"] is not None
                            and row["score_if_skip_rag"] is not None
                        ]
                    ),
                    "mean_routing_regret": _mean(
                        [
                            float(row["routing_regret"])
                            for row in rows
                            if row["routing_regret"] is not None
                        ]
                    ),
                }
                for rag_need in sorted(
                    {
                        str(row["rag_need"])
                        for row in adaptive_rows
                        if row.get("rag_need") is not None
                    }
                )
                for rows in [
                    [row for row in adaptive_rows if str(row.get("rag_need")) == rag_need]
                ]
            },
            "cases": adaptive_rows,
        },
    }


def build_router_update_plan(run_dir: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    """Summarize routing errors without reviving the retired solver-Skill loop."""
    responses = _response_lookup(run_dir)
    route_rows = (analysis.get("adaptive_rag") or {}).get("cases", [])
    use_expected_route = bool(
        ((analysis.get("adaptive_rag") or {}).get("config") or {}).get(
            "use_benchmark_expected_route", False
        )
    )
    correctness_key = "policy_routing_correct" if use_expected_route else "routing_correct"
    misroutes = [row for row in route_rows if row.get(correctness_key) is False]
    router_reason_counts = Counter(row.get("route_reason_code") or "UNKNOWN" for row in misroutes)
    return {
        "schema_version": "1.0",
        "source_run_id": run_dir.name,
        "status": "evidence_ready" if route_rows else "awaiting_routing_evidence",
        "guardrails": [
            "Revise the Router only from information-need errors, never from case IDs or reference answers.",
            "Create a new immutable Router version for each evaluated revision.",
            "Keep the solver prompt, 9B weights, Tools preset, and RAG corpus fixed.",
        ],
        "router_skill_target": {
            "router_skill_version": next(
                (
                    (response.get("routing") or {}).get("router_skill_version")
                    for response in responses.values()
                    if (response.get("routing") or {}).get("router_skill_version")
                ),
                None,
            ),
            "n_misroutes": len(misroutes),
            "misroute_reason_counts": dict(router_reason_counts),
            "mean_routing_regret": (analysis.get("adaptive_rag") or {}).get("mean_routing_regret"),
            "evidence_cases": [row.get("case_id") for row in misroutes],
        },
    }
