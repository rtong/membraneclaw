from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .io_utils import read_json, read_jsonl, write_json


REPEAT_SYSTEMS = ("baseline", "tools", "tools-rag")
REPEAT_COMPARISONS = (
    ("tools_gain", "baseline", "tools", "Tools - Baseline"),
    ("rag_effect", "tools", "tools-rag", "Tools + RAG - Tools"),
)


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def _load_run_rows(
    run_dir: Path,
    *,
    case_ids: list[str],
    system_ids: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, Any]]:
    mapping_doc = read_json(run_dir / "judge_mapping.json")
    mapping = {
        row["task_id"]: row
        for row in mapping_doc.get("mapping", [])
        if isinstance(row, dict) and row.get("task_id")
    }
    responses: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (run_dir / "responses").glob("*.json"):
        response = read_json(path)
        key = (str(response.get("case_id")), str(response.get("system_id")))
        responses[key] = response

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for rating in read_jsonl(run_dir / "ratings.jsonl"):
        identity = mapping.get(rating.get("task_id"), {})
        case_id = str(rating.get("case_id") or identity.get("case_id") or "")
        system_id = str(identity.get("system_id") or rating.get("system_id") or "")
        if case_id not in case_ids or system_id not in system_ids:
            continue
        response = responses.get((case_id, system_id), {})
        summary = (response.get("trajectory") or {}).get("summary") or {}
        path = (rating.get("trajectory_analysis") or {}).get("path_classification")
        first_error = (rating.get("causal_analysis") or {}).get("first_error_step_id")
        failures = {
            str(code)
            for step in rating.get("steps", [])
            for code in step.get("failure_codes", [])
        }
        rows[(case_id, system_id)] = {
            "score": float(rating["total_score"]),
            "status": response.get("status", "missing"),
            "native_status": response.get("native_status", response.get("status", "missing")),
            "completion_mode": response.get("completion_mode", "missing"),
            "tool_calls": float(summary.get("tool_interactions", 0) or 0),
            "latency_ms": (
                float(response["latency_ms"])
                if isinstance(response.get("latency_ms"), (int, float))
                else None
            ),
            "valid_path": path in {"golden_aligned", "valid_alternative"},
            "first_error_step": (
                float(first_error)
                if isinstance(first_error, (int, float))
                else None
            ),
            "tool_argument_failure": "TOOL_ARGUMENT" in failures,
        }

    missing = [
        f"{case_id}/{system_id}"
        for case_id in case_ids
        for system_id in system_ids
        if (case_id, system_id) not in rows
    ]
    if missing:
        examples = ", ".join(missing[:8])
        raise ValueError(
            f"Run {run_dir.name!r} is missing {len(missing)} required rating(s): {examples}. "
            "Complete the Judge stage before aggregating."
        )
    return rows


def build_repeatability_analysis(
    run_dirs: list[Path],
    *,
    selection: dict[str, Any],
    bootstrap_samples: int = 5000,
    seed: int = 20260904,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("Repeatability analysis requires at least two run IDs")
    run_ids = [run_dir.name for run_dir in run_dirs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Repeatability analysis requires unique run IDs")
    case_ids = [str(case_id) for case_id in selection.get("case_ids") or []]
    if not case_ids:
        raise ValueError("Repeatability selection has no case_ids")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Repeatability selection contains duplicate case IDs")

    per_run = {
        run_dir.name: _load_run_rows(
            run_dir, case_ids=case_ids, system_ids=REPEAT_SYSTEMS
        )
        for run_dir in run_dirs
    }
    run_summaries: list[dict[str, Any]] = []
    for run_id, lookup in per_run.items():
        for system_id in REPEAT_SYSTEMS:
            rows = [lookup[(case_id, system_id)] for case_id in case_ids]
            first_steps = [
                row["first_error_step"]
                for row in rows
                if row["first_error_step"] is not None
            ]
            latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
            run_summaries.append(
                {
                    "run_id": run_id,
                    "system_id": system_id,
                    "n": len(rows),
                    "mean_score": mean(row["score"] for row in rows),
                    "final_completion_rate": mean(row["status"] == "success" for row in rows),
                    "native_completion_rate": mean(row["native_status"] == "success" for row in rows),
                    "mean_tool_calls": mean(row["tool_calls"] for row in rows),
                    "mean_latency_ms": mean(latencies) if latencies else None,
                    "valid_path_rate": mean(row["valid_path"] for row in rows),
                    "mean_first_error_step": mean(first_steps) if first_steps else None,
                    "tool_argument_failure_rate": mean(
                        row["tool_argument_failure"] for row in rows
                    ),
                }
            )

    case_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        for system_id in REPEAT_SYSTEMS:
            scores = [lookup[(case_id, system_id)]["score"] for lookup in per_run.values()]
            case_rows.append(
                {
                    "case_id": case_id,
                    "system_id": system_id,
                    "replicates": len(scores),
                    "mean_score": mean(scores),
                    "score_sd": stdev(scores) if len(scores) > 1 else 0.0,
                    "minimum_score": min(scores),
                    "maximum_score": max(scores),
                }
            )

    system_summaries: dict[str, Any] = {}
    for system_id in REPEAT_SYSTEMS:
        case_means = [
            row["mean_score"]
            for row in case_rows
            if row["system_id"] == system_id
        ]
        run_means = [
            row["mean_score"]
            for row in run_summaries
            if row["system_id"] == system_id
        ]
        system_summaries[system_id] = {
            "n_cases": len(case_means),
            "n_replicates": len(run_dirs),
            "mean_of_case_means": mean(case_means),
            "run_mean_sd": stdev(run_means) if len(run_means) > 1 else 0.0,
            "run_means": dict(zip((run_dir.name for run_dir in run_dirs), run_means)),
        }

    comparisons: dict[str, Any] = {}
    for comparison_index, (comparison_id, baseline_id, candidate_id, label) in enumerate(
        REPEAT_COMPARISONS
    ):
        gains_by_case: dict[str, list[float]] = {}
        case_mean_gains = []
        signs = Counter()
        for case_id in case_ids:
            gains = [
                lookup[(case_id, candidate_id)]["score"]
                - lookup[(case_id, baseline_id)]["score"]
                for lookup in per_run.values()
            ]
            gains_by_case[case_id] = gains
            case_mean = mean(gains)
            case_mean_gains.append(case_mean)
            if all(gain > 0 for gain in gains):
                signs["positive_all_replicates"] += 1
            elif all(gain < 0 for gain in gains):
                signs["negative_all_replicates"] += 1
            elif all(gain == 0 for gain in gains):
                signs["tie_all_replicates"] += 1
            else:
                signs["mixed_direction"] += 1

        run_effects = {
            run_id: mean(
                lookup[(case_id, candidate_id)]["score"]
                - lookup[(case_id, baseline_id)]["score"]
                for case_id in case_ids
            )
            for run_id, lookup in per_run.items()
        }
        comparisons[comparison_id] = {
            "label": label,
            "baseline_system": baseline_id,
            "candidate_system": candidate_id,
            "n_cases": len(case_ids),
            "n_replicates": len(run_dirs),
            "mean_case_averaged_effect": mean(case_mean_gains),
            "case_bootstrap_ci95": _bootstrap_mean_ci(
                case_mean_gains,
                samples=bootstrap_samples,
                seed=seed + comparison_index,
            ),
            "run_effects": run_effects,
            "run_effect_sd": (
                stdev(run_effects.values()) if len(run_effects) > 1 else 0.0
            ),
            "case_direction_consistency": {
                key: signs.get(key, 0)
                for key in (
                    "positive_all_replicates",
                    "negative_all_replicates",
                    "tie_all_replicates",
                    "mixed_direction",
                )
            },
            "case_mean_wins_ties_losses": {
                "wins": sum(gain > 0 for gain in case_mean_gains),
                "ties": sum(gain == 0 for gain in case_mean_gains),
                "losses": sum(gain < 0 for gain in case_mean_gains),
            },
            "case_replicate_gains": gains_by_case,
        }

    return {
        "schema_version": "1.0",
        "selection_id": selection.get("selection_id"),
        "case_ids": case_ids,
        "systems": list(REPEAT_SYSTEMS),
        "run_ids": run_ids,
        "n_cases": len(case_ids),
        "n_replicates": len(run_dirs),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "run_summaries": run_summaries,
        "case_summaries": case_rows,
        "system_summaries": system_summaries,
        "comparisons": comparisons,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_repeatability_analysis(output_dir: Path, analysis: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", analysis)
    _write_csv(output_dir / "run_summary.csv", analysis["run_summaries"])
    _write_csv(output_dir / "case_summary.csv", analysis["case_summaries"])
    comparison_rows = []
    for comparison_id, row in analysis["comparisons"].items():
        ci = row.get("case_bootstrap_ci95") or [None, None]
        directions = row["case_direction_consistency"]
        outcomes = row["case_mean_wins_ties_losses"]
        comparison_rows.append(
            {
                "comparison_id": comparison_id,
                "label": row["label"],
                "n_cases": row["n_cases"],
                "n_replicates": row["n_replicates"],
                "mean_effect": row["mean_case_averaged_effect"],
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "run_effect_sd": row["run_effect_sd"],
                "wins": outcomes["wins"],
                "ties": outcomes["ties"],
                "losses": outcomes["losses"],
                "positive_all_replicates": directions["positive_all_replicates"],
                "negative_all_replicates": directions["negative_all_replicates"],
                "mixed_direction": directions["mixed_direction"],
            }
        )
    _write_csv(output_dir / "comparison_summary.csv", comparison_rows)

    lines = [
        "# D1–D6分层重复性审计",
        "",
        f"- Selection：`{analysis.get('selection_id')}`",
        f"- Cases：{analysis['n_cases']}",
        f"- Replicates：{analysis['n_replicates']}",
        f"- Runs：{', '.join(f'`{run_id}`' for run_id in analysis['run_ids'])}",
        "",
        "## 系统均分",
        "",
        "| 系统 | 跨题、跨重复均分 | run均分标准差 |",
        "|---|---:|---:|",
    ]
    for system_id in REPEAT_SYSTEMS:
        row = analysis["system_summaries"][system_id]
        lines.append(
            f"| {system_id} | {row['mean_of_case_means']:.2f} | {row['run_mean_sd']:.2f} |"
        )
    lines.extend(["", "## 配对效应", ""])
    for row in analysis["comparisons"].values():
        ci = row["case_bootstrap_ci95"]
        outcomes = row["case_mean_wins_ties_losses"]
        directions = row["case_direction_consistency"]
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 按题先平均后的效应：{row['mean_case_averaged_effect']:+.2f}。",
                f"- 95% case-bootstrap CI：[{ci[0]:+.2f}, {ci[1]:+.2f}]。",
                f"- run间效应标准差：{row['run_effect_sd']:.2f}。",
                f"- 按题平均后的胜/平/负：{outcomes['wins']}/{outcomes['ties']}/{outcomes['losses']}。",
                f"- 所有重复均正/均负/方向混合：{directions['positive_all_replicates']}/{directions['negative_all_replicates']}/{directions['mixed_direction']}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释规则",
            "",
            "- 每道题先在重复运行之间求平均，再把题目作为统计单位，避免把重复采样误当成新的独立题目。",
            "- 本审计只估计运行随机性；不能替代D7的R2和Router外部验证。",
            "- 若Tools效应方向稳定、系统排名稳定且完成率没有明显波动，则无需扩展到117题全量重复。",
        ]
    )
    (output_dir / "RESULTS_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir
