from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl
from .taxonomy import FAILURE_CODE_DEFINITIONS


def _load_run(run_dir: Path) -> dict[str, Any]:
    """Load all inputs of a single run and cross-link them."""
    ratings = read_jsonl(run_dir / "ratings.jsonl")

    mapping_doc = (
        read_json(run_dir / "judge_mapping.json")
        if (run_dir / "judge_mapping.json").exists()
        else {}
    )
    mapping = {
        row["task_id"]: row
        for row in mapping_doc.get("mapping", [])
        if row.get("task_id")
    }

    responses: list[dict[str, Any]] = []
    for path in sorted((run_dir / "responses").glob("*.json")):
        if path.name == "index.json":
            continue
        responses.append(read_json(path))

    families: dict[str, str] = {}
    if (run_dir / "benchmarks").exists():
        for path in sorted((run_dir / "benchmarks").glob("*.json")):
            if path.name == "index.json":
                continue
            row = read_json(path)
            case_id = row.get("case_id") or path.stem
            if case_id:
                families[case_id] = row.get("task_family") or "unknown"

    manifest = (
        read_json(run_dir / "manifest.json")
        if (run_dir / "manifest.json").exists()
        else {}
    )
    systems = manifest.get("systems") or []
    return {
        "run_dir": run_dir,
        "ratings": ratings,
        "mapping": mapping,
        "responses": responses,
        "families": families,
        "systems": systems,
    }


def _system_flags(systems: list[dict[str, Any]], system_id: str) -> dict[str, Any]:
    for system in systems:
        if system.get("id") == system_id:
            return {
                "tools_enabled": bool(system.get("tools_enabled", False)),
                "rag_enabled": bool(system.get("rag_enabled", False)),
                "skill_version": system.get("skill_version"),
            }
    return {"tools_enabled": False, "rag_enabled": False, "skill_version": None}


def build_failure_analysis(run_dir: Path) -> dict[str, Any]:
    """Aggregate scores, failure codes and tool/RAG utilization of one run.

    Every metric is attributed per system_id and, where possible, per task_family.
    Tool/RAG utilization is intentionally reported independently of the score:
    a model is neither penalized nor rewarded for calling an available tool; the
    numbers only reveal whether the model *chose* to use what it had.
    """
    run = _load_run(run_dir)
    rating_by_task: dict[str, dict[str, Any]] = {
        row.get("task_id"): row for row in run["ratings"] if row.get("task_id")
    }
    flags_by_system: dict[str, dict[str, Any]] = {}

    # ---- score matrix: system x task_family ---------------------------------
    score_cells: defaultdict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    efficiency_cells: defaultdict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    score_details: list[dict[str, Any]] = []
    for task_id, rating in rating_by_task.items():
        identity = run["mapping"].get(task_id, {})
        system_id = identity.get("system_id") or "unknown"
        case_id = rating.get("case_id") or identity.get("case_id") or "unknown"
        family = run["families"].get(case_id, "unknown")
        flags_by_system.setdefault(
            system_id,
            {"tools_enabled": False, "rag_enabled": False, "skill_version": None},
        )
        score = float(rating.get("total_score", 0))
        score_cells[system_id][family].append(score)
        efficiency = rating.get("tool_efficiency_score")
        if isinstance(efficiency, (int, float)):
            efficiency_cells[system_id][family].append(float(efficiency))
        score_details.append(
            {
                "task_id": task_id,
                "case_id": case_id,
                "task_family": family,
                "system_id": system_id,
                "display_name": identity.get("display_name", system_id),
                "total_score": score,
                "tool_efficiency_score": efficiency,
            }
        )

    for system in run["systems"]:
        flags_by_system[system.get("id")] = _system_flags(
            run["systems"], system.get("id")
        )

    # ---- failure codes: system x code ---------------------------------------
    code_stats: defaultdict[str, defaultdict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {"labels": 0, "affected_responses": 0, "affected_cases": 0}
        )
    )
    for task_id, rating in rating_by_task.items():
        identity = run["mapping"].get(task_id, {})
        system_id = identity.get("system_id") or "unknown"
        case_id = rating.get("case_id") or identity.get("case_id") or "unknown"
        seen_codes: set[str] = set()
        for step in rating.get("steps", []):
            loss = max(0.0, float(step.get("max_score", 0)) - float(step.get("score", 0)))
            if loss <= 0:
                continue
            codes = list(dict.fromkeys(step.get("failure_codes", []) or []))
            for code in codes:
                cell = code_stats[system_id][code]
                cell["labels"] += 1
                if code not in seen_codes:
                    cell["affected_responses"] += 1
                    seen_codes.add(code)
            if not codes and "UNLABELED" not in seen_codes:
                cell = code_stats[system_id]["UNLABELED"]
                cell["labels"] += 1
                cell["affected_responses"] += 1
                seen_codes.add("UNLABELED")

    # ---- tool / RAG utilization ---------------------------------------------
    response_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for response in run["responses"]:
        case_id = response.get("case_id")
        system_id = response.get("system_id")
        if case_id and system_id:
            response_by_key[(case_id, system_id)] = response

    utilization_cells: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "responses": 0,
            "with_tool_call": 0,
            "with_retrieval": 0,
            "total_tool_calls": 0,
            "total_retrievals": 0,
            "tool_errors": 0,
            "successes": 0,
        }
    )
    utilization_by_family: defaultdict[
        str, defaultdict[str, dict[str, Any]]
    ] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "responses": 0,
                "with_tool_call": 0,
                "with_retrieval": 0,
                "total_tool_calls": 0,
                "total_retrievals": 0,
                "tool_errors": 0,
            }
        )
    )
    for (case_id, system_id), response in sorted(response_by_key.items()):
        family = run["families"].get(case_id, "unknown")
        trajectory = response.get("trajectory") or {}
        summary = trajectory.get("summary") or {}
        tool_calls = int(summary.get("tool_interactions", 0) or 0)
        retrievals = int(summary.get("retrieval_interactions", 0) or 0)
        tool_errors = int(summary.get("tool_errors", 0) or 0)
        success = response.get("status") == "success"

        for target in (utilization_cells[system_id], utilization_by_family[family][system_id]):
            target["responses"] += 1
            if tool_calls > 0:
                target["with_tool_call"] += 1
            if retrievals > 0:
                target["with_retrieval"] += 1
            target["total_tool_calls"] += tool_calls
            target["total_retrievals"] += retrievals
            target["tool_errors"] += tool_errors
        if success:
            utilization_cells[system_id]["successes"] += 1

    # ---- assemble ------------------------------------------------------------
    family_names = sorted(
        {family for by_system in score_cells.values() for family in by_system}
    )
    system_ids = sorted(
        set(score_cells) | set(code_stats) | set(utilization_cells)
    )
    return {
        "run_id": run_dir.name,
        "family_names": family_names,
        "system_ids": system_ids,
        "systems": {
            system_id: flags_by_system.get(
                system_id,
                {"tools_enabled": False, "rag_enabled": False, "skill_version": None},
            )
            for system_id in system_ids
        },
        "scores_by_family": {
            system_id: {
                family: {
                    "n": len(score_cells[system_id][family]),
                    "mean_score": (
                        sum(score_cells[system_id][family])
                        / len(score_cells[system_id][family])
                        if score_cells[system_id][family]
                        else None
                    ),
                    "mean_tool_efficiency": (
                        sum(efficiency_cells[system_id][family])
                        / len(efficiency_cells[system_id][family])
                        if efficiency_cells[system_id][family]
                        else None
                    ),
                }
                for family in family_names
            }
            for system_id in system_ids
        },
        "score_details": sorted(
            score_details,
            key=lambda row: (row["task_family"], row["system_id"]),
        ),
        "failure_codes": {
            system_id: {
                code: {
                    "label_zh": FAILURE_CODE_DEFINITIONS.get(code, {}).get(
                        "label_zh", code
                    ),
                    **stats,
                }
                for code, stats in sorted(
                    code_stats[system_id].items(), key=lambda item: -item[1]["labels"]
                )
            }
            for system_id in system_ids
        },
        "utilization": {
            system_id: utilization_cells[system_id] for system_id in system_ids
        },
        "utilization_by_family": {
            family: {system_id: utilization_by_family[family][system_id] for system_id in system_ids}
            for family in family_names
        },
    }


# --------------------------------------------------------------------------- #
# Rendering helpers (console-friendly)
# --------------------------------------------------------------------------- #

def _fmt(value: Any, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(
        value, bool
    ) else str(value if value is not None else "—")


def render_text_analysis(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"== Failure / Utilization Analysis: {analysis['run_id']} ==")

    # 1. score matrix
    lines.append("\n[1] 均分矩阵（system x task_family）")
    header = ["system/task_family"] + analysis["family_names"] + ["all"]
    widths = [max(len(item) for item in header)]
    for system_id in analysis["system_ids"]:
        display = f"{system_id} (t={int(analysis['systems'][system_id]['tools_enabled'])}"
        display += f",r={int(analysis['systems'][system_id]['rag_enabled'])}"
        skill = analysis["systems"][system_id].get("skill_version")
        display += f",s={bool(skill)})"
        widths[0] = max(widths[0], len(display))
    header[0] = header[0].ljust(widths[0])
    for family in analysis["family_names"]:
        widths.append(max(len(family), (len(header) * 0 + 6)))
    rows = []
    for system_id in analysis["system_ids"]:
        display = f"{system_id} (t={int(analysis['systems'][system_id]['tools_enabled'])},"
        display += f"r={int(analysis['systems'][system_id]['rag_enabled'])},"
        display += f"s={bool(analysis['systems'][system_id].get('skill_version'))})"
        row = [display.ljust(widths[0])]
        for family in analysis["family_names"]:
            cell = analysis["scores_by_family"][system_id].get(family) or {}
            value = cell.get("mean_score")
            if value is None:
                row.append("—".center(6))
            else:
                row.append(f"{value:.1f}".center(6))
        all_scores = [
            value
            for family in analysis["family_names"]
            if (value := analysis["scores_by_family"][system_id].get(family, {}).get("mean_score")) is not None
        ]
        row.append((f"{sum(all_scores)/len(all_scores):.1f}" if all_scores else "—").center(6))
        rows.append(" ".join(row))
    lines.append(" | ".join(header))
    lines.append(" | ".join("-" * len(item) for item in header))
    lines.extend(rows)

    # 2. failure codes
    lines.append("\n[2] 失败码分布（按 system，次数 = rubric 步骤标签数）")
    code_rows = []
    for system_id in analysis["system_ids"]:
        for code, stats in analysis["failure_codes"].get(system_id, {}).items():
            code_rows.append((system_id, code, stats))
    if not code_rows:
        lines.append("（没有任何步骤失分）")
    else:
        lines.append(
            " | ".join(
                ["system".ljust(14), "failure_code".ljust(22), "labels", "resp", "cases"]
            )
        )
        lines.append(" | ".join(["-" * 14, "-" * 22, "-" * 6, "-" * 4, "-" * 5]))
        for system_id, code, stats in sorted(
            code_rows, key=lambda row: (-row[2]["labels"], row[0], row[1])
        ):
            lines.append(
                " | ".join(
                    [
                        system_id.ljust(14),
                        f"{code} ({stats['label_zh']})".ljust(22),
                        str(stats["labels"]).rjust(6),
                        str(stats["affected_responses"]).rjust(4),
                        str(stats["affected_cases"]).rjust(5),
                    ]
                )
            )

    # 3. tool / RAG utilization
    lines.append("\n[3] 工具/RAG 自主利用（模型实际调用情况，与得分独立）")
    lines.append(
        " | ".join(
            [
                "system".ljust(14),
                "cfg".ljust(9),
                "resp".rjust(4),
                "用工具".rjust(6),
                "调工次数".rjust(8),
                "用RAG".rjust(6),
                "检次数".rjust(6),
                "错误".rjust(4),
            ]
        )
    )
    lines.append(" | ".join(["-" * 14, "-" * 9, "-" * 4, "-" * 6, "-" * 8, "-" * 6, "-" * 6, "-" * 4]))
    for system_id in analysis["system_ids"]:
        cell = analysis["utilization"].get(system_id, {})
        flags = analysis["systems"].get(system_id, {})
        cfg = (
            ("T" if flags.get("tools_enabled") else ".")
            + ("R" if flags.get("rag_enabled") else ".")
            + ("S" if flags.get("skill_version") else ".")
        )
        n = cell.get("responses", 0)
        lines.append(
            " | ".join(
                [
                    system_id.ljust(14),
                    cfg.ljust(9),
                    str(n).rjust(4),
                    f"{cell.get('with_tool_call', 0)}/{n}".rjust(6),
                    str(cell.get("total_tool_calls", 0)).rjust(8),
                    f"{cell.get('with_retrieval', 0)}/{n}".rjust(6),
                    str(cell.get("total_retrievals", 0)).rjust(6),
                    str(cell.get("tool_errors", 0)).rjust(4),
                ]
            )
        )
    return "\n".join(lines)