from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .io_utils import read_json, read_jsonl


TRAJECTORY_SYSTEMS = ("baseline", "tools", "tools-rag")
FAILURE_CODES = (
    "TASK_CLASSIFICATION",
    "PARAMETER_EXTRACTION",
    "TOOL_ARGUMENT",
    "TOOL_NOT_CALLED",
    "SEARCH_STRATEGY",
    "NUMERICAL_REASONING",
    "CONSTRAINT_OMISSION",
    "OUTPUT_OMISSION",
    "ENGINEERING_JUDGMENT",
    "OVERCLAIM",
)
EFFICIENCY_DIMENSIONS = ("E1", "E2", "E3", "E4", "E5")
D6_STATE_SYSTEMS = ("tools", "tools-rag")
D6_GROUPS = ("D6a", "D6b", "D6c", "D6-all")
D6_GROUP_LABELS = {
    "D6a": "混合串并联",
    "D6b": "串联",
    "D6c": "并联",
    "D6-all": "D6总体",
}
STATE_SENSITIVE_FAILURE_CODES = (
    "PARAMETER_EXTRACTION",
    "TOOL_ARGUMENT",
    "CONSTRAINT_OMISSION",
    "OUTPUT_OMISSION",
)


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _response_index(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    response_dir = run_dir / "responses"
    if not response_dir.exists():
        return rows
    for path in response_dir.glob("*.json"):
        row = read_json(path)
        case_id = row.get("case_id")
        system_id = row.get("system_id")
        if case_id and system_id:
            rows[(str(case_id), str(system_id))] = row
    return rows


def _d6_group(run_dir: Path, case_id: str) -> str | None:
    benchmark_path = run_dir / "benchmarks" / f"{case_id}.json"
    family = ""
    if benchmark_path.exists():
        family = str(read_json(benchmark_path).get("task_family") or "").lower()
    value = f"{family} {case_id.lower()}"
    for token, group in (("d6_6a", "D6a"), ("d6-6a", "D6a"),
                         ("d6_6b", "D6b"), ("d6-6b", "D6b"),
                         ("d6_6c", "D6c"), ("d6-6c", "D6c")):
        if token in value:
            return group
    return None


def _state_lineage_dimension(run_dir: Path, case_id: str) -> str | None:
    """Return the scored dimension ID only when the rubric names State lineage.

    Seven legacy D6b cases use a different efficiency rubric. They are deliberately
    excluded from the numeric State-lineage mean instead of treating an unrelated E3
    score as the same construct.
    """
    benchmark_path = run_dir / "benchmarks" / f"{case_id}.json"
    if not benchmark_path.exists():
        return None
    benchmark = read_json(benchmark_path)
    rubric = benchmark.get("tool_efficiency_rubric") or {}
    for dimension in rubric.get("dimensions") or []:
        label = str(dimension.get("dimension_label") or "").lower()
        if "state lineage" in label:
            return str(dimension.get("dimension_id") or "") or None
    return None


def _build_d6_state_metrics(
    run_dir: Path,
    records: list[tuple[str, str, dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "scores": [],
            "state_scores": [],
            "valid_paths": [],
            "first_error_steps": [],
            "downstream_breadth": [],
            "state_sensitive_failures": [],
            "calls": [],
            "latency_ms": [],
        }
    )
    failure_counts: Counter[tuple[str, str, str]] = Counter()
    completion: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"scores": [], "state_scores": [], "valid_paths": []}
    )
    cases: list[dict[str, Any]] = []

    for case_id, system_id, rating, response in records:
        if system_id not in D6_STATE_SYSTEMS:
            continue
        group = _d6_group(run_dir, case_id)
        if group is None:
            continue

        total_score = rating.get("total_score")
        state_dimension_id = _state_lineage_dimension(run_dir, case_id)
        state_score = None
        state_score_max = None
        if state_dimension_id:
            for dimension in rating.get("tool_efficiency_dimensions") or []:
                if str(dimension.get("dimension_id") or "") == state_dimension_id:
                    score = dimension.get("score")
                    maximum = dimension.get("max_score")
                    if isinstance(score, (int, float)):
                        state_score = float(score)
                    if isinstance(maximum, (int, float)):
                        state_score_max = float(maximum)
                    break

        trajectory = rating.get("trajectory_analysis") or {}
        causal = rating.get("causal_analysis") or {}
        path = str(trajectory.get("path_classification") or "unknown")
        valid_path = path in {"golden_aligned", "valid_alternative"}
        first_error_step = causal.get("first_error_step_id")
        downstream = causal.get("downstream_affected_steps") or []
        case_failures = {
            str(code)
            for rubric_step in rating.get("steps") or []
            for code in rubric_step.get("failure_codes") or []
        }
        state_failures = sorted(case_failures.intersection(STATE_SENSITIVE_FAILURE_CODES))
        raw_mode = str(response.get("completion_mode") or "native")
        mode = "recovered" if raw_mode == "context_reset_finalizer" else "native"
        response_summary = (response.get("trajectory") or {}).get("summary") or {}
        calls = response_summary.get("tool_interactions")
        latency = response.get("latency_ms")

        for group_key in (group, "D6-all"):
            values = aggregate[(group_key, system_id)]
            if isinstance(total_score, (int, float)):
                values["scores"].append(float(total_score))
            if state_score is not None:
                values["state_scores"].append(state_score)
            values["valid_paths"].append(1.0 if valid_path else 0.0)
            if isinstance(first_error_step, (int, float)):
                values["first_error_steps"].append(float(first_error_step))
            values["downstream_breadth"].append(float(len(set(downstream))))
            values["state_sensitive_failures"].append(1.0 if state_failures else 0.0)
            if isinstance(calls, (int, float)):
                values["calls"].append(float(calls))
            if isinstance(latency, (int, float)):
                values["latency_ms"].append(float(latency))
            for code in state_failures:
                failure_counts[(group_key, system_id, code)] += 1

            mode_values = completion[(group_key, system_id, mode)]
            if isinstance(total_score, (int, float)):
                mode_values["scores"].append(float(total_score))
            if state_score is not None:
                mode_values["state_scores"].append(state_score)
            mode_values["valid_paths"].append(1.0 if valid_path else 0.0)

        cases.append(
            {
                "case_id": case_id,
                "d6_group": group,
                "topology_label": D6_GROUP_LABELS[group],
                "system_id": system_id,
                "task_score": float(total_score) if isinstance(total_score, (int, float)) else None,
                "state_lineage_score": state_score,
                "state_lineage_max": state_score_max,
                "state_lineage_rubric_available": bool(state_dimension_id),
                "path_classification": path,
                "valid_path": valid_path,
                "completion_mode": mode,
                "first_error_step": first_error_step,
                "first_error_event_id": trajectory.get("first_error_event_id"),
                "downstream_affected_steps": ";".join(str(value) for value in downstream),
                "state_sensitive_failure_codes": ";".join(state_failures),
                "root_cause": causal.get("root_cause"),
                "tool_interactions": calls,
                "latency_ms": latency,
            }
        )

    groups: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    completion_modes: dict[str, dict[str, Any]] = {}
    for group in D6_GROUPS:
        groups[group] = {}
        failures[group] = {}
        completion_modes[group] = {}
        for system_id in D6_STATE_SYSTEMS:
            values = aggregate.get((group, system_id))
            if not values:
                continue
            n = len(values["scores"])
            groups[group][system_id] = {
                "n": n,
                "state_lineage_n": len(values["state_scores"]),
                "mean_task_score": _mean(values["scores"]),
                "mean_state_lineage_score": _mean(values["state_scores"]),
                "state_lineage_max": 20,
                "valid_path_rate": _mean(values["valid_paths"]),
                "mean_first_error_step": _mean(values["first_error_steps"]),
                "mean_downstream_breadth": _mean(values["downstream_breadth"]),
                "state_sensitive_failure_rate": _mean(values["state_sensitive_failures"]),
                "mean_tool_interactions": _mean(values["calls"]),
                "mean_latency_ms": _mean(values["latency_ms"]),
            }
            failures[group][system_id] = {
                code: {
                    "count": failure_counts[(group, system_id, code)],
                    "incidence": failure_counts[(group, system_id, code)] / n if n else None,
                }
                for code in STATE_SENSITIVE_FAILURE_CODES
            }
            for mode in ("native", "recovered"):
                mode_values = completion.get((group, system_id, mode))
                if not mode_values:
                    continue
                completion_modes[group].setdefault(system_id, {})[mode] = {
                    "n": len(mode_values["scores"]),
                    "state_lineage_n": len(mode_values["state_scores"]),
                    "mean_task_score": _mean(mode_values["scores"]),
                    "mean_state_lineage_score": _mean(mode_values["state_scores"]),
                    "valid_path_rate": _mean(mode_values["valid_paths"]),
                }

    return {
        "groups": groups,
        "failures": failures,
        "completion_modes": completion_modes,
        "cases": sorted(cases, key=lambda row: (row["d6_group"], row["case_id"], row["system_id"])),
        "state_lineage_case_coverage": len(
            {row["case_id"] for row in cases if row["state_lineage_rubric_available"]}
        ),
        "d6_case_count": len({row["case_id"] for row in cases}),
    }


def build_trajectory_metrics(run_dir: Path) -> dict[str, Any]:
    mapping_doc = read_json(run_dir / "judge_mapping.json")
    mapping = {
        row["task_id"]: row
        for row in mapping_doc.get("mapping", [])
        if isinstance(row, dict) and row.get("task_id")
    }
    responses = _response_index(run_dir)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    rated_records: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for rating in read_jsonl(run_dir / "ratings.jsonl"):
        identity = mapping.get(rating.get("task_id"), {})
        system_id = str(identity.get("system_id") or rating.get("system_id") or "unknown")
        if system_id not in TRAJECTORY_SYSTEMS:
            continue
        case_id = str(rating.get("case_id") or identity.get("case_id") or "unknown")
        response = responses.get((case_id, system_id), {})
        grouped[system_id].append((rating, response))
        rated_records.append((case_id, system_id, rating, response))

    systems: dict[str, Any] = {}
    for system_id in TRAJECTORY_SYSTEMS:
        rows = grouped.get(system_id, [])
        first_error = Counter()
        path_classification = Counter()
        failure_incidence = Counter()
        efficiency_values: dict[str, list[float]] = defaultdict(list)
        downstream_breadth: list[float] = []
        completion_modes: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"scores": [], "valid_paths": [], "calls": [], "latency_ms": []}
        )
        score_by_first_error: dict[str, list[float]] = defaultdict(list)
        call_bins: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"scores": [], "latency_ms": [], "calls": []}
        )
        event_assessments = 0

        for rating, response in rows:
            trajectory = rating.get("trajectory_analysis") or {}
            causal = rating.get("causal_analysis") or {}
            path = str(trajectory.get("path_classification") or "unknown")
            path_classification[path] += 1
            valid_path = path in {"golden_aligned", "valid_alternative"}
            event_assessments += len(trajectory.get("event_assessments") or [])

            step = causal.get("first_error_step_id")
            step_key = f"TQ{int(step)}" if isinstance(step, (int, float)) else "no_error"
            first_error[step_key] += 1
            if isinstance(rating.get("total_score"), (int, float)):
                score_by_first_error[step_key].append(float(rating["total_score"]))
            affected = causal.get("downstream_affected_steps") or []
            downstream_breadth.append(float(len(set(affected))))

            case_failures = {
                str(code)
                for rubric_step in rating.get("steps", [])
                for code in rubric_step.get("failure_codes", [])
            }
            failure_incidence.update(case_failures)
            for dimension in rating.get("tool_efficiency_dimensions", []):
                dimension_id = dimension.get("dimension_id")
                score = dimension.get("score")
                if dimension_id and isinstance(score, (int, float)):
                    efficiency_values[str(dimension_id)].append(float(score))

            raw_mode = str(response.get("completion_mode") or "native")
            mode = "recovered" if raw_mode == "context_reset_finalizer" else "native"
            summary = (response.get("trajectory") or {}).get("summary") or {}
            calls = summary.get("tool_interactions")
            latency = response.get("latency_ms")
            score = rating.get("total_score")
            if isinstance(score, (int, float)):
                completion_modes[mode]["scores"].append(float(score))
            completion_modes[mode]["valid_paths"].append(1.0 if valid_path else 0.0)
            if isinstance(calls, (int, float)):
                completion_modes[mode]["calls"].append(float(calls))
            if isinstance(latency, (int, float)):
                completion_modes[mode]["latency_ms"].append(float(latency))

            if isinstance(calls, (int, float)):
                if calls <= 7:
                    call_bin = "0-7"
                elif calls <= 11:
                    call_bin = "8-11"
                elif calls <= 15:
                    call_bin = "12-15"
                else:
                    call_bin = "16+"
                if isinstance(score, (int, float)):
                    call_bins[call_bin]["scores"].append(float(score))
                call_bins[call_bin]["calls"].append(float(calls))
                if isinstance(latency, (int, float)):
                    call_bins[call_bin]["latency_ms"].append(float(latency))

        n = len(rows)
        numeric_error_steps = [
            float(key[2:]) * count
            for key, count in first_error.items()
            if key.startswith("TQ")
        ]
        numeric_error_n = sum(count for key, count in first_error.items() if key.startswith("TQ"))
        systems[system_id] = {
            "n": n,
            "event_assessments": event_assessments,
            "path_classification": dict(path_classification),
            "valid_path_rate": (
                sum(path_classification[key] for key in ("golden_aligned", "valid_alternative")) / n
                if n
                else None
            ),
            "first_error": {key: first_error.get(key, 0) for key in (*[f"TQ{i}" for i in range(1, 7)], "no_error")},
            "mean_first_error_step": sum(numeric_error_steps) / numeric_error_n if numeric_error_n else None,
            "mean_downstream_breadth": _mean(downstream_breadth),
            "failure_incidence": {
                code: failure_incidence.get(code, 0) / n if n else None for code in FAILURE_CODES
            },
            "efficiency_dimensions": {
                dimension: _mean(efficiency_values.get(dimension, []))
                for dimension in EFFICIENCY_DIMENSIONS
            },
            "completion_modes": {
                mode: {
                    "n": len(values["scores"]),
                    "mean_score": _mean(values["scores"]),
                    "valid_path_rate": _mean(values["valid_paths"]),
                    "mean_calls": _mean(values["calls"]),
                    "mean_latency_ms": _mean(values["latency_ms"]),
                }
                for mode, values in completion_modes.items()
            },
            "score_by_first_error": {
                key: {"n": len(values), "mean_score": _mean(values)}
                for key, values in score_by_first_error.items()
            },
            "call_bins": {
                key: {
                    "n": len(values["scores"]),
                    "mean_score": _mean(values["scores"]),
                    "mean_calls": _mean(values["calls"]),
                    "mean_latency_ms": _mean(values["latency_ms"]),
                }
                for key, values in call_bins.items()
            },
        }
    return {
        "run_id": run_dir.name,
        "systems": systems,
        "d6_state": _build_d6_state_metrics(run_dir, rated_records),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_trajectory_paper_package(run_dir: Path) -> Path:
    metrics = build_trajectory_metrics(run_dir)
    systems = metrics["systems"]
    output_dir = run_dir / "paper_stage1_trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    first_error_rows = []
    failure_rows = []
    efficiency_rows = []
    completion_rows = []
    call_rows = []
    for system_id in TRAJECTORY_SYSTEMS:
        row = systems[system_id]
        summary_rows.append(
            {
                "system_id": system_id,
                "n": row["n"],
                "event_assessments": row["event_assessments"],
                "valid_path_rate": row["valid_path_rate"],
                "mean_first_error_step": row["mean_first_error_step"],
                "mean_downstream_breadth": row["mean_downstream_breadth"],
            }
        )
        for step, count in row["first_error"].items():
            score_row = row["score_by_first_error"].get(step, {})
            first_error_rows.append(
                {"system_id": system_id, "first_error_step": step, "count": count,
                 "rate": count / row["n"] if row["n"] else None,
                 "mean_score": score_row.get("mean_score")}
            )
        for code, incidence in row["failure_incidence"].items():
            failure_rows.append({"system_id": system_id, "failure_code": code, "case_incidence": incidence})
        for dimension, score in row["efficiency_dimensions"].items():
            efficiency_rows.append({"system_id": system_id, "dimension_id": dimension, "mean_score": score, "max_score": 20})
        for mode, values in row["completion_modes"].items():
            completion_rows.append({"system_id": system_id, "completion_mode": mode, **values})
        for call_bin, values in row["call_bins"].items():
            call_rows.append({"system_id": system_id, "call_bin": call_bin, **values})

    _write_csv(
        output_dir / "summary.csv",
        ["system_id", "n", "event_assessments", "valid_path_rate", "mean_first_error_step", "mean_downstream_breadth"],
        summary_rows,
    )
    _write_csv(
        output_dir / "first_error.csv",
        ["system_id", "first_error_step", "count", "rate", "mean_score"],
        first_error_rows,
    )
    _write_csv(
        output_dir / "failure_incidence.csv",
        ["system_id", "failure_code", "case_incidence"],
        failure_rows,
    )
    _write_csv(
        output_dir / "efficiency_dimensions.csv",
        ["system_id", "dimension_id", "mean_score", "max_score"],
        efficiency_rows,
    )
    _write_csv(
        output_dir / "completion_modes.csv",
        ["system_id", "completion_mode", "n", "mean_score", "valid_path_rate", "mean_calls", "mean_latency_ms"],
        completion_rows,
    )
    _write_csv(
        output_dir / "call_burden.csv",
        ["system_id", "call_bin", "n", "mean_score", "mean_calls", "mean_latency_ms"],
        call_rows,
    )

    d6_state = metrics.get("d6_state") or {}
    d6_summary_rows = []
    d6_failure_rows = []
    d6_completion_rows = []
    for group in D6_GROUPS:
        for system_id in D6_STATE_SYSTEMS:
            values = ((d6_state.get("groups") or {}).get(group) or {}).get(system_id)
            if values:
                d6_summary_rows.append(
                    {
                        "d6_group": group,
                        "topology_label": D6_GROUP_LABELS[group],
                        "system_id": system_id,
                        **values,
                    }
                )
            failure_values = ((d6_state.get("failures") or {}).get(group) or {}).get(system_id) or {}
            for code in STATE_SENSITIVE_FAILURE_CODES:
                code_values = failure_values.get(code)
                if code_values:
                    d6_failure_rows.append(
                        {
                            "d6_group": group,
                            "topology_label": D6_GROUP_LABELS[group],
                            "system_id": system_id,
                            "failure_code": code,
                            **code_values,
                        }
                    )
            mode_values = ((d6_state.get("completion_modes") or {}).get(group) or {}).get(system_id) or {}
            for mode in ("native", "recovered"):
                values = mode_values.get(mode)
                if values:
                    d6_completion_rows.append(
                        {
                            "d6_group": group,
                            "topology_label": D6_GROUP_LABELS[group],
                            "system_id": system_id,
                            "completion_mode": mode,
                            **values,
                        }
                    )

    _write_csv(
        output_dir / "d6_state_summary.csv",
        [
            "d6_group", "topology_label", "system_id", "n", "state_lineage_n",
            "mean_task_score", "mean_state_lineage_score", "state_lineage_max",
            "valid_path_rate", "mean_first_error_step", "mean_downstream_breadth",
            "state_sensitive_failure_rate", "mean_tool_interactions", "mean_latency_ms",
        ],
        d6_summary_rows,
    )
    _write_csv(
        output_dir / "d6_state_failures.csv",
        ["d6_group", "topology_label", "system_id", "failure_code", "count", "incidence"],
        d6_failure_rows,
    )
    _write_csv(
        output_dir / "d6_state_completion.csv",
        [
            "d6_group", "topology_label", "system_id", "completion_mode", "n",
            "state_lineage_n", "mean_task_score", "mean_state_lineage_score", "valid_path_rate",
        ],
        d6_completion_rows,
    )
    _write_csv(
        output_dir / "d6_state_cases.csv",
        [
            "case_id", "d6_group", "topology_label", "system_id", "task_score",
            "state_lineage_score", "state_lineage_max", "state_lineage_rubric_available",
            "path_classification", "valid_path", "completion_mode", "first_error_step",
            "first_error_event_id", "downstream_affected_steps", "state_sensitive_failure_codes",
            "root_cause", "tool_interactions", "latency_ms",
        ],
        list(d6_state.get("cases") or []),
    )

    tools = systems["tools"]
    rag = systems["tools-rag"]
    baseline = systems["baseline"]
    tools_native = tools["completion_modes"].get("native", {})
    tools_recovered = tools["completion_modes"].get("recovered", {})
    rag_native = rag["completion_modes"].get("native", {})
    rag_recovered = rag["completion_modes"].get("recovered", {})
    text = f"""# D1-D6 Trajectory 主实验结果

数据来自冻结运行 `{run_dir.name}`，未重新调用模型或 Judge。有效路径定义为 `golden_aligned` 或 `valid_alternative`。

## 核心结果

1. **Tools 将首次可观察错误推迟。** Baseline 的平均首次错误步骤为 {baseline['mean_first_error_step']:.2f}，Tools 为 {tools['mean_first_error_step']:.2f}；平均受影响下游步骤数由 {baseline['mean_downstream_breadth']:.2f} 降至 {tools['mean_downstream_breadth']:.2f}。这说明 Tools 不只是改变最终分数，也改变了错误出现和传播的位置。
2. **Tools 缓解多类推理错误，但暴露工具参数瓶颈。** 相比 Baseline，Tools 的任务分类、数值推理、约束遗漏、工程判断和过度断言发生率均下降；与此同时，`TOOL_ARGUMENT` 出现在 {tools['failure_incidence']['TOOL_ARGUMENT'] * 100:.1f}% 的 Tools 样本中，成为执行层的新主要瓶颈。Baseline 没有真实工具调用，因此该项不构成对称比较。
3. **RAG 没有改善 D1-D6 的执行轨迹。** Tools 的有效路径率为 {tools['valid_path_rate'] * 100:.1f}%，Tools + RAG 为 {rag['valid_path_rate'] * 100:.1f}%；后者在五个工具效率维度上均未超过 Tools。这与 D1-D6 属于 R0、自包含题目的设定一致。
4. **恢复完成不等于证据恢复。** Tools 原生完成均分为 {tools_native.get('mean_score', 0):.2f}，恢复完成为 {tools_recovered.get('mean_score', 0):.2f}；有效路径率分别为 {tools_native.get('valid_path_rate', 0) * 100:.1f}% 和 {tools_recovered.get('valid_path_rate', 0) * 100:.1f}%。Tools + RAG 对应均分为 {rag_native.get('mean_score', 0):.2f} 和 {rag_recovered.get('mean_score', 0):.2f}。恢复机制能够补出最终回答，但不能补回缺失的仿真证据。
5. **调用过多是失败信号，而不是收益保证。** 16 次及以上调用组的 Tools 均分为 {tools['call_bins'].get('16+', {}).get('mean_score', 0):.2f}，Tools + RAG 为 {rag['call_bins'].get('16+', {}).get('mean_score', 0):.2f}。这是描述性关联；高调用数也可能反映题目更难或搜索已经失控，不能解释成“减少调用必然提高分数”。

## 论文中可采用的表述边界

- 可以写成“基于可观察轨迹的错误传播诊断”，不要写成严格的因果干预证明。
- native 与 recovered 必须分开报告，避免把补全回答误写成完成了原始工具链。
- D7 到达后追加 Router 的路由正确率、R2 检索收益和 routing regret；无需重算本组 D1-D6 执行轨迹指标。
"""
    (output_dir / "RESULTS_CN.md").write_text(text, encoding="utf-8")

    d6_overall = (d6_state.get("groups") or {}).get("D6-all") or {}
    d6_tools = d6_overall.get("tools") or {}
    d6_rag = d6_overall.get("tools-rag") or {}
    d6_failures = (d6_state.get("failures") or {}).get("D6-all") or {}

    def _number(values: dict[str, Any], key: str) -> str:
        value = values.get(key)
        return f"{float(value):.2f}" if isinstance(value, (int, float)) else "未计算"

    def _percent(values: dict[str, Any], key: str) -> str:
        value = values.get(key)
        return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "未计算"

    state_lines = []
    for group in ("D6a", "D6b", "D6c"):
        group_values = (d6_state.get("groups") or {}).get(group) or {}
        tools_values = group_values.get("tools") or {}
        rag_values = group_values.get("tools-rag") or {}
        state_lines.append(
            f"| {group}（{D6_GROUP_LABELS[group]}） | {tools_values.get('n', 0)} | "
            f"{tools_values.get('state_lineage_n', 0)} | {_number(tools_values, 'mean_task_score')} | "
            f"{_number(tools_values, 'mean_state_lineage_score')} | {_number(rag_values, 'mean_task_score')} | "
            f"{_number(rag_values, 'mean_state_lineage_score')} |"
        )

    failure_lines = []
    for code in STATE_SENSITIVE_FAILURE_CODES:
        tools_values = (d6_failures.get("tools") or {}).get(code) or {}
        rag_values = (d6_failures.get("tools-rag") or {}).get(code) or {}
        failure_lines.append(
            f"| `{code}` | {_percent(tools_values, 'incidence')} | {_percent(rag_values, 'incidence')} |"
        )

    d6_text = f"""# D6 State与证据血缘分析

数据来自冻结运行 `{run_dir.name}`，未重新调用模型或 Judge。本分析把State理解为同一道题内固定参数、候选编号、跨工具结果、约束状态和证据来源的持续保持；它不等同于跨任务长期Memory。

## 覆盖范围

- D6共 {d6_state.get('d6_case_count', 0)} 道题，每个物理工具系统各有同样数量的评分记录。
- 其中 {d6_state.get('state_lineage_case_coverage', 0)} 道题的工具效率Rubric明确包含 `State lineage and adaptive iteration`，其得分可作为直接State-lineage指标。
- 另外 {max(0, int(d6_state.get('d6_case_count', 0)) - int(d6_state.get('state_lineage_case_coverage', 0)))} 道旧版D6b使用不同Rubric，不把它们的E3强行解释为State得分；这些题仍进入任务得分、完成方式和State相关错误统计。

## 分层结果

| D6分组 | 每系统题数 | State得分覆盖 | Tools任务分 | Tools State/20 | Tools+RAG任务分 | Tools+RAG State/20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(state_lines)}

D6总体中，Tools任务均分为 {_number(d6_tools, 'mean_task_score')}，可直接评分的State-lineage均分为 {_number(d6_tools, 'mean_state_lineage_score')}/20；Tools+RAG对应为 {_number(d6_rag, 'mean_task_score')} 和 {_number(d6_rag, 'mean_state_lineage_score')}/20。D6a、D6b、D6c同时代表不同任务子族与不同依赖结构，因此这些差异只能作描述性分层，不能解释成严格的拓扑因果效应。

## State相关错误信号（全部D6）

| 错误信号 | Tools逐题发生率 | Tools+RAG逐题发生率 |
| --- | ---: | ---: |
{chr(10).join(failure_lines)}

这些错误代码不是State的唯一或纯粹测量。例如输出遗漏也可能来自上下文耗尽，而不只是状态丢失。因此正文应称为“State相关错误信号”，并结合逐题root cause与轨迹证据解释。

## 论文可采用的结论

1. D6的复杂性不只来自调用更多工具，还来自固定输入、候选身份、跨工具结果和约束状态能否一直保持到最终决策。
2. State-lineage得分只覆盖Rubric明确命名该维度的题目，不能把所有E3机械地当成State。
3. 分层结果用于定位问题，不证明串联、并联或混合拓扑本身造成差异。
4. 本结果是现有Harness的诊断，不是 `Tools + Structured State` 干预实验，因此不能宣称新增State模块改善了性能。
5. 若未来实现结构化State，应另设与Tools配对的新run，并同时报告质量、有效路径、错误率、完成率、调用数、时延和伤害案例。
"""
    (output_dir / "D6_STATE_RESULTS_CN.md").write_text(d6_text, encoding="utf-8")
    return output_dir
