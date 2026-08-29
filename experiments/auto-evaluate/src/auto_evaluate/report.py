from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluation import load_run_profile
from .figures import export_all_figures
from .io_utils import read_json, read_jsonl, utc_now
from .judge import validate_ratings
from .research_diagnostics import build_research_metrics, render_research_diagnostics
from .reward_analysis import build_reward_analysis
from .taxonomy import FAILURE_CODE_DEFINITIONS, failure_code_label
from .trajectory import extract_observable_trajectory


COLORS = {
    "baseline": "#64748b",
    "environment": "#0f766e",
    "environment-skill": "#2563eb",
    "tools": "#0f766e",
    "tools-rag": "#0891b2",
    "gpt-5.6-teacher": "#7c3aed",
    "gpt-5.6-teacher-general": "#7c3aed",
    "gpt-5.6-teacher-tools": "#c026d3",
}

EXECUTION_ERROR_LABELS = {
    "context_window_exceeded": "\u4e0a\u4e0b\u6587\u7a97\u53e3\u8d85\u9650",
    "output_budget_exhausted": "\u8f93\u51fa\u9884\u7b97\u8017\u5c3d",
    "empty_assistant_response": "\u6a21\u578b\u672a\u8fd4\u56de\u6700\u7ec8\u56de\u7b54",
    "incomplete_response": "\u56de\u7b54\u672a\u5b8c\u6210\u6216\u7f3a\u5c11\u7ed3\u6784\u5316\u7ed3\u5c3e",
    "connection_failure": "\u8fde\u63a5\u5931\u8d25",
    "authentication_failure": "\u8eab\u4efd\u9a8c\u8bc1\u5931\u8d25",
    "invalid_request": "\u8bf7\u6c42\u53c2\u6570\u65e0\u6548",
    "upstream_execution_failure": "\u4e0a\u6e38\u6267\u884c\u5931\u8d25",
}



def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _metric(value: Any, digits: int = 1) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"


def _label_parts(identifier: str, label: Any, fallback: str) -> tuple[str, str]:
    """Return one public code and a description without repeating that code."""
    raw = str(label or "").strip()
    match = re.match(r"^([A-Za-z]+[A-Za-z0-9_-]*\d+)\b\s*[:：\-–—]?\s*(.*)$", raw)
    if match:
        return match.group(1), match.group(2).strip()
    if raw.casefold() == identifier.casefold():
        return identifier, ""
    return identifier, raw or fallback


def _tool_capability(row: dict[str, Any]) -> bool | None:
    value = row.get("tools_enabled")
    if isinstance(value, bool):
        return value
    value = (row.get("trajectory") or {}).get("tools_enabled")
    return value if isinstance(value, bool) else None


def _heat(score_pct: float) -> str:
    score_pct = max(0.0, min(100.0, score_pct))
    hue = score_pct * 1.2
    return f"hsl({hue:.0f} 58% 88%)"


def _load_response_records(run_dir: Path) -> list[dict[str, Any]]:
    response_dir = run_dir / "responses"
    return [read_json(path) for path in sorted(response_dir.glob("*.json"))] if response_dir.exists() else []


def _benchmark_metadata(run_dir: Path) -> tuple[dict[str, str], dict[str, dict[int, str]], dict[str, dict[str, str]]]:
    titles: dict[str, str] = {}
    step_labels: dict[str, dict[int, str]] = {}
    efficiency_labels: dict[str, dict[str, str]] = {}
    benchmark_dir = run_dir / "benchmarks"
    if not benchmark_dir.exists():
        return titles, step_labels, efficiency_labels
    for path in benchmark_dir.glob("*.json"):
        if path.name == "index.json":
            continue
        try:
            row = read_json(path)
        except (ValueError, OSError):
            continue
        if row.get("case_id"):
            case_id = row["case_id"]
            titles[case_id] = row.get("title") or case_id
            step_labels[case_id] = {
                int(step["step_id"]): step.get("step_label") or f"步骤 {step['step_id']}"
                for step in row.get("rubric", {}).get("steps", [])
            }
            efficiency_labels[case_id] = {
                str(item["dimension_id"]): item.get("dimension_label") or str(item["dimension_id"])
                for item in row.get("tool_efficiency_rubric", {}).get("dimensions", [])
            }
    return titles, step_labels, efficiency_labels


def _display_names(mapping: dict[str, dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in mapping.values():
        system_id = row.get("system_id")
        if system_id:
            result[system_id] = row.get("display_name") or system_id
    return result


def build_report(run_dir: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or (run_dir / "report.html")
    manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    responses = _load_response_records(run_dir)
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    mapping_doc = read_json(run_dir / "judge_mapping.json") if (run_dir / "judge_mapping.json").exists() else {"mapping": []}
    mapping = {row["task_id"]: row for row in mapping_doc.get("mapping", []) if row.get("task_id")}
    names = _display_names(mapping)
    profile = load_run_profile(run_dir) or {}
    route_spec = profile.get("adaptive_rag_analysis") or {}
    always_rag_id = route_spec.get("always_rag_system")
    adaptive_id = route_spec.get("adaptive_system")
    titles, benchmark_step_labels, benchmark_efficiency_labels = _benchmark_metadata(run_dir)
    rating_errors = validate_ratings(run_dir) if (run_dir / "judge_batch.jsonl").exists() else []
    teacher_records = read_jsonl(run_dir / "teacher_responses.jsonl")
    tool_capability_by_key: dict[tuple[str, str], bool] = {}
    tool_capabilities_by_system: defaultdict[str, set[bool]] = defaultdict(set)
    for record in [*responses, *teacher_records]:
        case_id = record.get("case_id")
        system_id = record.get("system_id")
        capability = _tool_capability(record)
        if case_id and system_id and capability is not None:
            tool_capability_by_key[(case_id, system_id)] = capability
            tool_capabilities_by_system[system_id].add(capability)

    execution_status_by_key = {
        (record.get("case_id"), record.get("system_id")): record.get("status", "unknown")
        for record in responses
        if record.get("case_id") and record.get("system_id")
    }
    execution_status_by_key.update(
        {
            (record.get("case_id"), record.get("system_id")): "success"
            for record in teacher_records
            if record.get("case_id") and record.get("system_id") and record.get("response_text")
        }
    )
    per_system: dict[str, list[float]] = defaultdict(list)
    per_success_system: dict[str, list[float]] = defaultdict(list)
    completion_by_system: defaultdict[str, Counter[str]] = defaultdict(Counter)
    per_efficiency_system: dict[str, list[float]] = defaultdict(list)
    failures: Counter[str] = Counter()
    ratings_by_case: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    responses_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rating in ratings:
        identity = mapping.get(rating.get("task_id"), {})
        system_id = identity.get("system_id", "unknown")
        display_name = identity.get("display_name", names.get(system_id, system_id))
        case_id = rating.get("case_id") or identity.get("case_id") or "unknown"
        score = float(rating.get("total_score", 0))
        per_system[system_id].append(score)
        completion_by_system[system_id]["rated"] += 1
        if execution_status_by_key.get((case_id, system_id)) == "success":
            per_success_system[system_id].append(score)
            completion_by_system[system_id]["completed"] += 1
        efficiency_score = rating.get("tool_efficiency_score")
        tool_applicable = tool_capability_by_key.get((case_id, system_id)) is not False
        if (
            tool_applicable
            and isinstance(efficiency_score, (int, float))
            and not isinstance(efficiency_score, bool)
        ):
            per_efficiency_system[system_id].append(float(efficiency_score))
        for step in rating.get("steps", []):
            failures.update(step.get("failure_codes", []))
        ratings_by_case[case_id].append((display_name, system_id, rating))
    for record in responses:
        responses_by_case[record.get("case_id") or "unknown"].append(record)
    for record in teacher_records:
        if not record.get("response_text"):
            continue
        normalized_teacher = {
            **record,
            "system_id": record.get("system_id", "gpt-5.6-teacher"),
            "display_name": record.get("display_name") or "GPT-5.6 Teacher",
            "status": "reference",
            "tools_enabled": record.get("tools_enabled"),
            "trajectory": record.get("trajectory")
            or extract_observable_trajectory(record["response_text"], tools_enabled=None),
        }
        responses_by_case[record.get("case_id") or "unknown"].append(normalized_teacher)

    research_metrics = build_research_metrics(
        ratings,
        mapping,
        responses,
        step_labels=benchmark_step_labels,
    )
    research_section = render_research_diagnostics(research_metrics)
    reward_analysis = build_reward_analysis(run_dir)
    figure_paths = export_all_figures(run_dir)
    figure_svgs = {
        figure_id: path.read_text(encoding="utf-8")
        for figure_id, path in figure_paths.items()
    }

    benchmark_diagnostic_rows = []
    for case_id in sorted(ratings_by_case):
        rows = ratings_by_case[case_id]
        model_rows = [row for row in rows if not row[1].startswith("gpt-5.6-teacher")]
        model_scores = [float(row[2].get("total_score", 0)) for row in model_rows]
        teacher_rows = [row for row in rows if row[1].startswith("gpt-5.6-teacher")]
        error_counts: Counter[str] = Counter(
            code
            for _, _, rating in model_rows
            for step in rating.get("steps", [])
            for code in step.get("failure_codes", [])
        )
        dominant = "；".join(
            f"{failure_code_label(code)}×{count}" for code, count in error_counts.most_common(3)
        ) or "无"
        valid_alternatives = sum(
            (rating.get("trajectory_analysis") or {}).get("path_classification") == "valid_alternative"
            for _, _, rating in model_rows
        )
        mean = sum(model_scores) / len(model_scores) if model_scores else 0.0
        spread = max(model_scores) - min(model_scores) if model_scores else 0.0
        teacher_score_text = " / ".join(f"{float(row[2].get('total_score', 0)):.1f}" for row in teacher_rows) or "—"
        benchmark_diagnostic_rows.append(
            f'<tr><td><strong>{_e(titles.get(case_id, case_id))}</strong><br><code>{_e(case_id)}</code></td>'
            f'<td>{mean:.1f}</td><td>{spread:.1f}</td>'
            f'<td>{_e(teacher_score_text)}</td>'
            f'<td>{_e(dominant)}</td><td>{valid_alternatives}/{len(model_rows)}</td></tr>'
        )
    benchmark_diagnostics = (
        '<section class="panel"><div class="section-kicker">Benchmark view</div><h2>Benchmark 诊断</h2>'
        '<p class="muted">从题目区分度与错误触发角度观察 benchmark；模型均分和跨度只统计待测模型，Teacher 单独作为参考。有效替代路径表示执行顺序不同于参考轨迹但证据和结论仍成立。</p>'
        '<div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>待测模型均分</th><th>模型分数跨度</th><th>Teacher 参考分</th><th>主要错误类型</th><th>有效替代路径</th></tr></thead>'
        f'<tbody>{"".join(benchmark_diagnostic_rows)}</tbody></table></div></section>'
    )

    success_count = sum(row.get("status") == "success" for row in responses)
    error_count = sum(row.get("status") == "error" for row in responses)
    recovered_count = sum(
        row.get("completion_mode") == "context_reset_finalizer" for row in responses
    )
    native_error_count = sum(
        row.get("native_status", row.get("status")) == "error" for row in responses
    )
    system_order = [
        system_id
        for system_id, _ in sorted(
            per_system.items(), key=lambda item: (-sum(item[1]) / len(item[1]), item[0])
        )
    ]

    comparison_rows = []
    comparison_labels = {
        "tools_gain": "Tools 贡献",
        "rag_gain": "RAG 贡献",
        "skills_gain": "Skills 贡献",
        "teacher_tools_gain": "Teacher 工具贡献",
    }
    for comparison in profile.get("comparisons", []):
        comparison_id = comparison.get("id")
        if comparison_id in {"rag_availability_gain", "adaptive_gain_over_tools", "adaptive_gain_over_rag"}:
            continue
        baseline_id = comparison.get("baseline_system")
        candidate_id = comparison.get("candidate_system")
        baseline_values = per_system.get(baseline_id) or []
        candidate_values = per_system.get(candidate_id) or []
        if baseline_values and candidate_values:
            baseline_mean = sum(baseline_values) / len(baseline_values)
            candidate_mean = sum(candidate_values) / len(candidate_values)
            comparison_rows.append(
                f'<tr><td><strong>{_e(comparison_labels.get(comparison_id, comparison.get("label") or comparison_id))}</strong><br>'
                f'<small>{_e(names.get(baseline_id, baseline_id))} → {_e(names.get(candidate_id, candidate_id))}</small></td>'
                f'<td>{baseline_mean:.1f}</td><td>{candidate_mean:.1f}</td>'
                f'<td class="delta {"positive" if candidate_mean >= baseline_mean else "negative"}">{candidate_mean - baseline_mean:+.1f}</td></tr>'
            )
    comparison_section = (
        '<section class="panel"><div class="section-kicker">Controlled comparisons</div>'
        '<h2>受控能力贡献</h2><p class="muted">每一行只比较配置中声明的一项能力变化；数值为端到端任务质量均分（包含执行失败或不完整回答实际获得的分数），不混入 Tool 效率分。</p>'
        '<div class="table-wrap"><table><thead><tr><th>受控比较</th><th>对照条件均分</th><th>增强条件均分</th><th>任务质量分差</th></tr></thead>'
        f'<tbody>{"".join(comparison_rows)}</tbody></table></div></section>'
        if comparison_rows else ""
    )

    mean_cards = []
    teacher_cards = []
    for system_id in system_order:
        is_teacher = system_id.startswith("gpt-5.6-teacher")
        if system_id in {always_rag_id, adaptive_id}:
            continue
        values = per_system[system_id]
        end_to_end_mean = sum(values) / len(values)
        successful_values = per_success_system.get(system_id, [])
        successful_mean = (
            sum(successful_values) / len(successful_values) if successful_values else None
        )
        rated_count = completion_by_system[system_id]["rated"]
        completed_count = completion_by_system[system_id]["completed"]
        completion_pct = completed_count / rated_count * 100 if rated_count else 0.0
        efficiency_values = per_efficiency_system.get(system_id, [])
        efficiency_mean = sum(efficiency_values) / len(efficiency_values) if efficiency_values else None
        known_capabilities = tool_capabilities_by_system.get(system_id, set())
        efficiency_summary = (
            "Tool 效率不适用（无工具权限）"
            if known_capabilities == {False}
            else f"Tool 效率 {efficiency_mean:.1f} / 100"
            if efficiency_mean is not None
            else "Tool 效率尚未评分"
        )
        successful_summary = (
            f"成功回答质量均分 {successful_mean:.1f} / 100"
            if successful_mean is not None
            else "成功回答质量均分：无成功回答"
        )
        target_cards = teacher_cards if is_teacher else mean_cards
        target_cards.append(
            f'<div class="score-card"><div class="label">{_e(names.get(system_id, system_id))}</div>'
            f'<div class="score">{end_to_end_mean:.1f}</div><div class="muted">端到端均分 / 100 · 已评分 n={len(values)}</div>'
            f'<div class="bar"><span style="width:{end_to_end_mean:.1f}%;background:{COLORS.get(system_id, "#334155")}"></span></div>'
            f'<div class="muted">{_e(successful_summary)} · 完成率 {completed_count}/{rated_count} ({completion_pct:.1f}%)</div>'
            f'<div class="muted efficiency-summary">{_e(efficiency_summary)}</div></div>'
        )
    if not mean_cards:
        mean_cards.append('<div class="notice">自动评分尚未完成。生成 <code>ratings.jsonl</code> 后重新生成报告。</div>')
    if not teacher_cards:
        teacher_cards.append('<div class="muted">当前运行没有 Teacher 参考。</div>')

    adaptive_analysis = reward_analysis.get("adaptive_rag") or {}
    rag_rows = []
    for rag_need, row in (adaptive_analysis.get("by_rag_need") or {}).items():
        skip_score = row.get("mean_score_if_skip_rag")
        use_score = row.get("mean_score_if_use_rag")
        if not isinstance(skip_score, (int, float)) or not isinstance(use_score, (int, float)):
            continue
        delta = float(use_score) - float(skip_score)
        rag_rows.append(
            f'<tr><td><strong>{_e(rag_need)}</strong></td><td>{int(row.get("n", 0))}</td>'
            f'<td>{float(skip_score):.1f}</td><td>{float(use_score):.1f}</td>'
            f'<td class="delta {"positive" if delta >= 0 else "negative"}">{delta:+.1f}</td></tr>'
        )
    rag_counterfactual_section = (
        '<section class="panel"><div class="section-kicker">RAG counterfactual</div>'
        '<h2>RAG 反事实：什么时候应该检索</h2>'
        '<p class="muted">Tools 与 Tools + RAG 是同题的两个物理分支。R0 检验无用检索的代价，R2 检验缺失知识被补充后的收益。'
        '单题差值只作为描述性证据；它不等同于训练增益，也不能在没有重复实验时归因于某一个上下文因素。</p>'
        f'<div class="paper-figure">{figure_svgs["rag-effect"]}</div>'
        '<div class="table-wrap"><table><thead><tr><th>信息需求</th><th>n</th><th>skip_rag均分</th><th>use_rag均分</th><th>RAG分差</th></tr></thead>'
        f'<tbody>{"".join(rag_rows)}</tbody></table></div></section>'
        if rag_rows else ""
    )

    route_rows = []
    for row in adaptive_analysis.get("cases", []):
        policy_score = row.get("policy_replay_score")
        independent_score = row.get("independent_adaptive_score")
        correct = row.get("policy_routing_correct")
        route_rows.append(
            f'<tr><td><code>{_e(row.get("case_id"))}</code></td><td>{_e(row.get("rag_need") or "—")}</td>'
            f'<td>{_e(row.get("expected_action") or "—")}</td><td>{_e(row.get("route_action") or "—")}</td>'
            f'<td>{_metric(policy_score)}</td>'
            f'<td>{_metric(independent_score)}</td>'
            f'<td>{_metric(row.get("routing_regret"))}</td>'
            f'<td>{_e("正确" if correct is True else "错误" if correct is False else "未标注")}</td></tr>'
        )
    router_section = (
        '<section class="panel"><div class="section-kicker">Routing policy</div>'
        '<h2>Router 策略：分支选择与离线回放</h2>'
        '<p class="muted">策略分数直接复用 Router 所选物理分支的得分和回答，默认不再次调用solver或重复Judge。'
        '若另行开展独立E2E诊断，其分数只衡量再次采样后的稳定性；routing regret始终不受随机工具轨迹或context-reset差异污染。</p>'
        f'<div class="paper-figure">{figure_svgs["router-policy"]}</div>'
        '<div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>需求</th><th>预期路由</th><th>实际路由</th><th>策略回放分</th><th>独立E2E诊断（可选）</th><th>Routing regret</th><th>策略判断</th></tr></thead>'
        f'<tbody>{"".join(route_rows)}</tbody></table></div></section>'
        if route_rows else ""
    )

    figure_section = (
        '<section class="panel"><div class="section-kicker">Scientific figures</div>'
        '<h2>多维科研图形</h2><p class="muted">报告图形和数字表使用同一份评分记录。SVG 是可编辑的矢量原图，已同时写入本次运行的 <code>figures/</code> 目录。</p>'
        f'<div class="paper-figure">{figure_svgs["quality-efficiency"]}</div>'
        f'<div class="paper-figure">{figure_svgs["reliability"]}</div></section>'
    )

    failure_rows = []
    max_failure = max(failures.values(), default=1)
    for code, count in failures.most_common():
        width = count / max_failure * 100
        failure_rows.append(
            f'<div class="failure"><span><strong>{_e(failure_code_label(code))}</strong> '
            f'<code>{_e(code)}</code></span><b>{count}</b>'
            f'<div class="mini"><i style="width:{width:.1f}%"></i></div></div>'
        )
    if not failure_rows:
        failure_rows.append('<div class="muted">暂无错误标签</div>')

    case_ids = sorted(set(ratings_by_case) | set(responses_by_case))
    case_sections = []
    for case_index, case_id in enumerate(case_ids):
        case_ratings = sorted(ratings_by_case.get(case_id, []), key=lambda row: row[0])
        case_responses = sorted(
            responses_by_case.get(case_id, []), key=lambda row: row.get("display_name", row.get("system_id", ""))
        )
        score_by_system = {system_id: float(rating.get("total_score", 0)) for _, system_id, rating in case_ratings}
        efficiency_by_system = {
            system_id: float(rating["tool_efficiency_score"])
            for _, system_id, rating in case_ratings
            if isinstance(rating.get("tool_efficiency_score"), (int, float))
            and not isinstance(rating.get("tool_efficiency_score"), bool)
        }
        summary_chips = "".join(
            f'<span class="score-chip"><i style="background:{COLORS.get(system_id, "#334155")}"></i>'
            f'{_e(names.get(system_id, system_id))} '
            f'<b>质量 {score_by_system[system_id]:.0f}</b>'
            + (
                f'<b class="efficiency-chip">效率 {efficiency_by_system[system_id]:.0f}</b>'
                if system_id in efficiency_by_system
                and tool_capability_by_key.get((case_id, system_id)) is not False
                else ""
            )
            + '</span>'
            for system_id in system_order
            if system_id in score_by_system
        )

        step_ids = sorted(
            {int(step.get("step_id", 0)) for _, _, rating in case_ratings for step in rating.get("steps", [])}
        )
        step_lookup = {
            (system_id, int(step.get("step_id", 0))): step
            for _, system_id, rating in case_ratings
            for step in rating.get("steps", [])
        }
        step_headers = "".join(
            f'<th>{_e(names.get(system_id, system_id))}</th>'
            for system_id in system_order
            if system_id in score_by_system
        )
        step_rows = []
        for step_id in step_ids:
            cells = []
            for system_id in system_order:
                if system_id not in score_by_system:
                    continue
                step = step_lookup.get((system_id, step_id))
                if not step:
                    cells.append('<td class="muted">—</td>')
                    continue
                score = float(step.get("score", 0))
                maximum = float(step.get("max_score", 0))
                pct = 0 if maximum == 0 else score / maximum * 100
                loss = max(0.0, maximum - score)
                diagnosis = step.get("diagnosis", "")
                cells.append(
                    f'<td style="background:{_heat(pct)}" title="{_e(diagnosis)}">'
                    f'<strong>{score:.1f}/{maximum:.1f}</strong>'
                    f'<small>{pct:.0f}% · 失分 {loss:.1f}</small></td>'
                )
            step_name = benchmark_step_labels.get(case_id, {}).get(step_id, f"步骤 {step_id}")
            step_code, step_description = _label_parts(f"S{step_id}", step_name, f"步骤 {step_id}")
            description_html = f'<small>{_e(step_description)}</small>' if step_description else ""
            step_rows.append(
                f'<tr><th><span class="metric-code">{_e(step_code)}</span>{description_html}</th>'
                f'{"".join(cells)}</tr>'
            )
        step_table = (
            '<p class="metric-note"><strong>任务质量 Rubric：</strong>每行对应 Excel trajectory rubric 的一个评分维度。'
            '单元格依次显示得分/满分、完成率和该维度失分；行标签直接使用源 rubric 编号，不再叠加内部 S 序号。</p>'
            f'<div class="table-wrap"><table class="step-matrix"><thead><tr><th>Rubric 维度</th>{step_headers}</tr></thead>'
            f'<tbody>{"".join(step_rows)}</tbody></table></div>'
            if step_rows
            else '<div class="muted">暂无分步评分</div>'
        )

        efficiency_ids = sorted(
            {
                str(row.get("dimension_id", ""))
                for _, _, rating in case_ratings
                for row in rating.get("tool_efficiency_dimensions", [])
                if row.get("dimension_id")
            }
        )
        efficiency_lookup = {
            (system_id, str(row.get("dimension_id", ""))): row
            for _, system_id, rating in case_ratings
            for row in rating.get("tool_efficiency_dimensions", [])
        }
        efficiency_headers = "".join(
            f'<th>{_e(names.get(system_id, system_id))}</th>'
            for system_id in system_order
            if system_id in score_by_system
        )
        efficiency_rows = []
        for dimension_id in efficiency_ids:
            cells = []
            for system_id in system_order:
                if system_id not in score_by_system:
                    continue
                if tool_capability_by_key.get((case_id, system_id)) is False:
                    cells.append('<td class="muted">不适用<br><small>无工具权限</small></td>')
                    continue
                row = efficiency_lookup.get((system_id, dimension_id))
                if not row:
                    cells.append('<td class="muted">—</td>')
                    continue
                score = float(row.get("score", 0))
                maximum = float(row.get("max_score", 0))
                pct = 0 if maximum == 0 else score / maximum * 100
                loss = max(0.0, maximum - score)
                cells.append(
                    f'<td style="background:{_heat(pct)}" title="{_e(row.get("diagnosis", ""))}">'
                    f'<strong>{score:.1f}/{maximum:.1f}</strong>'
                    f'<small>{pct:.0f}% · 失分 {loss:.1f}</small></td>'
                )
            dimension_name = benchmark_efficiency_labels.get(case_id, {}).get(dimension_id, dimension_id)
            dimension_code, dimension_description = _label_parts(
                dimension_id, dimension_name, dimension_id
            )
            description_html = f'<small>{_e(dimension_description)}</small>' if dimension_description else ""
            efficiency_rows.append(
                f'<tr><th><span class="metric-code">{_e(dimension_code)}</span>{description_html}</th>'
                f'{"".join(cells)}</tr>'
            )
        efficiency_table = (
            '<p class="metric-note"><strong>Tool 效率 Rubric：</strong>只评价可观察的工具选择、信息增益、迭代、剪枝与收敛。'
            '它与任务质量分独立；无工具权限的条件显示“不适用”，不是 0 分。</p>'
            f'<div class="table-wrap"><table class="step-matrix"><thead><tr><th>效率维度</th>{efficiency_headers}</tr></thead>'
            f'<tbody>{"".join(efficiency_rows)}</tbody></table></div>'
            if efficiency_rows
            else '<div class="muted">暂无 Tool 效率评分</div>'
        )

        overview_row_items = []
        for display, system_id, rating in case_ratings:
            tool_applicable = tool_capability_by_key.get((case_id, system_id)) is not False
            efficiency_value = rating.get("tool_efficiency_score", "—") if tool_applicable else "不适用"
            efficiency_diagnosis = (
                rating.get("tool_efficiency_overall_diagnosis", "")
                if tool_applicable
                else "该条件没有工具权限"
            )
            overview_row_items.append(
                f'<tr><td>{_e(display)}</td><td><strong>{float(rating.get("total_score", 0)):.1f}</strong></td>'
                f'<td><strong>{_e(efficiency_value)}</strong></td>'
                f'<td>{_e(rating.get("overall_diagnosis", ""))}</td>'
                f'<td>{_e(efficiency_diagnosis)}</td></tr>'
            )
        overview_rows = "".join(overview_row_items) or '<tr><td colspan="5" class="muted">暂无评分</td></tr>'

        rating_by_system = {system_id: rating for _, system_id, rating in case_ratings}
        trajectory_sections = []
        response_by_system = {record.get("system_id"): record for record in case_responses}
        for display, system_id, rating in case_ratings:
            record = response_by_system.get(system_id, {})
            observable = record.get("trajectory") or extract_observable_trajectory(
                record.get("response_text", ""),
                raw_response=record.get("raw_response"),
                tools_enabled=record.get("tools_enabled"),
                rag_enabled=record.get("rag_enabled"),
            )
            analysis = rating.get("trajectory_analysis") or {}
            assessment_lookup = {
                item.get("event_id"): item
                for item in analysis.get("event_assessments", [])
                if isinstance(item, dict) and item.get("event_id")
            }
            path_labels = {
                "golden_aligned": "与参考路径一致",
                "valid_alternative": "有效替代路径",
                "invalid": "无效路径",
                "insufficient_trace": "轨迹证据不足",
            }
            verdict_labels = {
                "correct": "正确",
                "incorrect": "错误",
                "redundant": "冗余",
                "recovered": "已恢复",
                "insufficient_evidence": "证据不足",
            }
            event_rows = []
            for event in observable.get("events", []):
                assessment = assessment_lookup.get(event.get("event_id"), {})
                event_label = event.get("tool_name") or event.get("event_type")
                codes = "；".join(
                    f"{failure_code_label(code)} ({code})"
                    for code in assessment.get("failure_codes", [])
                )
                loss = float(assessment.get("attributed_task_loss", 0) or 0)
                detail = assessment.get("diagnosis") or event.get("status") or "未评估"
                event_payload = {
                    key: event.get(key)
                    for key in ("arguments", "observation", "content_preview")
                    if event.get(key) not in (None, {}, "")
                }
                evidence_details = (
                    f'<details><summary>查看参数/结果</summary><pre>{_e(json.dumps(event_payload, ensure_ascii=False, indent=2))}</pre></details>'
                    if event_payload
                    else "—"
                )
                event_rows.append(
                    f'<tr><td><code>{_e(event.get("event_id"))}</code></td><td>{_e(event_label)}{evidence_details}</td>'
                    f'<td>{_e(verdict_labels.get(assessment.get("verdict"), assessment.get("verdict", "未评估")))}</td><td>{_e(detail)}</td>'
                    f'<td>{_e(codes or "—")}</td><td>{loss:.1f}</td></tr>'
                )
            source_labels = {
                "api_structured_and_transcript": "API 结构化事件 + 可见 transcript",
                "visible_response_transcript": "由最终可见 transcript 重建",
                "final_response_only": "仅最终回答，轨迹信息不足",
            }
            applicability = (
                "不适用：系统无工具权限"
                if record.get("tools_enabled") is False
                else "完整：可见工具轨迹已采集"
                if observable.get("summary", {}).get("tool_interactions", 0) > 0
                else "受限：系统有工具权限但未采集到可见调用"
            )
            trajectory_sections.append(
                f'<details class="inner-detail"><summary><strong>{_e(display)}</strong> · '
                f'{_e(path_labels.get(analysis.get("path_classification"), analysis.get("path_classification", "旧数据/未评估")))} · '
                f'{observable.get("summary", {}).get("tool_interactions", 0)} 次工具交互</summary>'
                f'<p><strong>轨迹来源：</strong>{_e(source_labels.get(observable.get("source"), observable.get("source")))}</p>'
                f'<p><strong>Tool 效率适用性：</strong>{_e(applicability)}</p>'
                f'<p><strong>路径摘要：</strong>{_e(analysis.get("summary", "旧评分没有轨迹分析"))}</p>'
                f'<p><strong>首个错误事件：</strong>{_e(analysis.get("first_error_event_id") or "无/轨迹不足")}；'
                f'<strong>恢复：</strong>{_e("成功" if analysis.get("recovery_succeeded") is True else "失败" if analysis.get("recovery_succeeded") is False else "未尝试或不适用")}</p>'
                f'<div class="table-wrap"><table><thead><tr><th>事件</th><th>动作/结果</th><th>判断</th><th>诊断</th><th>错误类型</th><th>归因 task loss</th></tr></thead>'
                f'<tbody>{"".join(event_rows)}</tbody></table></div></details>'
            )

        diagnosis_sections = []
        for display, system_id, rating in case_ratings:
            step_items = []
            for step in rating.get("steps", []):
                step_id = int(step.get("step_id", 0))
                raw_label = benchmark_step_labels.get(case_id, {}).get(step_id, f"步骤 {step_id}")
                step_code, step_description = _label_parts(f"S{step_id}", raw_label, f"步骤 {step_id}")
                score = float(step.get("score", 0))
                maximum = float(step.get("max_score", 0))
                codes = "；".join(
                    f"{failure_code_label(code)} ({code})"
                    for code in step.get("failure_codes", [])
                ) or "无错误标签"
                step_items.append(
                    f'<li><strong>{_e(step_code)} {_e(step_description)} · {score:.1f}/{maximum:.1f} · '
                    f'失分 {max(0.0, maximum - score):.1f}</strong>'
                    f'<div><b>诊断：</b>{_e(step.get("diagnosis", ""))}</div>'
                    f'<div><b>证据：</b>{_e(step.get("evidence", "未提供"))}</div>'
                    f'<div class="muted"><b>错误类型：</b>{_e(codes)}</div></li>'
                )
            tool_applicable = tool_capability_by_key.get((case_id, system_id)) is not False
            efficiency_items = []
            if tool_applicable:
                for row in rating.get("tool_efficiency_dimensions", []):
                    dimension_id = str(row.get("dimension_id", ""))
                    raw_label = benchmark_efficiency_labels.get(case_id, {}).get(dimension_id, dimension_id)
                    dimension_code, dimension_description = _label_parts(
                        dimension_id, raw_label, dimension_id
                    )
                    score = float(row.get("score", 0))
                    maximum = float(row.get("max_score", 0))
                    efficiency_items.append(
                        f'<li><strong>{_e(dimension_code)} {_e(dimension_description)} · '
                        f'{score:.1f}/{maximum:.1f} · 失分 {max(0.0, maximum - score):.1f}</strong>'
                        f'<div><b>诊断：</b>{_e(row.get("diagnosis", ""))}</div>'
                        f'<div><b>证据：</b>{_e(row.get("evidence", "未提供"))}</div></li>'
                    )
            step_lines = "".join(step_items)
            efficiency_lines = (
                "".join(efficiency_items)
                if tool_applicable
                else '<li class="muted">不适用：该条件没有工具权限。</li>'
            )
            suggestions = "".join(
                f'<li>{_e(item)}</li>' for item in rating.get("skill_improvement_suggestions", [])
            )
            efficiency_summary = (
                f'{_e(rating.get("tool_efficiency_score", "—"))}/100'
                if tool_applicable
                else "不适用"
            )
            diagnosis_sections.append(
                f'<details class="inner-detail"><summary><strong>{_e(display)}</strong> · '
                f'任务质量 {float(rating.get("total_score", 0)):.1f}/100 · '
                f'Tool 效率 {efficiency_summary}</summary>'
                f'<h4>任务质量：逐项 Rubric 证据</h4><ul class="diagnostic-list">{step_lines}</ul>'
                f'<p>{_e(rating.get("overall_diagnosis", ""))}</p>'
                f'<h4>Tool 效率：逐维度证据</h4><ul class="diagnostic-list">{efficiency_lines or "<li>暂无</li>"}</ul>'
                f'<p>{_e(rating.get("tool_efficiency_overall_diagnosis", "")) if tool_applicable else ""}</p>'
                f'<h4>建议干预与优化动作</h4><ul>{suggestions or "<li>无</li>"}</ul></details>'
            )

        response_sections = []
        for record in case_responses:
            status = record.get("status", "unknown")
            completion_mode = record.get("completion_mode")
            body = record.get("response_text") or record.get("error") or ""
            error_type = record.get("error_type")
            error_banner = (
                f'<p class="execution-error"><strong>&#25191;&#34892;&#22833;&#36133;&#65306;</strong>'
                f'{_e(EXECUTION_ERROR_LABELS.get(error_type, error_type))}</p>'
                if status == "error" and error_type
                else ""
            )
            recovery_banner = (
                '<p class="execution-error"><strong>上下文恢复：</strong>'
                f'{_e(record.get("native_error_type", "unknown"))} → '
                f'{_e((record.get("recovery") or {}).get("policy_version", "context reset"))}</p>'
                if completion_mode == "context_reset_finalizer"
                else ""
            )
            status_label = (
                "恢复成功"
                if completion_mode == "context_reset_finalizer"
                else "请求成功"
                if status == "success"
                else "Teacher 参考"
                if status == "reference"
                else status
            )
            response_sections.append(
                f'<details class="inner-detail"><summary><span class="pill {status}">{_e(status_label)}</span> '
                f'{_e(record.get("display_name", record.get("system_id")))} · '
                f'{_e(record.get("latency_ms", "—"))} ms</summary><pre>{_e(body)}</pre></details>'
                f'{error_banner}{recovery_banner}'
            )

        panel_id = f"benchmark-{case_index}"
        title = titles.get(case_id, case_id)
        search_value = f"{case_id} {title}".lower()
        case_sections.append(f'''
<details class="benchmark-card" data-search="{_e(search_value)}" {"open" if case_index == 0 else ""}>
  <summary><div><span class="case-number">{case_index + 1:02d}</span><span><strong>{_e(title)}</strong>
  <small>{_e(case_id)}</small></span></div><div class="summary-scores">{summary_chips}</div></summary>
  <div class="benchmark-body">
    <div class="tabs" role="tablist" aria-label="{_e(case_id)} views">
      <button class="tab active" data-target="{panel_id}-overview">概览</button>
      <button class="tab" data-target="{panel_id}-steps">任务质量 Rubric</button>
      <button class="tab" data-target="{panel_id}-efficiency">Tool 效率 Rubric</button>
      <button class="tab" data-target="{panel_id}-trajectory">轨迹与替代路径</button>
      <button class="tab" data-target="{panel_id}-diagnosis">评分证据与诊断</button>
      <button class="tab" data-target="{panel_id}-responses">原始回答</button>
    </div>
    <section id="{panel_id}-overview" class="tab-panel active"><div class="table-wrap"><table>
      <thead><tr><th>系统</th><th>任务质量</th><th>Tool 效率</th><th>任务诊断</th><th>效率诊断</th></tr></thead><tbody>{overview_rows}</tbody></table></div></section>
    <section id="{panel_id}-steps" class="tab-panel">{step_table}</section>
    <section id="{panel_id}-efficiency" class="tab-panel">{efficiency_table}</section>
    <section id="{panel_id}-trajectory" class="tab-panel"><p class="metric-note"><strong>轨迹评价：</strong>只依据可观察事件；参考路径不是唯一合法顺序。路径不同但证据和结论成立时标记为“有效替代路径”。</p>{"".join(trajectory_sections) or '<div class="muted">暂无执行轨迹</div>'}</section>
    <section id="{panel_id}-diagnosis" class="tab-panel">{"".join(diagnosis_sections) or '<div class="muted">暂无诊断</div>'}</section>
    <section id="{panel_id}-responses" class="tab-panel">{"".join(response_sections) or '<div class="muted">暂无响应</div>'}</section>
  </div>
</details>''')

    if not case_sections:
        case_sections.append('<div class="notice">暂无 benchmark 明细。</div>')

    warnings = "".join(f"<li>{_e(item)}</li>" for item in rating_errors)
    warning_block = (
        f'<section class="warning"><h2>评分数据校验问题</h2><ul>{warnings}</ul></section>' if warnings else ""
    )

    generated = utc_now()
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SWRO Auto Evaluate · {_e(run_dir.name)}</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#e4e7ec;--paper:#fff;--bg:#f5f7fa;--blue:#2563eb;--teal:#0f766e;--red:#b42318;--green:#067647}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:40px 24px 80px}}h1{{font-size:30px;margin:0 0 4px}}h2{{font-size:19px;margin:0}}h4{{margin-bottom:4px}}
.eyebrow,.section-kicker{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}}.muted{{color:var(--muted)}}
.hero,.panel,.score-card,.notice,.warning,.benchmark-card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 18px rgba(16,24,40,.04)}}
.hero{{padding:28px;margin-bottom:18px;background:linear-gradient(125deg,#fff 20%,#eef4ff)}}.meta{{display:flex;gap:20px;flex-wrap:wrap;margin-top:18px;color:var(--muted)}}
.panel{{padding:22px;margin-top:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:16px}}
.score-card{{padding:18px}}.score-card .label{{font-weight:700}}.score{{font-size:32px;font-weight:750;margin:4px 0}}.bar,.mini{{height:7px;border-radius:99px;background:#eef1f5;overflow:hidden;margin-top:12px}}.bar span,.mini i{{display:block;height:100%;border-radius:inherit}}
.delta{{font-weight:800}}.positive{{color:var(--green)}}.negative{{color:var(--red)}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}thead th{{font-size:12px;color:var(--muted);text-transform:uppercase;white-space:nowrap}}
.failure{{display:grid;grid-template-columns:minmax(190px,1fr) 40px 2fr;gap:10px;align-items:center;margin:10px 0}}.failure b{{text-align:right}}.mini{{margin:0}}.mini i{{background:var(--teal)}}
.benchmark-toolbar{{position:sticky;top:0;z-index:10;display:flex;gap:10px;flex-wrap:wrap;padding:12px 0;background:var(--bg)}}.benchmark-toolbar input{{min-width:280px;flex:1;border:1px solid var(--line);border-radius:9px;padding:10px 12px;font:inherit}}button{{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 12px;cursor:pointer;font:inherit}}button:hover{{border-color:#98a2b3}}
.benchmark-card{{margin-top:12px;overflow:hidden}}.benchmark-card>summary{{list-style:none;cursor:pointer;padding:17px 20px;display:flex;justify-content:space-between;align-items:center;gap:18px}}.benchmark-card>summary::-webkit-details-marker{{display:none}}.benchmark-card>summary>div:first-child{{display:flex;align-items:center;gap:12px;min-width:0}}.benchmark-card>summary strong{{display:block}}.benchmark-card>summary small{{display:block;color:var(--muted);font-family:ui-monospace,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.benchmark-card[open]>summary{{border-bottom:1px solid var(--line);background:#fbfcfe}}.case-number{{display:grid;place-items:center;min-width:34px;height:34px;border-radius:9px;background:#eef4ff;color:var(--blue);font-weight:800}}.summary-scores{{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}}.score-chip{{font-size:12px;background:#f2f4f7;border-radius:99px;padding:4px 8px;white-space:nowrap}}.score-chip i{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}}
.benchmark-body{{padding:18px 20px 22px}}.tabs{{display:flex;gap:6px;overflow:auto;border-bottom:1px solid var(--line);margin-bottom:14px}}.tab{{border:0;border-radius:8px 8px 0 0;background:transparent;color:var(--muted);white-space:nowrap}}.tab.active{{color:var(--blue);background:#eef4ff;font-weight:750}}.tab-panel{{display:none}}.tab-panel.active{{display:block}}.step-matrix td strong{{display:block;white-space:nowrap}}.step-matrix th small,.step-matrix td small{{color:#475467;display:block;font-weight:400;text-transform:none;max-width:260px;white-space:normal}}.metric-code{{display:block;font-weight:800}}.metric-note{{margin:0 0 12px;padding:10px 12px;border-left:3px solid var(--blue);background:#f8fafc;color:var(--muted)}}.efficiency-summary{{margin-top:9px}}.efficiency-chip{{margin-left:6px;color:#1570ef}}
.inner-detail{{border-top:1px solid var(--line);padding:12px 0}}.inner-detail:first-child{{border-top:0}}.inner-detail summary{{cursor:pointer}}pre{{white-space:pre-wrap;background:#f8fafc;border:1px solid var(--line);padding:16px;border-radius:10px;max-height:520px;overflow:auto}}.pill{{font-size:11px;padding:3px 8px;border-radius:99px;background:#eef2f6}}.pill.success{{background:#dcfce7;color:#166534}}.pill.error{{background:#fee2e2;color:#991b1b}}
.notice,.warning{{padding:18px}}.warning{{border-color:#f7c948;background:#fffbeb;margin-top:16px}}code{{background:#eef2f6;padding:2px 5px;border-radius:5px}}
.research-panel h3{{margin:24px 0 8px;font-size:16px}}.research-warning{{margin:14px 0;padding:14px 16px;border-left:4px solid #b54708;background:#fffaeb;border-radius:8px}}.loss-chip{{display:inline-block;margin:2px 5px 2px 0;padding:2px 7px;border-radius:99px;background:#fff1f3;color:#c01048;font-size:12px;white-space:nowrap}}.guide-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:14px}}.guide-card{{padding:14px;border:1px solid var(--line);border-radius:10px;background:#fbfcfe}}.guide-card strong{{display:block;margin-bottom:4px}}.diagnostic-list{{list-style:none;padding:0}}.diagnostic-list li{{padding:10px 0;border-bottom:1px solid var(--line)}}.diagnostic-list li:last-child{{border-bottom:0}}
.paper-figure{{margin-top:18px;overflow:hidden}}.paper-figure svg{{display:block;width:100%;height:auto;max-height:680px}}
@media(max-width:760px){{main{{padding:24px 14px}}.failure{{grid-template-columns:1fr 35px}}.failure .mini{{grid-column:1/-1}}.guide-grid{{grid-template-columns:1fr}}.benchmark-card>summary{{align-items:flex-start;flex-direction:column}}.summary-scores{{justify-content:flex-start}}}}
</style></head><body><main>
<section class="hero"><div class="eyebrow">SWRO AUTO EVALUATE</div><h1>运行 {_e(run_dir.name)}</h1>
<div class="muted">待测模型表现、Teacher 参考、benchmark 诊断与可观察执行轨迹</div>
<div class="meta"><span>生成时间：{_e(generated)}</span><span>Benchmark：{len(case_ids)}</span><span>待测系统最终成功：{success_count}/{len(responses)}</span><span>原生失败：{native_error_count}</span><span>恢复成功：{recovered_count}</span><span>最终失败：{error_count}</span><span>Teacher 参考：{len(teacher_records)}</span><span>有效评分：{len(ratings)}</span></div></section>
{warning_block}
<section class="panel"><div class="section-kicker">How to read this report</div><h2>报告主线与阅读顺序</h2>
<p class="muted">本报告围绕三项评测任务组织；参考答案与 rubric 用于事实和评分锚定，不强制模型复现唯一的参考步骤顺序。</p>
<div class="guide-grid">
  <div class="guide-card"><strong>1. 更细粒度 Rubric 评分</strong>先看逐题“任务质量 Rubric”和“评分证据与诊断”：每个源 rubric 维度都有得分、失分、证据、诊断和错误类型。</div>
  <div class="guide-card"><strong>2. 评分失分归因（不是训练 loss）</strong>再看“任务失分、首错与失效传播”：这里的 loss 是满分与实际得分的差，不是参数训练的优化损失，也不存在 epoch 收敛曲线。</div>
  <div class="guide-card"><strong>3. 轨迹与有效替代路径</strong>最后看逐题“轨迹与替代路径”：按真实可见事件评价工具选择、迭代、恢复和收敛；证据充分的不同路径可判为有效替代路径。</div>
</div></section>
<section class="panel"><div class="section-kicker">Primary systems</div><h2>主要待测系统</h2><p class="muted">这里只展示论文的主要求解系统；强制RAG属于反事实分支，Adaptive属于路由策略，二者不再混入同一个模型排行榜。</p><div class="grid">{"".join(mean_cards)}</div><div class="paper-figure">{figure_svgs["main-scores"]}</div></section>
<section class="panel"><div class="section-kicker">Reference roles</div><h2>Teacher 参考</h2><p class="muted">Teacher 用于提供无工具参考和强模型工具上界，不参与待测系统主排名。</p><div class="grid">{"".join(teacher_cards)}</div></section>
{benchmark_diagnostics}
{comparison_section}
{rag_counterfactual_section}
{router_section}
{figure_section}
{research_section}
<section class="panel"><div class="section-kicker">Per-benchmark evidence</div><h2>逐题 Rubric、Loss 与轨迹证据</h2>
<p class="muted">每个 benchmark 独立归组。概览回答“总体表现如何”；两个 Rubric 页回答“具体在哪一维得分或失分”；轨迹页回答“模型实际走了什么路径”；证据页给出 Judge 的逐项依据。</p>
<div class="benchmark-toolbar"><input id="benchmark-search" type="search" placeholder="按 benchmark ID 或标题筛选…" aria-label="筛选 benchmark">
<button id="expand-all" type="button">展开全部</button><button id="collapse-all" type="button">收起全部</button></div>
<div id="benchmark-list">{"".join(case_sections)}</div></section>
<p class="muted">Run manifest: {_e(manifest.get("run_id", run_dir.name))} · report schema 5.0</p>
</main><script>
document.querySelectorAll('.benchmark-card').forEach((card) => {{
  card.querySelectorAll('.tab').forEach((button) => {{
    button.addEventListener('click', () => {{
      card.querySelectorAll('.tab').forEach((item) => item.classList.remove('active'));
      card.querySelectorAll('.tab-panel').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      document.getElementById(button.dataset.target).classList.add('active');
    }});
  }});
}});
document.getElementById('benchmark-search').addEventListener('input', (event) => {{
  const query = event.target.value.trim().toLowerCase();
  document.querySelectorAll('.benchmark-card').forEach((card) => {{
    card.hidden = !card.dataset.search.includes(query);
  }});
}});
document.getElementById('expand-all').addEventListener('click', () => document.querySelectorAll('.benchmark-card:not([hidden])').forEach((card) => card.open = true));
document.getElementById('collapse-all').addEventListener('click', () => document.querySelectorAll('.benchmark-card').forEach((card) => card.open = false));
</script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
