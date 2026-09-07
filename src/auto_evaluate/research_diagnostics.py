from __future__ import annotations

import html
from collections import Counter, defaultdict
from typing import Any

from .taxonomy import FAILURE_CODE_DEFINITIONS


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def build_research_metrics(
    ratings: list[dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
    responses: list[dict[str, Any]],
    *,
    step_labels: dict[str, dict[int, str]] | None = None,
) -> dict[str, Any]:
    response_lookup = {
        (row.get("case_id"), row.get("system_id")): row
        for row in responses
        if row.get("case_id") and row.get("system_id")
    }
    step_labels = step_labels or {}

    def step_reference(case_id: str, value: Any) -> str | None:
        if value is None:
            return None
        try:
            numeric_id = int(value)
        except (TypeError, ValueError):
            return str(value)
        return step_labels.get(case_id, {}).get(numeric_id) or f"S{numeric_id}"
    systems: list[dict[str, Any]] = []
    code_step_counts: Counter[str] = Counter()
    code_responses: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    code_cases: defaultdict[str, set[str]] = defaultdict(set)
    attributed_loss: Counter[str] = Counter()
    attribution_method = "trajectory_event_primary_code"

    for rating in ratings:
        identity = mapping.get(rating.get("task_id"), {})
        system_id = identity.get("system_id", "unknown")
        display_name = identity.get("display_name", system_id)
        case_id = rating.get("case_id") or identity.get("case_id") or "unknown"
        score = float(rating.get("total_score", 0))
        steps = []
        first_score_deficit_step_id = None
        for step in rating.get("steps", []):
            earned = float(step.get("score", 0))
            maximum = float(step.get("max_score", 0))
            loss = max(0.0, maximum - earned)
            codes = list(dict.fromkeys(step.get("failure_codes", [])))
            if loss > 0 and first_score_deficit_step_id is None:
                first_score_deficit_step_id = step.get("step_id")
            for code in codes:
                code_step_counts[code] += 1
                code_responses[code].add((case_id, system_id))
                code_cases[code].add(case_id)
            steps.append(
                {
                    "step_id": step.get("step_id"),
                    "earned": earned,
                    "maximum": maximum,
                    "loss": loss,
                    "failure_codes": codes,
                    "evidence": step.get("evidence", ""),
                    "diagnosis": step.get("diagnosis", ""),
                }
            )

        trajectory = rating.get("trajectory_analysis") or {}
        assessments = trajectory.get("event_assessments") or []
        if assessments:
            for event in assessments:
                loss = float(event.get("attributed_task_loss", 0) or 0)
                if loss > 0:
                    attributed_loss[event.get("primary_failure_code") or "UNLABELED"] += loss
        else:
            attribution_method = "legacy_equal_split_fallback"
            for step in steps:
                if step["loss"] <= 0:
                    continue
                codes = step["failure_codes"] or ["UNLABELED"]
                for code in codes:
                    attributed_loss[code] += step["loss"] / len(codes)

        response = response_lookup.get((case_id, system_id), {})
        systems.append(
            {
                "case_id": case_id,
                "system_id": system_id,
                "display_name": display_name,
                "score": score,
                "evaluation_loss": max(0.0, 100.0 - score),
                "first_score_deficit_step_id": first_score_deficit_step_id,
                "first_score_deficit_label": step_reference(case_id, first_score_deficit_step_id),
                "first_causal_error_step_id": (rating.get("causal_analysis") or {}).get("first_error_step_id"),
                "first_causal_error_label": step_reference(
                    case_id,
                    (rating.get("causal_analysis") or {}).get("first_error_step_id"),
                ),
                "first_trajectory_error_event_id": trajectory.get("first_error_event_id"),
                "path_classification": trajectory.get("path_classification"),
                "steps": steps,
                "attempts": response.get("attempts"),
                "latency_ms": response.get("latency_ms"),
                "status": response.get("status", "reference" if system_id == "gpt-5.6-teacher" else "unknown"),
                "causal_analysis": rating.get("causal_analysis"),
                "trajectory_analysis": trajectory,
                "improvement_suggestions": rating.get("skill_improvement_suggestions", []),
            }
        )

    reliability: list[dict[str, Any]] = []
    responses_by_system: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        responses_by_system[response.get("system_id", "unknown")].append(response)
    for system_id, rows in sorted(responses_by_system.items()):
        latencies = sorted(
            float(row["latency_ms"])
            for row in rows
            if isinstance(row.get("latency_ms"), (int, float))
        )
        reliability.append(
            {
                "system_id": system_id,
                "display_name": rows[0].get("display_name", system_id),
                "responses": len(rows),
                "successes": sum(row.get("status") == "success" for row in rows),
                "retries": sum(max(0, int(row.get("attempts", 1)) - 1) for row in rows),
                "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
            }
        )

    error_stats = []
    all_codes = set(code_step_counts) | set(attributed_loss)
    for code in sorted(all_codes, key=lambda item: (-attributed_loss[item], item)):
        definition = FAILURE_CODE_DEFINITIONS.get(code, {})
        error_stats.append(
            {
                "code": code,
                "label_zh": definition.get("label_zh", code),
                "description_zh": definition.get("description_zh", ""),
                "step_labels": code_step_counts[code],
                "affected_responses": len(code_responses[code]),
                "affected_cases": len(code_cases[code]),
                "attributed_loss": attributed_loss[code],
            }
        )
    return {
        "systems": systems,
        "error_stats": error_stats,
        "attribution_method": attribution_method,
        "reliability": reliability,
        "response_denominator": len(ratings),
        "case_denominator": len({row["case_id"] for row in systems}),
    }


def render_research_diagnostics(metrics: dict[str, Any]) -> str:
    systems = metrics.get("systems", [])
    loss_rows = []
    causal_cards = []
    for system in sorted(systems, key=lambda row: (row["case_id"], row["display_name"])):
        score_first = system.get("first_score_deficit_label")
        causal_first = system.get("first_causal_error_label")
        event_first = system.get("first_trajectory_error_event_id")
        loss_rows.append(
            "<tr>"
            f"<td>{_e(system['case_id'])}</td><td>{_e(system['display_name'])}</td>"
            f"<td>{system['score']:.1f}</td><td><strong>{system['evaluation_loss']:.1f}</strong></td>"
            f"<td>{_e(score_first or '无')}</td>"
            f"<td>{_e(causal_first or '无')}</td>"
            f"<td>{_e(event_first or '无/轨迹不足')}</td></tr>"
        )
        structured = system.get("causal_analysis") or {}
        trajectory = system.get("trajectory_analysis") or {}
        chain = " → ".join(_e(item) for item in structured.get("error_propagation", []))
        suggestions = "".join(f"<li>{_e(item)}</li>" for item in system.get("improvement_suggestions", []))
        path_labels = {
            "golden_aligned": "与参考路径一致",
            "valid_alternative": "有效替代路径",
            "invalid": "无效路径",
            "insufficient_trace": "轨迹证据不足",
        }
        path_value = trajectory.get("path_classification", "旧数据/未评估")
        causal_cards.append(
            f'<details class="inner-detail"><summary><strong>{_e(system["display_name"])}</strong> · '
            f'{_e(system["case_id"])} · task loss {system["evaluation_loss"]:.1f}</summary>'
            f'<p><strong>路径判断：</strong>{_e(path_labels.get(path_value, path_value))}</p>'
            f'<p><strong>根因：</strong>{_e(structured.get("root_cause", "未提供"))}</p>'
            f'<p><strong>传播链：</strong>{chain or "未提供"}</p>'
            f'<p><strong>最小修复：</strong>{_e(structured.get("minimal_fix", "未提供"))}</p>'
            f'<p><strong>反事实：</strong>{_e(structured.get("counterfactual_outcome", "未提供"))}</p>'
            f'<h4>建议干预与优化动作</h4><ul>{suggestions or "<li>无</li>"}</ul></details>'
        )

    error_rows = "".join(
        "<tr>"
        f'<td><strong>{_e(row["label_zh"])}</strong><br><code>{_e(row["code"])}</code></td>'
        f'<td>{_e(row["description_zh"])}</td><td>{row["step_labels"]}</td>'
        f'<td>{row["affected_responses"]}/{metrics.get("response_denominator", 0)}</td>'
        f'<td>{row["affected_cases"]}/{metrics.get("case_denominator", 0)}</td>'
        f'<td>{row["attributed_loss"]:.2f}</td></tr>'
        for row in metrics.get("error_stats", [])
    ) or '<tr><td colspan="6" class="muted">本次运行没有已标注的错误</td></tr>'

    reliability_rows = []
    for row in metrics.get("reliability", []):
        latency = row.get("median_latency_ms")
        latency_text = f"{latency / 1000:.1f} s" if latency is not None else "—"
        reliability_rows.append(
            "<tr>"
            f'<td>{_e(row["display_name"])}</td><td>{row["successes"]}/{row["responses"]}</td>'
            f'<td>{row["retries"]}</td><td>{latency_text}</td></tr>'
        )
    reliability_html = "".join(reliability_rows) or '<tr><td colspan="4" class="muted">暂无待测系统运行记录</td></tr>'
    method_note = (
        "事件 loss 按 Judge 指定的主错误类型归因，全部事件 loss 之和严格等于 100 − 任务质量得分。"
        if metrics.get("attribution_method") == "trajectory_event_primary_code"
        else "部分旧评分没有事件级归因；这些记录暂时采用同一步多个错误码等分的兼容规则。"
    )
    return f'''
<section class="panel research-panel">
  <div class="section-kicker">Loss analysis</div><h2>任务失分、首错与失效传播</h2>
  <p class="muted"><strong>Task loss</strong> 定义为 <code>100 − 任务质量得分</code>，不是训练 loss；<strong>Tool efficiency loss</strong> 只对有工具权限的条件定义为 <code>100 − Tool 效率得分</code>。两套分数相互独立、不相加。{_e(method_note)}</p>
  <div class="research-warning"><strong>轨迹不受唯一 golden 顺序约束：</strong>参考答案和逐步 rubric 用于事实核验与评分落点；若模型采用不同顺序，但可观察证据、约束检查和最终结论成立，应标记为“有效替代路径”，而不是因为偏离参考步骤自动扣分。</div>
  <h3>评分缺口、因果首错与轨迹首错</h3>
  <p class="muted">三列回答不同问题：“首个扣分 Rubric”是最早出现分数缺口的评分维度；“最早因果错误 Rubric”是 Judge 判断的根因落点；“首个错误轨迹事件”是可观察执行记录中的实际事件 ID。它们可能相同，也可能不同。</p>
  <div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>模型/参考</th><th>任务得分</th><th>Task loss</th><th>首个扣分 Rubric</th><th>最早因果错误 Rubric</th><th>首个错误轨迹事件</th></tr></thead><tbody>{''.join(loss_rows)}</tbody></table></div>
  <h3>本次运行的总体错误分布</h3>
  <p class="muted">“步骤标签次数”是模型 × benchmark × rubric step 的标签数量，不等于独立错误事件数；“受影响响应”和“受影响题目”给出对应分母。</p>
  <div class="table-wrap"><table><thead><tr><th>错误类型</th><th>中文说明</th><th>步骤标签次数</th><th>受影响响应</th><th>受影响题目</th><th>事件归因 task loss</th></tr></thead><tbody>{error_rows}</tbody></table></div>
  <h3>失效链与最小修复</h3>{''.join(causal_cards)}
  <h3>待测系统运行可靠性</h3>
  <p class="muted">这里的“成功”仅表示请求正常返回，不代表答案正确；Teacher 参考任务不计入待测系统 API 可靠性。</p>
  <div class="table-wrap"><table><thead><tr><th>系统</th><th>请求成功</th><th>额外重试次数</th><th>中位延迟</th></tr></thead><tbody>{reliability_html}</tbody></table></div>
</section>'''