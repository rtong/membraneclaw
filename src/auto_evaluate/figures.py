from __future__ import annotations

import html
from pathlib import Path
from statistics import mean
from typing import Any

from .evaluation import load_run_profile
from .io_utils import read_json, read_jsonl
from .reward_analysis import build_reward_analysis
from .trajectory_metrics import build_trajectory_metrics


PAPER_FIGURE_IDS = (
    "main-scores",
    "tools-family-gain",
    "tools-step-gain",
    "quality-efficiency",
    "rag-effect",
    "rag-paired-outcomes",
    "rag-family-effect",
    "router-policy",
    "reliability",
    "trajectory-first-error",
    "trajectory-failure-incidence",
    "trajectory-efficiency-dimensions",
    "trajectory-recovery-quality",
    "d6-state-lineage",
    "d6-state-failures",
)

FIGURE_TITLES = {
    "main-scores": "主要系统与 Teacher 参考的任务质量",
    "tools-family-gain": "Tools 在各任务族上的配对增益",
    "tools-step-gain": "Tools 在轨迹评估步骤上的配对增益",
    "quality-efficiency": "任务质量与工具效率",
    "rag-effect": "RAG 反事实效应（按信息需求分组）",
    "rag-paired-outcomes": "强制 RAG 的逐题配对结果",
    "rag-family-effect": "强制 RAG 在各任务族上的配对效应",
    "router-policy": "Router 策略回放与独立端到端复跑",
    "reliability": "原生完成、上下文恢复与最终失败",
    "trajectory-first-error": "首次可观察错误在任务流程中的位置",
    "trajectory-failure-incidence": "主要错误类型的逐题发生率",
    "trajectory-efficiency-dimensions": "工具执行效率的五维分解",
    "trajectory-recovery-quality": "原生完成与恢复完成的证据质量",
    "d6-state-lineage": "D6任务内State保持与证据血缘",
    "d6-state-failures": "D6中的State相关错误信号",
}

PALETTE = {
    "baseline": "#64748b",
    "tools": "#0f766e",
    "tools-rag": "#0891b2",
    "solver-control": "#64748b",
    "solver-v060": "#d97706",
    "solver-v089": "#2563eb",
    "adaptive": "#2563eb",
    "teacher-general": "#7c3aed",
    "teacher-tools": "#c026d3",
    "native": "#15803d",
    "recovered": "#d97706",
    "other-success": "#60a5fa",
    "failure": "#dc2626",
    "grid": "#d0d5dd",
    "ink": "#172033",
    "muted": "#667085",
    "paper": "#ffffff",
}


def _x(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _mean_numeric(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numbers) if numbers else None


def _role(system_id: str, route_spec: dict[str, Any]) -> str:
    if system_id.startswith("gpt-5.6-teacher"):
        return "teacher"
    if system_id == route_spec.get("adaptive_system"):
        return "adaptive"
    if system_id == route_spec.get("always_rag_system"):
        return "rag_counterfactual"
    return "primary"


def _color(system_id: str, role: str) -> str:
    if system_id == "baseline":
        return PALETTE["baseline"]
    if system_id == "tools":
        return PALETTE["tools"]
    if system_id == "tools-rag":
        return PALETTE["tools-rag"]
    if system_id == "tools-rag-skill-control":
        return PALETTE["solver-control"]
    if system_id == "tools-rag-solver-v060":
        return PALETTE["solver-v060"]
    if system_id == "tools-rag-solver-v089":
        return PALETTE["solver-v089"]
    if role == "adaptive":
        return PALETTE["adaptive"]
    if system_id.endswith("teacher-tools"):
        return PALETTE["teacher-tools"]
    if role == "teacher":
        return PALETTE["teacher-general"]
    return "#475467"


def _score_rows(run_dir: Path, route_spec: dict[str, Any]) -> list[dict[str, Any]]:
    mapping_doc = (
        read_json(run_dir / "judge_mapping.json")
        if (run_dir / "judge_mapping.json").exists()
        else {"mapping": []}
    )
    mapping = {
        row.get("task_id"): row
        for row in mapping_doc.get("mapping", [])
        if isinstance(row, dict) and row.get("task_id")
    }
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    capabilities: dict[tuple[str, str], bool] = {}
    response_dir = run_dir / "responses"
    if response_dir.exists():
        for path in response_dir.glob("*.json"):
            if path.name == "index.json":
                continue
            row = read_json(path)
            if row.get("case_id") and row.get("system_id") and isinstance(row.get("tools_enabled"), bool):
                capabilities[(str(row["case_id"]), str(row["system_id"]))] = row["tools_enabled"]
    for row in read_jsonl(run_dir / "teacher_responses.jsonl"):
        trajectory = row.get("trajectory") or {}
        tools_enabled = trajectory.get("tools_enabled")
        if row.get("case_id") and row.get("system_id") and isinstance(tools_enabled, bool):
            capabilities[(str(row["case_id"]), str(row["system_id"]))] = tools_enabled

    grouped: dict[str, dict[str, Any]] = {}
    for rating in ratings:
        identity = mapping.get(rating.get("task_id"), {})
        system_id = str(identity.get("system_id") or rating.get("system_id") or "unknown")
        case_id = str(rating.get("case_id") or identity.get("case_id") or "unknown")
        row = grouped.setdefault(
            system_id,
            {
                "system_id": system_id,
                "display_name": identity.get("display_name") or system_id,
                "scores": [],
                "efficiency": [],
                "tool_capabilities": [],
            },
        )
        if isinstance(rating.get("total_score"), (int, float)):
            row["scores"].append(float(rating["total_score"]))
        capability = capabilities.get((case_id, system_id))
        if capability is not None:
            row["tool_capabilities"].append(capability)
        if capability is not False and isinstance(rating.get("tool_efficiency_score"), (int, float)):
            row["efficiency"].append(float(rating["tool_efficiency_score"]))

    rows = []
    for system_id, row in grouped.items():
        role = _role(system_id, route_spec)
        rows.append(
            {
                "system_id": system_id,
                "display_name": row["display_name"],
                "task_score": _mean_numeric(row["scores"]),
                "tool_efficiency": _mean_numeric(row["efficiency"]),
                "tools_applicable": row["tool_capabilities"] != [False]
                and not (row["tool_capabilities"] and all(value is False for value in row["tool_capabilities"])),
                "role": role,
                "color": _color(system_id, role),
                "n": len(row["scores"]),
            }
        )
    return sorted(rows, key=lambda item: (-float(item["task_score"] or 0), item["system_id"]))


def build_figure_data(run_dir: Path) -> dict[str, Any]:
    profile = load_run_profile(run_dir) or {}
    route_spec = profile.get("adaptive_rag_analysis") or {}
    analysis = build_reward_analysis(run_dir)
    case_families: dict[str, str] = {}
    for case in analysis.get("cases", []):
        case_id = case.get("case_id")
        if not case_id:
            continue
        benchmark_path = run_dir / "benchmarks" / f"{case_id}.json"
        if benchmark_path.exists():
            benchmark = read_json(benchmark_path)
            case_families[str(case_id)] = str(benchmark.get("task_family") or "unknown")
    return {
        "run_id": run_dir.name,
        "profile": profile,
        "route_spec": route_spec,
        "analysis": analysis,
        "case_families": case_families,
        "score_rows": _score_rows(run_dir, route_spec),
        "trajectory_metrics": build_trajectory_metrics(run_dir),
    }


def _svg(title: str, description: str, body: str, *, width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="figure-title figure-desc">
<title id="figure-title">{_x(title)}</title><desc id="figure-desc">{_x(description)}</desc>
<style>
text{{font-family:Arial,"Microsoft YaHei",sans-serif;fill:{PALETTE['ink']}}}.title{{font-size:24px;font-weight:700}}.subtitle{{font-size:13px;fill:{PALETTE['muted']}}}.axis{{font-size:12px;fill:{PALETTE['muted']}}}.label{{font-size:14px}}.value{{font-size:13px;font-weight:700}}.group{{font-size:12px;font-weight:700;fill:{PALETTE['muted']};letter-spacing:.04em}}.grid{{stroke:{PALETTE['grid']};stroke-width:1}}.frame{{stroke:#98a2b3;stroke-width:1;fill:none}}
</style><rect width="100%" height="100%" fill="{PALETTE['paper']}"/>{body}</svg>'''


def _empty_figure(title: str, message: str) -> str:
    body = (
        f'<text class="title" x="48" y="52">{_x(title)}</text>'
        f'<text class="subtitle" x="48" y="104">{_x(message)}</text>'
    )
    return _svg(title, message, body, width=1120, height=180)


def _axis_grid(*, x0: float, x1: float, y0: float, y1: float) -> str:
    parts = []
    for tick in range(0, 101, 20):
        x = x0 + (x1 - x0) * tick / 100
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}"/>')
        parts.append(f'<text class="axis" text-anchor="middle" x="{x:.1f}" y="{y1 + 22}">{tick}</text>')
    parts.append(f'<rect class="frame" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/>')
    return "".join(parts)


def _main_scores(data: dict[str, Any]) -> str:
    rows = [row for row in data["score_rows"] if row["role"] in {"primary", "teacher"}]
    if not rows:
        return _empty_figure(FIGURE_TITLES["main-scores"], "没有可用评分。")
    primary = [row for row in rows if row["role"] == "primary"]
    teachers = [row for row in rows if row["role"] == "teacher"]
    ordered = primary + teachers
    x0, x1 = 310, 1040
    row_gap = 54
    divider = 30 if primary and teachers else 0
    y_start = 116
    height = y_start + len(ordered) * row_gap + divider + 82
    plot_bottom = height - 58
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["main-scores"])}</text>',
        '<text class="subtitle" x="48" y="72">主要系统与参考角色分组展示；Teacher 不参与待测系统排名。</text>',
        _axis_grid(x0=x0, x1=x1, y0=94, y1=plot_bottom),
    ]
    y = y_start
    if primary:
        body.append(f'<text class="group" x="48" y="{y - 12}">主要待测系统</text>')
    for index, row in enumerate(ordered):
        if index == len(primary) and teachers:
            y += divider
            body.append(f'<line class="grid" x1="48" y1="{y - 22}" x2="1040" y2="{y - 22}"/>')
            body.append(f'<text class="group" x="48" y="{y - 6}">TEACHER 参考</text>')
        score = float(row["task_score"] or 0)
        width = (x1 - x0) * score / 100
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 16}" y="{y + 17}">{_x(row["display_name"])}</text>')
        body.append(f'<rect x="{x0}" y="{y}" width="{width:.1f}" height="24" fill="{row["color"]}"/>')
        body.append(f'<text class="value" x="{min(x0 + width + 9, x1 - 4):.1f}" y="{y + 17}" text-anchor="{ "end" if x0 + width + 48 > x1 else "start" }">{score:.1f}</text>')
        y += row_gap
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 14}">端到端任务质量分（0–100）</text>')
    return _svg(FIGURE_TITLES["main-scores"], "主要系统与Teacher参考的平均任务质量分。", "".join(body), width=1120, height=height)


def _tools_family_gain(data: dict[str, Any]) -> str:
    grouped: dict[str, list[float]] = {}
    for case in data["analysis"].get("cases", []):
        systems = case.get("systems") or {}
        baseline = (systems.get("baseline") or {}).get("task_reward")
        tools = (systems.get("tools") or {}).get("task_reward")
        if not isinstance(baseline, (int, float)) or not isinstance(tools, (int, float)):
            continue
        case_id = str(case.get("case_id") or "")
        family = data["case_families"].get(case_id, "unknown")
        grouped.setdefault(family, []).append(float(tools) - float(baseline))
    rows = [
        (family, mean(values), len(values))
        for family, values in sorted(grouped.items())
        if values
    ]
    if not rows:
        return _empty_figure(FIGURE_TITLES["tools-family-gain"], "缺少Baseline与Tools的成对评分。")

    width = 1120
    y_start, row_gap = 112, 38
    height = y_start + len(rows) * row_gap + 82
    x0, x1 = 245, 1035
    min_gain = min(0.0, min(value for _, value, _ in rows))
    max_gain = max(0.0, max(value for _, value, _ in rows))
    lower = min(-10.0, 10.0 * int(min_gain // 10)) if min_gain < 0 else 0.0
    upper = max(10.0, 10.0 * int((max_gain + 9.999) // 10))
    span = upper - lower
    zero_x = x0 + (x1 - x0) * (0 - lower) / span
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["tools-family-gain"])}</text>',
        '<text class="subtitle" x="48" y="72">每个任务族先对同题计算 Tools − Baseline，再对该任务族取平均；n为题目数。</text>',
    ]
    for index in range(5):
        value = lower + span * index / 4
        x = x0 + (x1 - x0) * index / 4
        body.append(f'<line class="grid" x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="{height - 54}"/>')
        body.append(f'<text class="axis" text-anchor="middle" x="{x:.1f}" y="{height - 32}">{value:.0f}</text>')
    body.append(f'<line x1="{zero_x:.1f}" y1="90" x2="{zero_x:.1f}" y2="{height - 54}" stroke="#667085" stroke-width="1.5"/>')
    for index, (family, gain, count) in enumerate(rows):
        y = y_start + index * row_gap
        gain_x = x0 + (x1 - x0) * (gain - lower) / span
        left = min(zero_x, gain_x)
        bar_width = max(1.0, abs(gain_x - zero_x))
        color = PALETTE["tools"] if gain >= 0 else PALETTE["failure"]
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 14}" y="{y + 16}">{_x(family)} · n={count}</text>')
        body.append(f'<rect x="{left:.1f}" y="{y}" width="{bar_width:.1f}" height="22" fill="{color}"/>')
        anchor = "start" if gain >= 0 else "end"
        label_x = gain_x + 8 if gain >= 0 else gain_x - 8
        body.append(f'<text class="value" text-anchor="{anchor}" x="{label_x:.1f}" y="{y + 16}">{gain:+.1f}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 8}">任务质量配对增益（Tools − Baseline）</text>')
    return _svg(
        FIGURE_TITLES["tools-family-gain"],
        "WaterTAP Tools在各任务族上的同题配对任务质量增益。",
        "".join(body),
        width=width,
        height=height,
    )


def _tools_step_gain(data: dict[str, Any]) -> str:
    comparison = (data["analysis"].get("comparisons") or {}).get("tools_gain") or {}
    gains = comparison.get("mean_step_gains") or {}
    labels = {
        "1": "TQ1 问题理解",
        "2": "TQ2 规划与分解",
        "3": "TQ3 工具调用正确性",
        "4": "TQ4 结果解释",
        "5": "TQ5 比较与重新规划",
        "6": "TQ6 最终工程决策",
    }
    rows = [
        (labels[key], float(gains[key]))
        for key in labels
        if isinstance(gains.get(key), (int, float))
    ]
    if not rows:
        return _empty_figure(FIGURE_TITLES["tools-step-gain"], "缺少Tools配对步骤增益。")

    width, height = 1120, 430
    x0, x1 = 320, 1035
    y_start, row_gap = 112, 46
    max_gain = max(value for _, value in rows)
    upper = max(2.0, 2.0 * int((max_gain + 1.999) // 2))
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["tools-step-gain"])}</text>',
        '<text class="subtitle" x="48" y="72">各步骤分别对含该步骤的同题评分取平均；TQ1–TQ5 n=117，TQ6 n=103。</text>',
    ]
    for index in range(6):
        value = upper * index / 5
        x = x0 + (x1 - x0) * index / 5
        body.append(f'<line class="grid" x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="{height - 54}"/>')
        body.append(f'<text class="axis" text-anchor="middle" x="{x:.1f}" y="{height - 32}">{value:.1f}</text>')
    for index, (label, gain) in enumerate(rows):
        y = y_start + index * row_gap
        bar_width = (x1 - x0) * gain / upper
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 14}" y="{y + 17}">{_x(label)}</text>')
        body.append(f'<rect x="{x0}" y="{y}" width="{bar_width:.1f}" height="24" fill="{PALETTE["tools"]}"/>')
        body.append(f'<text class="value" x="{x0 + bar_width + 8:.1f}" y="{y + 17}">+{gain:.2f}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 8}">平均步骤分增益（Tools − Baseline）</text>')
    return _svg(
        FIGURE_TITLES["tools-step-gain"],
        "WaterTAP Tools在六个轨迹评估步骤上的同题配对平均增益。",
        "".join(body),
        width=width,
        height=height,
    )


def _quality_efficiency(data: dict[str, Any]) -> str:
    rows = [
        row for row in data["score_rows"]
        if row["task_score"] is not None and row["tool_efficiency"] is not None
    ]
    if not rows:
        return _empty_figure(FIGURE_TITLES["quality-efficiency"], "没有同时具备任务质量和工具效率的数据。")
    width, height = 1120, 650
    x0, x1, y0, y1 = 110, 1010, 105, 550
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["quality-efficiency"])}</text>',
        '<text class="subtitle" x="48" y="72">右上区域代表任务结论与工具轨迹同时较好；该图不是训练 loss 曲线。</text>',
    ]
    for tick in range(0, 101, 20):
        x = x0 + (x1 - x0) * tick / 100
        y = y1 - (y1 - y0) * tick / 100
        body.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}"/>',
                f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>',
                f'<text class="axis" text-anchor="middle" x="{x:.1f}" y="{y1 + 22}">{tick}</text>',
                f'<text class="axis" text-anchor="end" x="{x0 - 12}" y="{y + 4:.1f}">{tick}</text>',
            ]
        )
    body.append(f'<rect class="frame" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/>')
    for index, row in enumerate(rows):
        px = x0 + (x1 - x0) * float(row["task_score"]) / 100
        py = y1 - (y1 - y0) * float(row["tool_efficiency"]) / 100
        marker = (
            f'<rect x="{px - 7:.1f}" y="{py - 7:.1f}" width="14" height="14" fill="{row["color"]}"/>'
            if row["role"] == "teacher"
            else f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{row["color"]}"/>'
        )
        label_y = py - 12 if index % 2 == 0 else py + 24
        anchor = "end" if px > x1 - 220 else "start"
        label_x = px - 10 if anchor == "end" else px + 10
        body.append(marker)
        display_name = (
            f'{row["display_name"]}（独立E2E复跑）'
            if row["role"] == "adaptive"
            else row["display_name"]
        )
        body.append(f'<text class="label" text-anchor="{anchor}" x="{label_x:.1f}" y="{label_y:.1f}">{_x(display_name)} ({row["task_score"]:.0f}, {row["tool_efficiency"]:.0f})</text>')
    body.extend(
        [
            f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 28}">任务质量分（0–100）</text>',
            f'<text class="axis" text-anchor="middle" transform="rotate(-90 30 {(y0 + y1) / 2:.1f})" x="30" y="{(y0 + y1) / 2:.1f}">工具效率分（0–100）</text>',
        ]
    )
    return _svg(FIGURE_TITLES["quality-efficiency"], "各工具可用系统的任务质量与工具效率散点图。", "".join(body), width=width, height=height)


def _rag_effect(data: dict[str, Any]) -> str:
    grouped = (data["analysis"].get("adaptive_rag") or {}).get("by_rag_need") or {}
    rows = [
        (rag_need, values)
        for rag_need, values in grouped.items()
        if isinstance(values.get("mean_score_if_skip_rag"), (int, float))
        and isinstance(values.get("mean_score_if_use_rag"), (int, float))
    ]
    if not rows:
        return _empty_figure(FIGURE_TITLES["rag-effect"], "缺少成对的 Tools 与 Tools + RAG 评分。")
    width, height = 1120, 520
    x0, x1, y0, y1 = 110, 1040, 110, 400
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["rag-effect"])}</text>',
        '<text class="subtitle" x="48" y="72">同一题分别强制 skip_rag 与 use_rag；差值是描述性反事实，不等同于训练增益。</text>',
    ]
    for tick in range(0, 101, 20):
        y = y1 - (y1 - y0) * tick / 100
        body.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        body.append(f'<text class="axis" text-anchor="end" x="{x0 - 12}" y="{y + 4:.1f}">{tick}</text>')
    group_width = (x1 - x0) / len(rows)
    bar_width = min(90, group_width * 0.24)
    for index, (rag_need, values) in enumerate(rows):
        center = x0 + group_width * (index + 0.5)
        skip = float(values["mean_score_if_skip_rag"])
        use = float(values["mean_score_if_use_rag"])
        for offset, score, color, label in (
            (-bar_width * 0.58, skip, PALETTE["tools"], "skip_rag"),
            (bar_width * 0.58, use, PALETTE["tools-rag"], "use_rag"),
        ):
            top = y1 - (y1 - y0) * score / 100
            body.append(f'<rect x="{center + offset - bar_width / 2:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{y1 - top:.1f}" fill="{color}"/>')
            body.append(f'<text class="value" text-anchor="middle" x="{center + offset:.1f}" y="{top - 8:.1f}">{score:.1f}</text>')
            body.append(f'<text class="axis" text-anchor="middle" x="{center + offset:.1f}" y="{y1 + 20}">{label}</text>')
        body.append(f'<text class="label" text-anchor="middle" x="{center:.1f}" y="{y1 + 48}">{_x(rag_need)} · n={int(values.get("n", 0))}</text>')
        body.append(f'<text class="value" text-anchor="middle" x="{center:.1f}" y="{y0 - 14}">RAG Δ {use - skip:+.1f}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 20}">端到端任务质量分（0–100）</text>')
    return _svg(FIGURE_TITLES["rag-effect"], "R0和R2分组的skip_rag与use_rag成对平均得分。", "".join(body), width=width, height=height)


def _rag_paired_outcomes(data: dict[str, Any]) -> str:
    gains = []
    for case in data["analysis"].get("cases", []):
        systems = case.get("systems") or {}
        tools = (systems.get("tools") or {}).get("task_reward")
        tools_rag = (systems.get("tools-rag") or {}).get("task_reward")
        if isinstance(tools, (int, float)) and isinstance(tools_rag, (int, float)):
            gains.append(float(tools_rag) - float(tools))
    if not gains:
        return _empty_figure(FIGURE_TITLES["rag-paired-outcomes"], "缺少成对的Tools与Tools + RAG评分。")

    comparison = (data["analysis"].get("comparisons") or {}).get("rag_availability_gain") or {}
    mean_gain = float(comparison.get("mean_total_gain", mean(gains)))
    ci = comparison.get("mean_total_gain_ci95") or []
    ci_text = (
        f'95% CI [{float(ci[0]):+.2f}, {float(ci[1]):+.2f}]'
        if len(ci) == 2 and all(isinstance(value, (int, float)) for value in ci)
        else "95% CI —"
    )
    rows = [
        ("RAG更高", sum(value > 0 for value in gains), PALETTE["tools-rag"]),
        ("同分", sum(value == 0 for value in gains), PALETTE["muted"]),
        ("RAG更低", sum(value < 0 for value in gains), PALETTE["failure"]),
    ]
    width, height = 1120, 350
    x0, x1 = 260, 1035
    y_start, row_gap = 126, 58
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["rag-paired-outcomes"])}</text>',
        f'<text class="subtitle" x="48" y="72">Tools + RAG − Tools 平均 {mean_gain:+.2f}分 · {_x(ci_text)} · n={len(gains)}</text>',
        _axis_grid(x0=x0, x1=x1, y0=96, y1=height - 58),
    ]
    for index, (label, count, color) in enumerate(rows):
        y = y_start + index * row_gap
        bar_width = (x1 - x0) * count / len(gains)
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 16}" y="{y + 18}">{_x(label)}</text>')
        body.append(f'<rect x="{x0}" y="{y}" width="{bar_width:.1f}" height="26" fill="{color}"/>')
        body.append(f'<text class="value" x="{x0 + bar_width + 9:.1f}" y="{y + 18}">{count}题（{count / len(gains) * 100:.1f}%）</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 12}">占全部成对题目的比例（0–100%）</text>')
    return _svg(
        FIGURE_TITLES["rag-paired-outcomes"],
        "强制RAG相对Tools在逐题配对评分中的正向、持平和负向数量，并报告平均效应置信区间。",
        "".join(body),
        width=width,
        height=height,
    )


def _rag_family_effect(data: dict[str, Any]) -> str:
    grouped: dict[str, list[float]] = {}
    for case in data["analysis"].get("cases", []):
        systems = case.get("systems") or {}
        tools = (systems.get("tools") or {}).get("task_reward")
        tools_rag = (systems.get("tools-rag") or {}).get("task_reward")
        if not isinstance(tools, (int, float)) or not isinstance(tools_rag, (int, float)):
            continue
        case_id = str(case.get("case_id") or "")
        family = data["case_families"].get(case_id, "unknown")
        grouped.setdefault(family, []).append(float(tools_rag) - float(tools))
    rows = [(family, mean(values), len(values)) for family, values in sorted(grouped.items()) if values]
    if not rows:
        return _empty_figure(FIGURE_TITLES["rag-family-effect"], "缺少成对的Tools与Tools + RAG评分。")

    width = 1120
    y_start, row_gap = 112, 38
    height = y_start + len(rows) * row_gap + 82
    x0, x1 = 245, 1035
    min_gain = min(0.0, min(value for _, value, _ in rows))
    max_gain = max(0.0, max(value for _, value, _ in rows))
    lower = 5.0 * int(min_gain // 5) if min_gain < 0 else 0.0
    upper = max(5.0, 5.0 * int((max_gain + 4.999) // 5))
    span = upper - lower
    zero_x = x0 + (x1 - x0) * (0 - lower) / span
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["rag-family-effect"])}</text>',
        '<text class="subtitle" x="48" y="72">每个任务族先计算同题 Tools + RAG − Tools，再对该任务族取平均；正负效应并存。</text>',
    ]
    for index in range(5):
        value = lower + span * index / 4
        x = x0 + (x1 - x0) * index / 4
        body.append(f'<line class="grid" x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="{height - 54}"/>')
        body.append(f'<text class="axis" text-anchor="middle" x="{x:.1f}" y="{height - 32}">{value:.0f}</text>')
    body.append(f'<line x1="{zero_x:.1f}" y1="90" x2="{zero_x:.1f}" y2="{height - 54}" stroke="#667085" stroke-width="1.5"/>')
    for index, (family, gain, count) in enumerate(rows):
        y = y_start + index * row_gap
        gain_x = x0 + (x1 - x0) * (gain - lower) / span
        left = min(zero_x, gain_x)
        bar_width = max(1.0, abs(gain_x - zero_x))
        color = PALETTE["tools-rag"] if gain >= 0 else PALETTE["failure"]
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 14}" y="{y + 16}">{_x(family)} · n={count}</text>')
        body.append(f'<rect x="{left:.1f}" y="{y}" width="{bar_width:.1f}" height="22" fill="{color}"/>')
        anchor = "start" if gain >= 0 else "end"
        label_x = gain_x + 8 if gain >= 0 else gain_x - 8
        body.append(f'<text class="value" text-anchor="{anchor}" x="{label_x:.1f}" y="{y + 16}">{gain:+.1f}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 8}">任务质量配对效应（Tools + RAG − Tools）</text>')
    return _svg(
        FIGURE_TITLES["rag-family-effect"],
        "强制RAG在各任务族上的同题配对任务质量效应。",
        "".join(body),
        width=width,
        height=height,
    )


def _router_policy(data: dict[str, Any]) -> str:
    adaptive = data["analysis"].get("adaptive_rag") or {}
    cases = [row for row in adaptive.get("cases", []) if row.get("policy_replay_score") is not None]
    if not cases:
        return _empty_figure(FIGURE_TITLES["router-policy"], "缺少可回放的Router分支评分。")
    policy = mean(float(row["policy_replay_score"]) for row in cases)
    optimal = mean(max(float(row["score_if_skip_rag"]), float(row["score_if_use_rag"])) for row in cases)
    independent_values = [float(row["independent_adaptive_score"]) for row in cases if row.get("independent_adaptive_score") is not None]
    bars = [
        ("Router策略回放", policy, PALETTE["adaptive"]),
        ("逐题最优物理分支", optimal, PALETTE["tools"]),
    ]
    if independent_values:
        bars.append(("独立E2E复跑（稳定性）", mean(independent_values), PALETTE["baseline"]))
    width, height = 1120, 400
    x0, x1 = 330, 1040
    y0, row_gap = 118, 66
    accuracy = adaptive.get("policy_routing_accuracy")
    regret = adaptive.get("mean_routing_regret")
    accuracy_text = "—" if not isinstance(accuracy, (int, float)) else f"{float(accuracy) * 100:.1f}%"
    regret_text = "—" if not isinstance(regret, (int, float)) else f"{float(regret):.1f}"
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["router-policy"])}</text>',
        f'<text class="subtitle" x="48" y="72">策略正确率 {_x(accuracy_text)} · 平均routing regret {_x(regret_text)} · n={len(cases)}</text>',
        _axis_grid(x0=x0, x1=x1, y0=94, y1=height - 60),
    ]
    for index, (label, score, color) in enumerate(bars):
        y = y0 + index * row_gap
        bar_width = (x1 - x0) * score / 100
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 16}" y="{y + 18}">{_x(label)}</text>')
        body.append(f'<rect x="{x0}" y="{y}" width="{bar_width:.1f}" height="26" fill="{color}"/>')
        body.append(f'<text class="value" x="{min(x0 + bar_width + 9, x1 - 4):.1f}" y="{y + 18}" text-anchor="{ "end" if x0 + bar_width + 45 > x1 else "start" }">{score:.1f}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 16}">任务质量分（0–100）</text>')
    return _svg(FIGURE_TITLES["router-policy"], "Router选择分支的离线策略回放分数、逐题最优分支和可选独立端到端诊断分数。", "".join(body), width=width, height=height)


def _reliability(data: dict[str, Any]) -> str:
    systems = data["analysis"].get("systems") or {}
    rows = []
    score_names = {row["system_id"]: row["display_name"] for row in data["score_rows"]}
    for system_id, metrics in systems.items():
        if not isinstance(metrics.get("completion_rate"), (int, float)):
            continue
        native = float(metrics.get("native_completion_rate") or 0)
        recovered = float(metrics.get("recovery_rate") or 0)
        completed = float(metrics.get("completion_rate") or 0)
        other_success = max(0.0, completed - native - recovered)
        failure = max(0.0, 1.0 - completed)
        rows.append((score_names.get(system_id, system_id), native, recovered, other_success, failure))
    if not rows:
        return _empty_figure(FIGURE_TITLES["reliability"], "没有系统运行可靠性数据。")
    width = 1120
    y_start, row_gap = 130, 55
    height = y_start + len(rows) * row_gap + 75
    x0, x1 = 300, 1040
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["reliability"])}</text>',
        '<text class="subtitle" x="48" y="72">完成率分解为原生完成、context-reset恢复、本地策略回放/其他成功与最终失败。</text>',
        f'<rect class="frame" x="{x0}" y="104" width="{x1 - x0}" height="{len(rows) * row_gap}"/>',
    ]
    legend_x = 48
    for label, color in (
        ("原生完成", PALETTE["native"]),
        ("上下文恢复", PALETTE["recovered"]),
        ("策略回放/其他", PALETTE["other-success"]),
        ("最终失败", PALETTE["failure"]),
    ):
        body.append(f'<rect x="{legend_x}" y="88" width="12" height="12" fill="{color}"/><text class="axis" x="{legend_x + 18}" y="99">{label}</text>')
        legend_x += 120
    for index, (label, native, recovered, other_success, failure) in enumerate(rows):
        y = y_start + index * row_gap
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 16}" y="{y + 17}">{_x(label)}</text>')
        cursor = x0
        for value, color in (
            (native, PALETTE["native"]),
            (recovered, PALETTE["recovered"]),
            (other_success, PALETTE["other-success"]),
            (failure, PALETTE["failure"]),
        ):
            segment = (x1 - x0) * value
            if segment > 0:
                body.append(f'<rect x="{cursor:.1f}" y="{y}" width="{segment:.1f}" height="24" fill="{color}"/>')
                if segment >= 44:
                    body.append(f'<text x="{cursor + segment / 2:.1f}" y="{y + 17}" text-anchor="middle" font-size="12" fill="#ffffff">{value * 100:.0f}%</text>')
            cursor += segment
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 18}">运行比例（0–100%）</text>')
    return _svg(FIGURE_TITLES["reliability"], "各系统的原生完成、恢复完成、本地策略回放和最终失败比例。", "".join(body), width=width, height=height)


def _trajectory_first_error(data: dict[str, Any]) -> str:
    systems = (data.get("trajectory_metrics") or {}).get("systems") or {}
    rows = [(system_id, systems.get(system_id)) for system_id in ("baseline", "tools", "tools-rag")]
    rows = [(system_id, row) for system_id, row in rows if row and row.get("n")]
    if not rows:
        return _empty_figure(FIGURE_TITLES["trajectory-first-error"], "缺少Trajectory评分。")
    colors = ("#dc2626", "#ea580c", "#d97706", "#65a30d", "#0f766e", "#2563eb", "#94a3b8")
    steps = (*[f"TQ{i}" for i in range(1, 7)], "no_error")
    labels = {"baseline": "Baseline", "tools": "Tools", "tools-rag": "Tools + RAG"}
    width, height = 1120, 350
    x0, x1 = 270, 1040
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["trajectory-first-error"])}</text>',
        '<text class="subtitle" x="48" y="72">越靠右表示错误越晚出现；条形按每个系统的117道题归一化。</text>',
    ]
    legend_x = 48
    for step, color in zip(steps, colors):
        label = "无错误" if step == "no_error" else step
        body.append(f'<rect x="{legend_x}" y="91" width="12" height="12" fill="{color}"/><text class="axis" x="{legend_x + 17}" y="102">{label}</text>')
        legend_x += 82
    for index, (system_id, row) in enumerate(rows):
        y = 138 + index * 62
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 16}" y="{y + 18}">{labels[system_id]}</text>')
        cursor = x0
        for step, color in zip(steps, colors):
            rate = float(row["first_error"].get(step, 0)) / float(row["n"])
            segment = (x1 - x0) * rate
            if segment:
                body.append(f'<rect x="{cursor:.1f}" y="{y}" width="{segment:.1f}" height="27" fill="{color}"/>')
                if segment >= 48:
                    body.append(f'<text x="{cursor + segment / 2:.1f}" y="{y + 18}" text-anchor="middle" font-size="11" fill="#ffffff">{rate * 100:.0f}%</text>')
            cursor += segment
        mean_step = row.get("mean_first_error_step")
        if isinstance(mean_step, (int, float)):
            body.append(f'<text class="axis" x="{x1}" y="{y + 45}" text-anchor="end">平均首次错误步骤 {mean_step:.2f}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 16}">题目比例（0–100%）</text>')
    return _svg(FIGURE_TITLES["trajectory-first-error"], "三个系统首次可观察错误步骤的归一化分布。", "".join(body), width=width, height=height)


def _trajectory_failure_incidence(data: dict[str, Any]) -> str:
    systems = (data.get("trajectory_metrics") or {}).get("systems") or {}
    if not all(systems.get(system_id) for system_id in ("baseline", "tools", "tools-rag")):
        return _empty_figure(FIGURE_TITLES["trajectory-failure-incidence"], "缺少三组系统的错误标签。")
    codes = (
        "TASK_CLASSIFICATION", "PARAMETER_EXTRACTION", "TOOL_ARGUMENT", "TOOL_NOT_CALLED",
        "NUMERICAL_REASONING", "CONSTRAINT_OMISSION", "ENGINEERING_JUDGMENT", "OVERCLAIM",
    )
    labels = {
        "TASK_CLASSIFICATION": "任务分类", "PARAMETER_EXTRACTION": "参数提取",
        "TOOL_ARGUMENT": "工具参数", "TOOL_NOT_CALLED": "所需工具证据缺失",
        "NUMERICAL_REASONING": "数值推理", "CONSTRAINT_OMISSION": "约束遗漏",
        "ENGINEERING_JUDGMENT": "工程判断", "OVERCLAIM": "过度断言",
    }
    width, height = 1120, 570
    x0, x1, y0, gap = 300, 1040, 132, 50
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["trajectory-failure-incidence"])}</text>',
        '<text class="subtitle" x="48" y="72">每个点表示该错误至少在一道题的一个评分步骤中出现；Baseline的工具参数项不作对称能力解释。</text>',
        _axis_grid(x0=x0, x1=x1, y0=104, y1=y0 + (len(codes) - 1) * gap + 25),
    ]
    legend_x = 48
    for system_id, label in (("baseline", "Baseline"), ("tools", "Tools"), ("tools-rag", "Tools + RAG")):
        color = PALETTE[system_id]
        body.append(f'<circle cx="{legend_x + 6}" cy="94" r="6" fill="{color}"/><text class="axis" x="{legend_x + 18}" y="98">{label}</text>')
        legend_x += 130
    for index, code in enumerate(codes):
        y = y0 + index * gap
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 18}" y="{y + 5}">{labels[code]}</text>')
        for offset, system_id in zip((-7, 0, 7), ("baseline", "tools", "tools-rag")):
            rate = systems[system_id]["failure_incidence"].get(code)
            if isinstance(rate, (int, float)):
                x = x0 + (x1 - x0) * float(rate)
                body.append(f'<circle cx="{x:.1f}" cy="{y + offset}" r="5.5" fill="{PALETTE[system_id]}"/>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 18}">逐题发生率（%）</text>')
    return _svg(FIGURE_TITLES["trajectory-failure-incidence"], "Baseline、Tools和Tools加RAG主要错误类型的逐题发生率。", "".join(body), width=width, height=height)


def _trajectory_efficiency_dimensions(data: dict[str, Any]) -> str:
    systems = (data.get("trajectory_metrics") or {}).get("systems") or {}
    rows = [("tools", systems.get("tools")), ("tools-rag", systems.get("tools-rag"))]
    if not all(row for _, row in rows):
        return _empty_figure(FIGURE_TITLES["trajectory-efficiency-dimensions"], "缺少工具效率维度评分。")
    labels = {"E1": "工具适配", "E2": "调用规划", "E3": "结果驱动迭代", "E4": "冗余控制", "E5": "证据充分与停止"}
    width, height = 1120, 440
    x0, x1, y0, gap = 330, 1040, 128, 56
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["trajectory-efficiency-dimensions"])}</text>',
        '<text class="subtitle" x="48" y="72">每维满分20；D1-D6中RAG未带来更高的工具执行效率。</text>',
    ]
    for tick in range(0, 21, 5):
        x = x0 + (x1 - x0) * tick / 20
        body.append(f'<line class="grid" x1="{x:.1f}" y1="100" x2="{x:.1f}" y2="{y0 + 4 * gap + 34}"/><text class="axis" text-anchor="middle" x="{x:.1f}" y="{height - 20}">{tick}</text>')
    for index, dimension in enumerate(("E1", "E2", "E3", "E4", "E5")):
        y = y0 + index * gap
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 18}" y="{y + 11}">{dimension} {labels[dimension]}</text>')
        for offset, (system_id, row) in zip((0, 17), rows):
            score = row["efficiency_dimensions"].get(dimension)
            if not isinstance(score, (int, float)):
                continue
            bar_width = (x1 - x0) * float(score) / 20
            body.append(f'<rect x="{x0}" y="{y + offset}" width="{bar_width:.1f}" height="13" fill="{PALETTE[system_id]}"/><text class="axis" x="{x0 + bar_width + 7:.1f}" y="{y + offset + 11}">{score:.1f}</text>')
    body.append(f'<rect x="48" y="96" width="12" height="12" fill="{PALETTE["tools"]}"/><text class="axis" x="66" y="107">Tools</text><rect x="128" y="96" width="12" height="12" fill="{PALETTE["tools-rag"]}"/><text class="axis" x="146" y="107">Tools + RAG</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 3}">维度得分（0–20）</text>')
    return _svg(FIGURE_TITLES["trajectory-efficiency-dimensions"], "Tools与Tools加RAG的五维工具效率均分。", "".join(body), width=width, height=height)


def _trajectory_recovery_quality(data: dict[str, Any]) -> str:
    systems = (data.get("trajectory_metrics") or {}).get("systems") or {}
    rows = []
    for system_id, system_label in (("tools", "Tools"), ("tools-rag", "Tools + RAG")):
        for mode, mode_label in (("native", "原生"), ("recovered", "恢复")):
            values = (systems.get(system_id) or {}).get("completion_modes", {}).get(mode)
            if values and values.get("n"):
                rows.append((f"{system_label} · {mode_label}", values))
    if not rows:
        return _empty_figure(FIGURE_TITLES["trajectory-recovery-quality"], "缺少完成模式数据。")
    width, height = 1120, 390
    x0, x1, y0, gap = 330, 1040, 126, 58
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["trajectory-recovery-quality"])}</text>',
        '<text class="subtitle" x="48" y="72">恢复机制能够生成最终回答，但不会补回未执行或未观察到的仿真证据。</text>',
        _axis_grid(x0=x0, x1=x1, y0=100, y1=y0 + (len(rows) - 1) * gap + 36),
    ]
    body.append('<rect x="48" y="92" width="12" height="12" fill="#2563eb"/><text class="axis" x="66" y="103">任务质量分</text><rect x="158" y="92" width="12" height="12" fill="#d97706"/><text class="axis" x="176" y="103">有效路径率</text>')
    for index, (label, values) in enumerate(rows):
        y = y0 + index * gap
        score = float(values.get("mean_score") or 0)
        valid = float(values.get("valid_path_rate") or 0) * 100
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 18}" y="{y + 14}">{label} (n={values["n"]})</text>')
        for offset, value, color in ((0, score, "#2563eb"), (18, valid, "#d97706")):
            bar_width = (x1 - x0) * value / 100
            body.append(f'<rect x="{x0}" y="{y + offset}" width="{bar_width:.1f}" height="14" fill="{color}"/><text class="axis" x="{x0 + bar_width + 7:.1f}" y="{y + offset + 12}">{value:.1f}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 14}">分数或比例（0–100）</text>')
    return _svg(FIGURE_TITLES["trajectory-recovery-quality"], "Tools和Tools加RAG按原生或恢复完成分组的质量与有效路径率。", "".join(body), width=width, height=height)


def _d6_state_lineage(data: dict[str, Any]) -> str:
    state = (data.get("trajectory_metrics") or {}).get("d6_state") or {}
    groups = state.get("groups") or {}
    rows = []
    for group, label in (("D6a", "D6a 混合串并联"), ("D6b", "D6b 串联"), ("D6c", "D6c 并联")):
        group_values = groups.get(group) or {}
        tools = group_values.get("tools") or {}
        rag = group_values.get("tools-rag") or {}
        if tools.get("state_lineage_n") or rag.get("state_lineage_n"):
            rows.append((label, tools, rag))
    if not rows:
        return _empty_figure(FIGURE_TITLES["d6-state-lineage"], "缺少明确命名State lineage的D6评分维度。")

    width, height = 1120, 390
    x0, x1, y0, gap = 330, 1040, 130, 72
    coverage = int(state.get("state_lineage_case_coverage") or 0)
    total = int(state.get("d6_case_count") or 0)
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["d6-state-lineage"])}</text>',
        f'<text class="subtitle" x="48" y="72">每维满分20；仅统计Rubric明确包含State lineage的{coverage}/{total}道D6题，其余旧版D6b不强行并入。</text>',
    ]
    for tick in range(0, 21, 5):
        x = x0 + (x1 - x0) * tick / 20
        body.append(f'<line class="grid" x1="{x:.1f}" y1="100" x2="{x:.1f}" y2="{y0 + (len(rows) - 1) * gap + 42}"/>')
        body.append(f'<text class="axis" text-anchor="middle" x="{x:.1f}" y="{height - 22}">{tick}</text>')
    body.append(f'<rect x="48" y="94" width="12" height="12" fill="{PALETTE["tools"]}"/><text class="axis" x="66" y="105">Tools</text>')
    body.append(f'<rect x="128" y="94" width="12" height="12" fill="{PALETTE["tools-rag"]}"/><text class="axis" x="146" y="105">Tools + RAG</text>')
    for index, (label, tools, rag) in enumerate(rows):
        y = y0 + index * gap
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 18}" y="{y + 19}">{_x(label)}</text>')
        for offset, system_id, values in ((0, "tools", tools), (22, "tools-rag", rag)):
            score = values.get("mean_state_lineage_score")
            if not isinstance(score, (int, float)):
                continue
            bar_width = (x1 - x0) * float(score) / 20
            n = int(values.get("state_lineage_n") or 0)
            body.append(f'<rect x="{x0}" y="{y + offset}" width="{bar_width:.1f}" height="16" fill="{PALETTE[system_id]}"/>')
            body.append(f'<text class="axis" x="{x0 + bar_width + 8:.1f}" y="{y + offset + 13}">{float(score):.1f} · n={n}</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 4}">State lineage评分（0–20）</text>')
    return _svg(
        FIGURE_TITLES["d6-state-lineage"],
        "D6a、D6b、D6c中明确命名State lineage维度的分层均分。",
        "".join(body),
        width=width,
        height=height,
    )


def _d6_state_failures(data: dict[str, Any]) -> str:
    state = (data.get("trajectory_metrics") or {}).get("d6_state") or {}
    failures = ((state.get("failures") or {}).get("D6-all") or {})
    if not failures:
        return _empty_figure(FIGURE_TITLES["d6-state-failures"], "缺少D6错误标签。")
    codes = ("PARAMETER_EXTRACTION", "TOOL_ARGUMENT", "CONSTRAINT_OMISSION", "OUTPUT_OMISSION")
    labels = {
        "PARAMETER_EXTRACTION": "题目参数提取",
        "TOOL_ARGUMENT": "工具参数传递",
        "CONSTRAINT_OMISSION": "约束状态遗漏",
        "OUTPUT_OMISSION": "结果或证据遗漏",
    }
    width, height = 1120, 390
    x0, x1, y0, gap = 330, 1040, 132, 56
    body = [
        f'<text class="title" x="48" y="46">{_x(FIGURE_TITLES["d6-state-failures"])}</text>',
        '<text class="subtitle" x="48" y="72">逐题发生率；这些代码与State保持有关，但不是State的纯粹或唯一测量。</text>',
        _axis_grid(x0=x0, x1=x1, y0=104, y1=y0 + (len(codes) - 1) * gap + 30),
    ]
    body.append(f'<circle cx="54" cy="96" r="6" fill="{PALETTE["tools"]}"/><text class="axis" x="67" y="100">Tools</text>')
    body.append(f'<circle cx="134" cy="96" r="6" fill="{PALETTE["tools-rag"]}"/><text class="axis" x="147" y="100">Tools + RAG</text>')
    for index, code in enumerate(codes):
        y = y0 + index * gap
        body.append(f'<text class="label" text-anchor="end" x="{x0 - 18}" y="{y + 5}">{_x(labels[code])}</text>')
        for offset, system_id in ((-7, "tools"), (7, "tools-rag")):
            values = (failures.get(system_id) or {}).get(code) or {}
            incidence = values.get("incidence")
            if not isinstance(incidence, (int, float)):
                continue
            x = x0 + (x1 - x0) * float(incidence)
            body.append(f'<circle cx="{x:.1f}" cy="{y + offset}" r="6" fill="{PALETTE[system_id]}"/>')
            body.append(f'<text class="axis" x="{x + 10:.1f}" y="{y + offset + 4}">{float(incidence) * 100:.0f}%</text>')
    body.append(f'<text class="axis" text-anchor="middle" x="{(x0 + x1) / 2:.1f}" y="{height - 8}">逐题发生率（0–100%）</text>')
    return _svg(
        FIGURE_TITLES["d6-state-failures"],
        "Tools与Tools加RAG在D6中的四类State相关错误信号。",
        "".join(body),
        width=width,
        height=height,
    )


def render_figure_svg(run_dir: Path, figure_id: str) -> str:
    if figure_id not in PAPER_FIGURE_IDS:
        raise ValueError(
            f"unknown figure {figure_id!r}; choose one of: {', '.join(PAPER_FIGURE_IDS)}"
        )
    data = build_figure_data(run_dir)
    renderers = {
        "main-scores": _main_scores,
        "tools-family-gain": _tools_family_gain,
        "tools-step-gain": _tools_step_gain,
        "quality-efficiency": _quality_efficiency,
        "rag-effect": _rag_effect,
        "rag-paired-outcomes": _rag_paired_outcomes,
        "rag-family-effect": _rag_family_effect,
        "router-policy": _router_policy,
        "reliability": _reliability,
        "trajectory-first-error": _trajectory_first_error,
        "trajectory-failure-incidence": _trajectory_failure_incidence,
        "trajectory-efficiency-dimensions": _trajectory_efficiency_dimensions,
        "trajectory-recovery-quality": _trajectory_recovery_quality,
        "d6-state-lineage": _d6_state_lineage,
        "d6-state-failures": _d6_state_failures,
    }
    return renderers[figure_id](data)


def export_figure(
    run_dir: Path,
    figure_id: str,
    output_path: Path | None = None,
) -> Path:
    output_path = output_path or (run_dir / "figures" / f"{figure_id}.svg")
    if output_path.suffix.lower() != ".svg":
        raise ValueError("paper figures are exported as SVG vector originals; output must end in .svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_figure_svg(run_dir, figure_id), encoding="utf-8")
    return output_path


def export_all_figures(run_dir: Path) -> dict[str, Path]:
    return {figure_id: export_figure(run_dir, figure_id) for figure_id in PAPER_FIGURE_IDS}
