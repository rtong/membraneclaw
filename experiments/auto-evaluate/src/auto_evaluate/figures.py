from __future__ import annotations

import html
from pathlib import Path
from statistics import mean
from typing import Any

from .evaluation import load_run_profile
from .io_utils import read_json, read_jsonl
from .reward_analysis import build_reward_analysis


PAPER_FIGURE_IDS = (
    "main-scores",
    "quality-efficiency",
    "rag-effect",
    "router-policy",
    "reliability",
)

FIGURE_TITLES = {
    "main-scores": "主要系统与 Teacher 参考的任务质量",
    "quality-efficiency": "任务质量与工具效率",
    "rag-effect": "RAG 反事实效应（按信息需求分组）",
    "router-policy": "Router 策略回放与独立端到端复跑",
    "reliability": "原生完成、上下文恢复与最终失败",
}

PALETTE = {
    "baseline": "#64748b",
    "tools": "#0f766e",
    "tools-rag": "#0891b2",
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
    return {
        "run_id": run_dir.name,
        "profile": profile,
        "route_spec": route_spec,
        "analysis": analysis,
        "score_rows": _score_rows(run_dir, route_spec),
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


def render_figure_svg(run_dir: Path, figure_id: str) -> str:
    if figure_id not in PAPER_FIGURE_IDS:
        raise ValueError(
            f"unknown figure {figure_id!r}; choose one of: {', '.join(PAPER_FIGURE_IDS)}"
        )
    data = build_figure_data(run_dir)
    renderers = {
        "main-scores": _main_scores,
        "quality-efficiency": _quality_efficiency,
        "rag-effect": _rag_effect,
        "router-policy": _router_policy,
        "reliability": _reliability,
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
