from __future__ import annotations

import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, utc_now
from .judge import validate_ratings
from .skill_gate import DEFAULT_CONFIG, evaluate_skill_gate


COLORS = {
    "baseline": "#64748b",
    "environment": "#0f766e",
    "environment-skill": "#2563eb",
    "gpt-5.6-teacher": "#7c3aed",
}


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _heat(score_pct: float) -> str:
    score_pct = max(0.0, min(100.0, score_pct))
    hue = score_pct * 1.2
    return f"hsl({hue:.0f} 58% 88%)"


def _load_response_records(run_dir: Path) -> list[dict[str, Any]]:
    response_dir = run_dir / "responses"
    return [read_json(path) for path in sorted(response_dir.glob("*.json"))] if response_dir.exists() else []


def _benchmark_titles(project_root: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    benchmark_dir = project_root / "benchmarks" / "normalized"
    if not benchmark_dir.exists():
        return titles
    for path in benchmark_dir.glob("*.json"):
        if path.name == "index.json":
            continue
        try:
            row = read_json(path)
        except (ValueError, OSError):
            continue
        if row.get("case_id"):
            titles[row["case_id"]] = row.get("title") or row["case_id"]
    return titles


def _display_names(mapping: dict[str, dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in mapping.values():
        system_id = row.get("system_id")
        if system_id:
            result[system_id] = row.get("display_name") or system_id
    return result


def _render_gate(gate: dict[str, Any], names: dict[str, str]) -> str:
    if gate.get("status") == "unavailable":
        return (
            '<section class="panel gate unavailable"><div><div class="section-kicker">Skill promotion gate</div>'
            '<h2>尚不能判定</h2></div><p class="muted">'
            f'{_e(gate.get("reason"))}</p></section>'
        )
    config = gate["config"]
    candidate = names.get(config["candidate_system"], config["candidate_system"])
    baseline = names.get(config["baseline_system"], config["baseline_system"])
    status = "PASS" if gate["passed"] else "FAIL"
    rows = "".join(
        '<tr>'
        f'<td>{_e(row["case_id"])}</td>'
        f'<td>{row["baseline_score"]:.1f}</td>'
        f'<td>{row["candidate_score"]:.1f}</td>'
        f'<td class="delta {"positive" if row["gain"] > 0 else "negative"}">{row["gain"]:+.1f}</td>'
        f'<td>{"通过" if row["passed"] else "未通过"}</td></tr>'
        for row in gate["cases"]
    )
    forbidden = gate.get("forbidden_failure_codes_found", {})
    failure_note = (
        "；禁止错误仍存在：" + ", ".join(f"{code}×{count}" for code, count in forbidden.items())
        if forbidden
        else "；未发现禁止错误"
    )
    return f'''
<section class="panel gate {"pass" if gate["passed"] else "fail"}">
  <div class="gate-head"><div><div class="section-kicker">Skill promotion gate</div>
  <h2>{_e(candidate)} vs {_e(baseline)}</h2></div><span class="gate-status">{status}</span></div>
  <p>平均分：{gate["candidate_mean"]:.1f} vs {gate["baseline_mean"]:.1f}，增益
  <strong class="delta {"positive" if gate["mean_gain"] > 0 else "negative"}">{gate["mean_gain"]:+.1f}</strong>
  {failure_note}。晋级要求为每道开发题严格增分、平均分增分，且无 TOOL_ARGUMENT/PARAMETER_EXTRACTION。</p>
  <div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>{_e(baseline)}</th>
  <th>{_e(candidate)}</th><th>增益</th><th>结果</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>'''


def build_report(run_dir: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or (run_dir / "report.html")
    project_root = run_dir.parents[1]
    manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    responses = _load_response_records(run_dir)
    ratings = read_jsonl(run_dir / "ratings.jsonl")
    mapping_doc = read_json(run_dir / "judge_mapping.json") if (run_dir / "judge_mapping.json").exists() else {"mapping": []}
    mapping = {row["task_id"]: row for row in mapping_doc.get("mapping", []) if row.get("task_id")}
    names = _display_names(mapping)
    titles = _benchmark_titles(project_root)
    rating_errors = validate_ratings(run_dir) if (run_dir / "judge_batch.jsonl").exists() else []

    gate_config_path = project_root / "configs" / "skill_promotion.json"
    gate_config = read_json(gate_config_path) if gate_config_path.exists() else DEFAULT_CONFIG
    gate = evaluate_skill_gate(run_dir, gate_config)

    per_system: dict[str, list[float]] = defaultdict(list)
    failures: Counter[str] = Counter()
    ratings_by_case: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    responses_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rating in ratings:
        identity = mapping.get(rating.get("task_id"), {})
        system_id = identity.get("system_id", "unknown")
        display_name = identity.get("display_name", names.get(system_id, system_id))
        case_id = rating.get("case_id") or identity.get("case_id") or "unknown"
        per_system[system_id].append(float(rating.get("total_score", 0)))
        for step in rating.get("steps", []):
            failures.update(step.get("failure_codes", []))
        ratings_by_case[case_id].append((display_name, system_id, rating))
    for record in responses:
        responses_by_case[record.get("case_id") or "unknown"].append(record)

    success_count = sum(row.get("status") == "success" for row in responses)
    error_count = sum(row.get("status") == "error" for row in responses)
    system_order = [
        system_id
        for system_id, _ in sorted(
            per_system.items(), key=lambda item: (-sum(item[1]) / len(item[1]), item[0])
        )
    ]

    mean_cards = []
    for system_id in system_order:
        values = per_system[system_id]
        mean = sum(values) / len(values)
        mean_cards.append(
            f'<div class="score-card"><div class="label">{_e(names.get(system_id, system_id))}</div>'
            f'<div class="score">{mean:.1f}</div><div class="muted">平均分 / 100 · n={len(values)}</div>'
            f'<div class="bar"><span style="width:{mean:.1f}%;background:{COLORS.get(system_id, "#334155")}"></span></div></div>'
        )
    if not mean_cards:
        mean_cards.append('<div class="notice">自动评分尚未完成。生成 <code>ratings.jsonl</code> 后重新生成报告。</div>')

    failure_rows = []
    max_failure = max(failures.values(), default=1)
    for code, count in failures.most_common():
        width = count / max_failure * 100
        failure_rows.append(
            f'<div class="failure"><span>{_e(code)}</span><b>{count}</b>'
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
        summary_chips = "".join(
            f'<span class="score-chip"><i style="background:{COLORS.get(system_id, "#334155")}"></i>'
            f'{_e(names.get(system_id, system_id))} <b>{score_by_system[system_id]:.0f}</b></span>'
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
        step_headers = "".join(f'<th>{_e(names.get(system_id, system_id))}</th>' for system_id in system_order if system_id in score_by_system)
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
                diagnosis = step.get("diagnosis", "")
                cells.append(
                    f'<td style="background:{_heat(pct)}" title="{_e(diagnosis)}">'
                    f'<strong>{score:.1f}/{maximum:.1f}</strong><small>{pct:.0f}%</small></td>'
                )
            step_rows.append(f'<tr><th>S{step_id}</th>{"".join(cells)}</tr>')
        step_table = (
            f'<div class="table-wrap"><table class="step-matrix"><thead><tr><th>步骤</th>{step_headers}</tr></thead>'
            f'<tbody>{"".join(step_rows)}</tbody></table></div>'
            if step_rows
            else '<div class="muted">暂无分步评分</div>'
        )

        overview_rows = "".join(
            f'<tr><td>{_e(display)}</td><td><strong>{float(rating.get("total_score", 0)):.1f}</strong></td>'
            f'<td>{_e(rating.get("overall_diagnosis", ""))}</td></tr>'
            for display, _, rating in case_ratings
        ) or '<tr><td colspan="3" class="muted">暂无评分</td></tr>'

        diagnosis_sections = []
        for display, _, rating in case_ratings:
            step_lines = "".join(
                f'<li><strong>S{_e(step.get("step_id"))}: {_e(step.get("score"))}/{_e(step.get("max_score"))}</strong> '
                f'{_e(step.get("diagnosis"))}<br><span class="muted">{_e(", ".join(step.get("failure_codes", [])))}</span></li>'
                for step in rating.get("steps", [])
            )
            suggestions = "".join(f'<li>{_e(item)}</li>' for item in rating.get("skill_improvement_suggestions", []))
            diagnosis_sections.append(
                f'<details class="inner-detail"><summary><strong>{_e(display)}</strong> · '
                f'{float(rating.get("total_score", 0)):.1f}/100</summary><ol>{step_lines}</ol>'
                f'<p>{_e(rating.get("overall_diagnosis", ""))}</p><h4>Skill 改进建议</h4>'
                f'<ul>{suggestions or "<li>无</li>"}</ul></details>'
            )

        response_sections = []
        for record in case_responses:
            status = record.get("status", "unknown")
            body = record.get("response_text") or record.get("error") or ""
            response_sections.append(
                f'<details class="inner-detail"><summary><span class="pill {status}">{_e(status)}</span> '
                f'{_e(record.get("display_name", record.get("system_id")))} · '
                f'{_e(record.get("latency_ms", "—"))} ms</summary><pre>{_e(body)}</pre></details>'
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
      <button class="tab" data-target="{panel_id}-steps">Step scores</button>
      <button class="tab" data-target="{panel_id}-diagnosis">逐步诊断</button>
      <button class="tab" data-target="{panel_id}-responses">原始响应</button>
    </div>
    <section id="{panel_id}-overview" class="tab-panel active"><div class="table-wrap"><table>
      <thead><tr><th>系统</th><th>总分</th><th>总体诊断</th></tr></thead><tbody>{overview_rows}</tbody></table></div></section>
    <section id="{panel_id}-steps" class="tab-panel">{step_table}</section>
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
.gate{{border-left:5px solid var(--red)}}.gate.pass{{border-left-color:var(--green)}}.gate.unavailable{{border-left-color:#98a2b3}}.gate-head{{display:flex;justify-content:space-between;align-items:start;gap:18px}}.gate-status{{font-size:20px;font-weight:850;padding:5px 12px;border-radius:9px;background:#fee4e2;color:var(--red)}}.gate.pass .gate-status{{background:#dcfae6;color:var(--green)}}.delta{{font-weight:800}}.positive{{color:var(--green)}}.negative{{color:var(--red)}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}thead th{{font-size:12px;color:var(--muted);text-transform:uppercase;white-space:nowrap}}
.failure{{display:grid;grid-template-columns:minmax(190px,1fr) 40px 2fr;gap:10px;align-items:center;margin:10px 0}}.failure b{{text-align:right}}.mini{{margin:0}}.mini i{{background:var(--teal)}}
.benchmark-toolbar{{position:sticky;top:0;z-index:10;display:flex;gap:10px;flex-wrap:wrap;padding:12px 0;background:var(--bg)}}.benchmark-toolbar input{{min-width:280px;flex:1;border:1px solid var(--line);border-radius:9px;padding:10px 12px;font:inherit}}button{{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 12px;cursor:pointer;font:inherit}}button:hover{{border-color:#98a2b3}}
.benchmark-card{{margin-top:12px;overflow:hidden}}.benchmark-card>summary{{list-style:none;cursor:pointer;padding:17px 20px;display:flex;justify-content:space-between;align-items:center;gap:18px}}.benchmark-card>summary::-webkit-details-marker{{display:none}}.benchmark-card>summary>div:first-child{{display:flex;align-items:center;gap:12px;min-width:0}}.benchmark-card>summary strong{{display:block}}.benchmark-card>summary small{{display:block;color:var(--muted);font-family:ui-monospace,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.benchmark-card[open]>summary{{border-bottom:1px solid var(--line);background:#fbfcfe}}.case-number{{display:grid;place-items:center;min-width:34px;height:34px;border-radius:9px;background:#eef4ff;color:var(--blue);font-weight:800}}.summary-scores{{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}}.score-chip{{font-size:12px;background:#f2f4f7;border-radius:99px;padding:4px 8px;white-space:nowrap}}.score-chip i{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}}
.benchmark-body{{padding:18px 20px 22px}}.tabs{{display:flex;gap:6px;overflow:auto;border-bottom:1px solid var(--line);margin-bottom:14px}}.tab{{border:0;border-radius:8px 8px 0 0;background:transparent;color:var(--muted);white-space:nowrap}}.tab.active{{color:var(--blue);background:#eef4ff;font-weight:750}}.tab-panel{{display:none}}.tab-panel.active{{display:block}}.step-matrix td strong{{display:block;white-space:nowrap}}.step-matrix td small{{color:#475467}}
.inner-detail{{border-top:1px solid var(--line);padding:12px 0}}.inner-detail:first-child{{border-top:0}}.inner-detail summary{{cursor:pointer}}pre{{white-space:pre-wrap;background:#f8fafc;border:1px solid var(--line);padding:16px;border-radius:10px;max-height:520px;overflow:auto}}.pill{{font-size:11px;padding:3px 8px;border-radius:99px;background:#eef2f6}}.pill.success{{background:#dcfce7;color:#166534}}.pill.error{{background:#fee2e2;color:#991b1b}}
.notice,.warning{{padding:18px}}.warning{{border-color:#f7c948;background:#fffbeb;margin-top:16px}}code{{background:#eef2f6;padding:2px 5px;border-radius:5px}}
@media(max-width:760px){{main{{padding:24px 14px}}.failure{{grid-template-columns:1fr 35px}}.failure .mini{{grid-column:1/-1}}.benchmark-card>summary{{align-items:flex-start;flex-direction:column}}.summary-scores{{justify-content:flex-start}}}}
</style></head><body><main>
<section class="hero"><div class="eyebrow">SWRO AUTO EVALUATE</div><h1>运行 {_e(run_dir.name)}</h1>
<div class="muted">Baseline、Environment 与 Environment-Skill 的分步诊断报告</div>
<div class="meta"><span>生成时间：{_e(generated)}</span><span>Benchmark：{len(case_ids)}</span><span>成功响应：{success_count}</span><span>失败响应：{error_count}</span><span>评分：{len(ratings)}</span></div></section>
{warning_block}
{_render_gate(gate, names)}
<section class="panel"><h2>系统得分</h2><div class="grid">{"".join(mean_cards)}</div></section>
<section class="panel"><h2>错误类型分布</h2>{"".join(failure_rows)}</section>
<section class="panel"><div class="section-kicker">Scalable benchmark explorer</div><h2>Step-level score 与逐题诊断</h2>
<p class="muted">每个 benchmark 独立归组；展开题目后可在概览、分步得分、诊断和原始响应之间切换。</p>
<div class="benchmark-toolbar"><input id="benchmark-search" type="search" placeholder="按 benchmark ID 或标题筛选…" aria-label="筛选 benchmark">
<button id="expand-all" type="button">展开全部</button><button id="collapse-all" type="button">收起全部</button></div>
<div id="benchmark-list">{"".join(case_sections)}</div></section>
<p class="muted">Run manifest: {_e(manifest.get("run_id", run_dir.name))} · report schema 2.0</p>
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
