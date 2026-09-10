from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate.figures import PALETTE, _svg, _x  # noqa: E402


def main() -> int:
    mainline = json.loads(
        (ROOT / "runs/d1-d6-full-20260828-v1/reward_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    solver = json.loads(
        (
            ROOT
            / "runs/solver-skill-p3-validation-input-complete-3case-v1"
            / "solver_skill_analysis.json"
        ).read_text(encoding="utf-8")
    )
    systems = mainline["systems"]
    primary = solver["primary_outcome"]
    groups = [
        (
            "Stage A: Add engineering tools (117 cases)",
            [
                ("9B only", systems["baseline"]["mean_task_reward"], PALETTE["baseline"]),
                ("9B + WaterTAP tools", systems["tools"]["mean_task_reward"], PALETTE["tools"]),
                ("9B + tools + RAG", systems["tools-rag"]["mean_task_reward"], PALETTE["tools-rag"]),
            ],
            "+31.44 from tools",
        ),
        (
            "Stage B: Add Solver Skill on top (3 cases)",
            [
                ("9B + tools, no Skill", primary["c00_mean_case_best_score"], PALETTE["baseline"]),
                ("9B + tools + Solver Skill", primary["c10_mean_case_best_score"], "#7c3aed"),
            ],
            "+9.50 from Solver Skill",
        ),
    ]
    body = [
        '<text class="title" x="48" y="44">Two separate improvements in the SWRO agent</text>',
        '<text class="subtitle" x="48" y="70">Stage A adds tools to the 9B model. Stage B keeps those tools and adds only the Solver Skill.</text>',
    ]
    for group_index, (title, rows, note) in enumerate(groups):
        x_base = 48 + group_index * 550
        body.append(
            f'<text class="label" x="{x_base}" y="112" font-weight="700">{_x(title)}</text>'
        )
        body.append(
            f'<text class="value" x="{x_base + 520}" y="138" text-anchor="end" fill="#15803d">{_x(note)}</text>'
        )
        for row_index, (label, value, color) in enumerate(rows):
            y = 164 + row_index * 72
            width = 300 * float(value) / 100
            body.extend(
                [
                    f'<text class="label" x="{x_base}" y="{y + 18}">{_x(label)}</text>',
                    f'<rect x="{x_base + 210}" y="{y}" width="300" height="26" rx="4" fill="#eef2f6"/>',
                    f'<rect x="{x_base + 210}" y="{y}" width="{width:.1f}" height="26" rx="4" fill="{color}"/>',
                    f'<text class="value" x="{x_base + 520}" y="{y + 18}" text-anchor="end">{float(value):.2f}</text>',
                ]
            )
        body.extend(
            [
                f'<text class="subtitle" x="{x_base + 210}" y="404">0</text>',
                f'<text class="subtitle" x="{x_base + 510}" y="404" text-anchor="end">100</text>',
            ]
        )
    body.append(
        '<text class="subtitle" x="48" y="450">Stage A uses mean scores over 117 D1-D6 cases. Stage B uses mean per-case best-of-3 over D6-6c-02/03/06.</text>'
    )
    output = ROOT / "docs/figures/swro-skills-progress-2026-09-10.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _svg(
            "Two separate improvements in the SWRO agent",
            "First engineering tools are added to the 9B model. Then the Solver Skill is tested while keeping those tools fixed.",
            "".join(body),
            width=1120,
            height=470,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
