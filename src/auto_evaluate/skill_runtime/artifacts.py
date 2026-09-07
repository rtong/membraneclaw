"""Load and validate a versioned executable Skill."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import ToolSpec, digest, require, validate_plan


def load_skill(directory: Path, tools: dict[str, ToolSpec]) -> dict:
    text = (directory / "SKILL.md").read_text(encoding="utf-8")
    require(len(text) <= 12000, "SKILL_TOO_LONG", "Character safety limit, not a token estimate")
    header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|$)", text, re.DOTALL)
    require(header is not None, "INVALID_SKILL", "Frontmatter required")
    name = re.search(r"(?m)^name: ([a-z0-9-]{1,63})\s*$", header.group(1))
    description = re.search(r"(?m)^description: (\S.*)$", header.group(1))
    require(name is not None and description is not None, "INVALID_SKILL", "Name and description required")
    plan = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    plan, _ = validate_plan(plan, tools)
    require(re.fullmatch(re.escape(name.group(1)) + r"@\d+\.\d+\.\d+", plan.get("skill_id", "")) is not None,
            "INVALID_SKILL", "Workflow must identify this skill and a version")
    return {"skill_id": plan["skill_id"], "instructions": text, "plan": plan,
            "sha256": digest({"instructions": text, "plan": plan})}
