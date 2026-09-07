"""Separate offline entrypoint; never loads .env or starts a model/tool service."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import load_skill
from .contracts import ContractError, digest, validate_plan, validate_task
from .engine import SkillRuntime
from .replay import FixtureBackend, fixture_tool_specs


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline executable-Skill prototype (no live models)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run-fixture"):
        command = sub.add_parser(name)
        command.add_argument("--task", type=Path, required=True)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--skill", type=Path)
        source.add_argument("--plan", type=Path)
        if name == "run-fixture":
            command.add_argument("--fixture", type=Path, required=True)
            command.add_argument("--out", type=Path, required=True, help="New directory, never overwritten")
            command.add_argument("--enforcement", choices=("enforce", "observe"), default="enforce")
            command.add_argument("--max-tool-calls", type=int, default=24)
    args = parser.parse_args(argv)
    try:
        tools = fixture_tool_specs()
        task = validate_task(_read(args.task), tools)
        artifact = load_skill(args.skill, tools) if args.skill else None
        plan = artifact["plan"] if artifact else _read(args.plan)
        plan, order = validate_plan(plan, tools)
        if args.command == "validate":
            # Constructor additionally checks task/plan branch compatibility, without any calls.
            SkillRuntime(task, plan, tools, FixtureBackend({"fixture_only": True, "source": "validation", "records": []}))
            print(json.dumps({"status": "valid", "mode": "offline_contract_check", "nodes": order,
                              "skill_id": artifact["skill_id"] if artifact else None,
                              "extraction_correctness_verified": False}, ensure_ascii=False, indent=2))
            return 0
        fixture = _read(args.fixture)
        backend = FixtureBackend(fixture)
        metadata = {"skill_id": artifact["skill_id"], "sha256": artifact["sha256"]} if artifact else {}
        metadata.update({"fixture_only": True, "fixture_hash": digest(fixture)})
        runtime = SkillRuntime(task, plan, tools, backend, enforcement=args.enforcement,
                               max_tool_calls=args.max_tool_calls, run_dir=args.out, artifact=metadata)
        if artifact:
            with (args.out / "skill-snapshot.json").open("x", encoding="utf-8") as handle:
                json.dump(artifact, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        result = runtime.run()
        result.update({"fixture_only": True, "live_model_calls": 0, "live_simulator_calls": 0})
        # Runtime owns a new directory; writing this result cannot overwrite another run.
        with (args.out / "result.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"completed", "infeasible_supported"} else 2
    except (ContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
