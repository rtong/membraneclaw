from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_evaluate.io_utils import read_json, sha256_tree, write_json, write_jsonl  # noqa: E402
from auto_evaluate.runner import load_systems, system_prompt_for_system  # noqa: E402
from auto_evaluate.solver_skill_pilot import (  # noqa: E402
    build_solver_skill_analysis,
    prepare_solver_skill_pilot,
)


def benchmark() -> dict:
    return {
        "schema_version": "1.0",
        "case_id": "case-parallel",
        "task_family": "parallel",
        "title": "Parallel candidate comparison",
        "question_prompt": "Compare two public candidates with both tools.",
        "reference_answer": "Candidate one is feasible.",
        "rubric": {
            "total_points": 100,
            "steps": [{"step_id": 1, "step_label": "Decision", "max_score": 100}],
        },
        "tool_efficiency_rubric": {
            "total_points": 100,
            "dimensions": [
                {"dimension_id": "E1", "dimension_label": "Calls", "max_score": 100}
            ],
        },
        "source": {"sha256": "benchmark-sha"},
    }


class SolverSkillPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        normalized = self.root / "benchmarks" / "normalized"
        normalized.mkdir(parents=True)
        write_json(normalized / "case-parallel.json", benchmark())
        write_json(
            normalized / "index.json",
            {
                "schema_version": "1.0",
                "cases": [
                    {
                        "case_id": "case-parallel",
                        "task_family": "parallel",
                        "title": "Parallel candidate comparison",
                        "file": "case-parallel.json",
                        "source_sha256": "benchmark-sha",
                    }
                ],
            },
        )
        skill_dir = self.root / "skills" / "swro-parallel-compare" / "v0.1.0"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: swro-parallel-compare\ndescription: Test procedure.\n---\n\nKeep branches aligned.\n",
            encoding="utf-8",
        )
        write_json(
            skill_dir / "workflow.json",
            {"schema_version": "1.0", "skill_id": "swro-parallel-compare@0.1.0", "nodes": []},
        )
        configs = self.root / "configs"
        configs.mkdir()
        write_json(
            configs / "systems.json",
            {
                "schema_version": "1.0",
                "shared_system_prompt": "Shared solver prompt.",
                "generation": {"stream": True, "temperature": 0.2, "top_p": 0.9},
                "systems": [
                    {
                        "id": "tools",
                        "display_name": "Tools",
                        "model_env": "OPENWEBUI_MODEL_TOOLS",
                        "tools_enabled": True,
                        "require_observable_tool_call": True,
                        "rag_enabled": False,
                        "skill_version": None,
                    }
                ],
            },
        )
        self.config_path = configs / "solver_skill_pilot.json"
        write_json(
            self.config_path,
            {
                "schema_version": "1.0",
                "selection_id": "pilot-test@0.1.0",
                "source_benchmark_set": "test",
                "source_benchmarks_dir": "benchmarks/normalized",
                "base_systems_config": "configs/systems.json",
                "base_system_id": "tools",
                "skill_version": "swro-parallel-compare@0.1.0",
                "case_ids": ["case-parallel"],
                "conditions": ["c00", "c10"],
                "repeats": 3,
                "primary_outcome": "best_of_3_task_score",
                "required_observable_tools": [
                    "simulate_ro",
                    "simulate_swro_system",
                ],
                "max_collection_attempts": 4,
            },
        )

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self) -> tuple[Path, dict]:
        manifest = prepare_solver_skill_pilot(
            root=self.root,
            config_path=self.config_path,
            run_id="pilot-run",
        )
        return self.root / "runs" / "pilot-run", manifest

    def test_prepare_freezes_six_episodes_and_local_skill(self):
        run_dir, manifest = self.prepare()
        self.assertEqual(6, manifest["expected_solver_episodes"])
        self.assertEqual(["case-parallel"], manifest["case_ids"])
        systems = read_json(run_dir / "systems.json")["systems"]
        self.assertEqual(
            ["c00-r1", "c10-r1", "c00-r2", "c10-r2", "c00-r3", "c10-r3"],
            [row["id"] for row in systems],
        )
        self.assertNotIn("skill_version", systems[0])
        self.assertEqual(
            ["simulate_ro", "simulate_swro_system"],
            systems[0]["required_observable_tools"],
        )
        self.assertEqual("prompt", systems[1]["skill_delivery"])
        self.assertEqual("solver_skill", systems[1]["skill_path"])
        self.assertTrue((run_dir / "solver_skill" / "SKILL.md").is_file())
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_prompt_delivery_loads_frozen_snapshot_and_changes_only_prompt(self):
        run_dir, manifest = self.prepare()
        with patch.dict(os.environ, {"OPENWEBUI_MODEL_TOOLS": "tools"}, clear=False):
            config = load_systems(
                run_dir / "systems.json", ["c00-r1", "c10-r1"], load_recovery=False
            )
        c00, c10 = config["systems"]
        self.assertNotIn("skill_prompt", c00)
        self.assertEqual(manifest["solver_skill_sha256"], c10["skill_artifact_sha256"])
        self.assertEqual("Shared solver prompt.", system_prompt_for_system("Shared solver prompt.", c00))
        prompt = system_prompt_for_system("Shared solver prompt.", c10)
        self.assertIn("[FROZEN_SOLVER_SKILL_BEGIN]", prompt)
        self.assertIn("Keep branches aligned.", prompt)
        self.assertIn("[FROZEN_SOLVER_SKILL_END]", prompt)

    def test_analysis_pairs_repeats_without_claiming_evolution(self):
        run_dir, manifest = self.prepare()
        responses_dir = run_dir / "responses"
        responses_dir.mkdir()
        mapping = []
        ratings = []
        for repeat_index in range(1, 4):
            for condition, score in (("c00", 50 + repeat_index), ("c10", 55 + repeat_index)):
                system_id = f"{condition}-r{repeat_index}"
                response = {
                    "case_id": "case-parallel",
                    "system_id": system_id,
                    "model_id": "tools",
                    "status": "success",
                    "native_status": "success",
                    "completion_mode": "native",
                    "rag_enabled": False,
                    "skill_artifact_sha256": (
                        manifest["solver_skill_sha256"] if condition == "c10" else None
                    ),
                    "latency_ms": 1000,
                    "usage": {"total_tokens": 100},
                    "trajectory": {
                        "summary": {"tool_interactions": 8, "tool_errors": 0}
                    },
                }
                write_json(responses_dir / f"case-parallel__{system_id}.json", response)
                task_id = f"judge-{system_id}"
                mapping.append(
                    {
                        "task_id": task_id,
                        "case_id": "case-parallel",
                        "system_id": system_id,
                        "display_name": system_id,
                    }
                )
                ratings.append(
                    {
                        "task_id": task_id,
                        "case_id": "case-parallel",
                        "total_score": score,
                        "steps": [],
                        "trajectory_analysis": {"path_classification": "valid_alternative"},
                    }
                )
        write_json(run_dir / "judge_mapping.json", {"schema_version": "1.0", "mapping": mapping})
        write_jsonl(run_dir / "ratings.jsonl", ratings)
        analysis = build_solver_skill_analysis(run_dir)
        self.assertEqual("best_of_3_task_score", analysis["primary_outcome"]["metric"])
        self.assertEqual(53, analysis["primary_outcome"]["c00_best_score"])
        self.assertEqual(58, analysis["primary_outcome"]["c10_best_score"])
        self.assertEqual(5, analysis["primary_outcome"]["c10_minus_c00"])
        self.assertEqual(5, analysis["paired_effect"]["mean_effect"])
        self.assertEqual(3, analysis["paired_effect"]["wins"])
        self.assertIn("does not test the hard executor", analysis["interpretation_boundary"])


if __name__ == "__main__":
    unittest.main()
