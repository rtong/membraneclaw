from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_evaluate.io_utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from auto_evaluate.judge_calibration import (  # noqa: E402
    build_calibration_analysis,
    prepare_calibration,
    write_calibration_analysis,
)


def _benchmark(case_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "task_family": "test",
        "title": "Calibration fixture",
        "question_prompt": "Determine whether the candidate meets the stated constraint.",
        "reference_answer": "The constraint is met.",
        "rubric": {
            "total_points": 100,
            "steps": [
                {
                    "step_id": 1,
                    "max_score": 100,
                    "criterion": "Correct decision with evidence",
                }
            ],
        },
        "tool_efficiency_rubric": {
            "total_points": 100,
            "dimensions": [
                {
                    "dimension_id": "E1",
                    "max_score": 100,
                    "criterion": "Sufficient observable evidence",
                }
            ],
        },
    }


class JudgeCalibrationTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        source = root / "runs" / "source"
        benchmark = _benchmark("Q1")
        write_json(source / "benchmarks" / "Q1.json", benchmark)
        write_json(
            source / "benchmarks" / "index.json",
            {
                "schema_version": "1.0",
                "cases": [
                    {
                        "case_id": "Q1",
                        "task_family": "test",
                        "title": "Calibration fixture",
                        "file": "Q1.json",
                    }
                ],
            },
        )
        for system_id in ("baseline", "tools"):
            write_json(
                source / "responses" / f"Q1__{system_id}.json",
                {
                    "case_id": "Q1",
                    "system_id": system_id,
                    "display_name": system_id.title(),
                    "response_text": "The constraint is met based on the observed result.",
                    "status": "success",
                    "native_status": "success",
                    "completion_mode": "native",
                    "trajectory": {
                        "source": "fixture",
                        "events": [
                            {
                                "event_id": "T001",
                                "event_type": "tool_interaction",
                                "status": "success",
                                "observation": {"value": 1},
                            }
                        ],
                    },
                },
            )
        skill = root / "skills" / "swro-evaluation-judge" / "v0.1.0" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            "---\nname: swro-evaluation-judge\ndescription: Judge evidence.\n---\n\n"
            "Check every rubric step against observable evidence.\n",
            encoding="utf-8",
        )
        config = root / "configs" / "judge_calibration.json"
        write_json(
            config,
            {
                "schema_version": "1.0",
                "selection_id": "fixture@0.1.0",
                "source_run": "source",
                "evaluation_skill": "skills/swro-evaluation-judge/v0.1.0/SKILL.md",
                "judge_conditions": ["j0", "j1"],
                "repeats": 2,
                "seed": 7,
                "case_ids": ["Q1"],
                "system_ids": ["baseline", "tools"],
            },
        )
        return config

    def test_prepare_freezes_candidates_and_hides_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_fixture(root)
            manifest = prepare_calibration(root=root, config_path=config, run_id="calibration")
            run = root / "runs" / "calibration"
            tasks = read_jsonl(run / "judge_batch.jsonl")
            mapping = read_json(run / "judge_mapping.json")["mapping"]

            self.assertEqual(2, manifest["candidate_count"])
            self.assertEqual(8, manifest["judge_task_count"])
            self.assertEqual(8, len(tasks))
            self.assertEqual(8, len(mapping))
            self.assertEqual(4, sum("evaluation_skill" in task for task in tasks))
            self.assertTrue(all("judge_condition" not in task for task in tasks))
            self.assertTrue(all("system_id" not in task for task in tasks))
            self.assertTrue(all("j0" not in task["task_id"] and "j1" not in task["task_id"] for task in tasks))
            self.assertEqual(2, len(read_jsonl(run / "expert_review_batch.jsonl")))
            self.assertEqual(2, len(read_jsonl(run / "expert_ratings.template.jsonl")))
            self.assertTrue((root / "runs" / "source" / "responses" / "Q1__tools.json").is_file())

    def test_analysis_reports_repeat_stability_without_expert_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_fixture(root)
            prepare_calibration(root=root, config_path=config, run_id="calibration")
            run = root / "runs" / "calibration"
            mapping = read_json(run / "judge_mapping.json")["mapping"]
            ratings = []
            for identity in mapping:
                base = 60 if identity["judge_condition"] == "j0" else 64
                score = base + identity["repeat_index"] - 1
                ratings.append(
                    {
                        "task_id": identity["task_id"],
                        "case_id": identity["case_id"],
                        "candidate_label": identity["candidate_label"],
                        "total_score": score,
                        "steps": [{"failure_codes": ["OTHER"]}],
                    }
                )
            write_jsonl(run / "ratings.jsonl", ratings)

            analysis = build_calibration_analysis(run)
            self.assertEqual(1.0, analysis["condition_summaries"]["j0"]["repeat_mean_absolute_difference"])
            self.assertEqual(4.0, analysis["j1_minus_j0"]["mean_score_shift"])
            self.assertEqual("pending", analysis["expert_calibration"]["status"])
            output = write_calibration_analysis(run, analysis)
            self.assertIn("不能声称J1更正确", output.read_text(encoding="utf-8"))

    def test_prepare_refuses_to_overwrite_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_fixture(root)
            prepare_calibration(root=root, config_path=config, run_id="calibration")
            with self.assertRaises(FileExistsError):
                prepare_calibration(root=root, config_path=config, run_id="calibration")


if __name__ == "__main__":
    unittest.main()
