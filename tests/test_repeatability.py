from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_evaluate.io_utils import write_json, write_jsonl  # noqa: E402
from auto_evaluate.repeatability import (  # noqa: E402
    build_repeatability_analysis,
    write_repeatability_analysis,
)


class RepeatabilityTests(unittest.TestCase):
    def _write_run(self, run_dir: Path, scores: dict[tuple[str, str], float]) -> None:
        mapping = []
        ratings = []
        for (case_id, system_id), score in scores.items():
            task_id = f"judge::{case_id}::{system_id}"
            mapping.append(
                {"task_id": task_id, "case_id": case_id, "system_id": system_id}
            )
            ratings.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "total_score": score,
                    "steps": [],
                    "trajectory_analysis": {"path_classification": "golden_aligned"},
                    "causal_analysis": {"first_error_step_id": None},
                }
            )
            write_json(
                run_dir / "responses" / f"{case_id}__{system_id}.json",
                {
                    "case_id": case_id,
                    "system_id": system_id,
                    "status": "success",
                    "native_status": "success",
                    "completion_mode": "native",
                    "latency_ms": 100,
                    "trajectory": {"summary": {"tool_interactions": 2}},
                },
            )
        write_json(run_dir / "judge_mapping.json", {"mapping": mapping})
        write_jsonl(run_dir / "ratings.jsonl", ratings)

    def test_case_averaged_repeatability_effects_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run1 = root / "r1"
            run2 = root / "r2"
            self._write_run(
                run1,
                {
                    ("q1", "baseline"): 10,
                    ("q1", "tools"): 30,
                    ("q1", "tools-rag"): 25,
                    ("q2", "baseline"): 20,
                    ("q2", "tools"): 40,
                    ("q2", "tools-rag"): 50,
                },
            )
            self._write_run(
                run2,
                {
                    ("q1", "baseline"): 12,
                    ("q1", "tools"): 32,
                    ("q1", "tools-rag"): 36,
                    ("q2", "baseline"): 18,
                    ("q2", "tools"): 38,
                    ("q2", "tools-rag"): 35,
                },
            )
            analysis = build_repeatability_analysis(
                [run1, run2],
                selection={"selection_id": "test", "case_ids": ["q1", "q2"]},
                bootstrap_samples=100,
                seed=1,
            )
            self.assertEqual(
                ("baseline", "tools", "tools-rag"), tuple(analysis["systems"])
            )
            tools = analysis["comparisons"]["tools_gain"]
            self.assertEqual(20.0, tools["mean_case_averaged_effect"])
            self.assertEqual(
                2,
                tools["case_direction_consistency"]["positive_all_replicates"],
            )
            rag = analysis["comparisons"]["rag_effect"]
            self.assertEqual(1.5, rag["mean_case_averaged_effect"])
            self.assertEqual(2, rag["case_direction_consistency"]["mixed_direction"])

            output = write_repeatability_analysis(root / "analysis", analysis)
            self.assertTrue((output / "analysis.json").exists())
            self.assertTrue((output / "run_summary.csv").exists())
            self.assertTrue((output / "case_summary.csv").exists())
            self.assertTrue((output / "comparison_summary.csv").exists())
            self.assertIn(
                "按题先平均",
                (output / "RESULTS_CN.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
