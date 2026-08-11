from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate.benchmark import import_all, iter_benchmarks  # noqa: E402


class BenchmarkImportTests(unittest.TestCase):
    def test_imports_all_configured_workbooks_with_100_point_rubrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "normalized"
            imported = import_all(ROOT / "configs/benchmarks.json", output, ROOT)
            configured = json.loads((ROOT / "configs/benchmarks.json").read_text(encoding="utf-8"))
            self.assertEqual(len(configured["sources"]), len(imported))
            self.assertTrue(all(item["rubric"]["steps"] for item in imported))
            self.assertTrue(all(item["rubric"]["total_points"] == 100 for item in imported))
            loaded = list(iter_benchmarks(output))
            self.assertEqual([item["case_id"] for item in imported], [item["case_id"] for item in loaded])

    def test_question_prompt_excludes_answer_and_rubric_sheet_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "normalized"
            imported = import_all(ROOT / "configs/benchmarks.json", output, ROOT)
            for item in imported:
                self.assertTrue(item["question_prompt"].strip())
                self.assertNotEqual(item["question_prompt"], item["reference_answer"])
                for step in item["rubric"]["steps"]:
                    for field in ("full_credit", "common_failures"):
                        snippet = step.get(field, "")
                        if len(snippet) >= 6:
                            self.assertNotIn(snippet, item["question_prompt"])

    def test_skill_does_not_contain_pilot_reference_answers(self):
        skill_root = ROOT / "skills/swro-watertap"
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in skill_root.rglob("*")
            if path.is_file() and path.suffix in {".md", ".json"}
        )
        for leaked_value in (
            "5.05",
            "5.10 MPa",
            "3.726",
            "396.895",
            "98.72%",
            "0.00020 mol/s",
            "0.00024 mol/s",
            "0.071423",
            "-0.051875",
        ):
            self.assertNotIn(leaked_value, text)


if __name__ == "__main__":
    unittest.main()
