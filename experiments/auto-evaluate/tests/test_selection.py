from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate.io_utils import read_json, write_json  # noqa: E402
from auto_evaluate.selection import prepare_selected_argv  # noqa: E402


class SelectionTests(unittest.TestCase):
    def test_probe_chat_case_flag_is_left_for_the_existing_cli(self):
        argv = ["probe-chat", "--system", "baseline", "--case", "q1"]
        self.assertEqual(argv, prepare_selected_argv(argv, Path("unused")))

    def test_repeatable_case_flags_create_a_run_local_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "benchmarks" / "normalized"
            normalized.mkdir(parents=True)
            write_json(
                root / "configs" / "benchmark_sets.json",
                {
                    "active_set": "all",
                    "sets": {"all": {"normalized_dir": "benchmarks/normalized"}},
                },
            )
            cases = [
                {"case_id": "q1", "file": "q1.json"},
                {"case_id": "q2", "file": "q2.json"},
            ]
            write_json(normalized / "index.json", {"schema_version": "1.0", "cases": cases})
            write_json(normalized / "q1.json", {"case_id": "q1"})
            write_json(normalized / "q2.json", {"case_id": "q2"})

            argv = prepare_selected_argv(
                ["auto", "--run-id", "smoke", "--case", "q2", "--case=q1"],
                root,
            )

            self.assertNotIn("--case", argv)
            selection = read_json(root / "runs" / "smoke" / "benchmarks" / "index.json")
            self.assertEqual(["q2", "q1"], selection["selection"]["case_ids"])
            self.assertEqual(["q2", "q1"], [row["case_id"] for row in selection["cases"]])
            self.assertTrue((root / "runs" / "smoke" / "benchmarks" / "q1.json").exists())
            self.assertTrue((root / "runs" / "smoke" / "benchmarks" / "q2.json").exists())

    def test_unknown_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "configs" / "benchmark_sets.json",
                {
                    "active_set": "all",
                    "sets": {"all": {"normalized_dir": "benchmarks/normalized"}},
                },
            )
            write_json(
                root / "benchmarks" / "normalized" / "index.json",
                {"schema_version": "1.0", "cases": []},
            )
            with self.assertRaisesRegex(ValueError, "Unknown benchmark case ID"):
                prepare_selected_argv(
                    ["run", "--run-id", "bad", "--case", "missing"],
                    root,
                )


if __name__ == "__main__":
    unittest.main()
