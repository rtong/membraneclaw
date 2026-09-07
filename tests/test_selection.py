from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate.io_utils import read_json, sha256_file, write_json  # noqa: E402
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

    def test_case_file_creates_and_then_reuses_frozen_selection(self):
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
            write_json(
                normalized / "index.json",
                {
                    "schema_version": "1.0",
                    "cases": [
                        {"case_id": "q1", "file": "q1.json"},
                        {"case_id": "q2", "file": "q2.json"},
                    ],
                },
            )
            write_json(normalized / "q1.json", {"case_id": "q1"})
            write_json(normalized / "q2.json", {"case_id": "q2"})
            case_file = root / "configs" / "repeat.json"
            write_json(case_file, {"selection_id": "repeat", "case_ids": ["q2", "q1"]})

            argv = prepare_selected_argv(
                [
                    "auto",
                    "--run-id",
                    "repeat-r2",
                    "--benchmark-set",
                    "all",
                    "--case-file",
                    "configs/repeat.json",
                ],
                root,
            )
            self.assertNotIn("--case-file", argv)
            index_path = root / "runs" / "repeat-r2" / "benchmarks" / "index.json"
            first = index_path.read_bytes()
            selection = read_json(index_path)["selection"]
            self.assertEqual(["q2", "q1"], selection["case_ids"])
            self.assertEqual(sha256_file(case_file), selection["case_file_sha256"])

            prepare_selected_argv(
                [
                    "auto",
                    "--run-id",
                    "repeat-r2",
                    "--benchmark-set",
                    "all",
                    "--case-file=configs/repeat.json",
                ],
                root,
            )
            self.assertEqual(first, index_path.read_bytes())

    def test_case_file_can_read_cases_from_a_frozen_source_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "benchmarks" / "normalized"
            frozen = root / "runs" / "frozen-v1" / "benchmarks"
            current.mkdir(parents=True)
            frozen.mkdir(parents=True)
            write_json(
                root / "configs" / "benchmark_sets.json",
                {
                    "active_set": "all",
                    "sets": {"all": {"normalized_dir": "benchmarks/normalized"}},
                },
            )
            write_json(current / "index.json", {"schema_version": "1.0", "cases": []})
            write_json(
                frozen / "index.json",
                {
                    "schema_version": "1.0",
                    "cases": [{"case_id": "q1", "file": "q1.json"}],
                },
            )
            write_json(frozen / "q1.json", {"case_id": "q1", "source": "frozen"})
            case_file = root / "configs" / "repeat.json"
            write_json(
                case_file,
                {
                    "case_ids": ["q1"],
                    "source_benchmarks_dir": "runs/frozen-v1/benchmarks",
                },
            )

            prepare_selected_argv(
                [
                    "auto",
                    "--run-id",
                    "repeat-r2",
                    "--benchmark-set",
                    "all",
                    "--case-file",
                    "configs/repeat.json",
                ],
                root,
            )

            copied = read_json(root / "runs" / "repeat-r2" / "benchmarks" / "q1.json")
            selection = read_json(
                root / "runs" / "repeat-r2" / "benchmarks" / "index.json"
            )["selection"]
            self.assertEqual("frozen", copied["source"])
            self.assertEqual(str(frozen.resolve()), selection["source_normalized_dir"])


if __name__ == "__main__":
    unittest.main()
