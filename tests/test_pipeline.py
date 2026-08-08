from __future__ import annotations

import http.client
import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate.benchmark import import_all  # noqa: E402
from auto_evaluate import cli  # noqa: E402
from auto_evaluate.io_utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from auto_evaluate.judge import extract_score_points, prepare_judge_tasks, prepare_teacher_tasks, validate_ratings  # noqa: E402
from auto_evaluate.openwebui import OpenWebUIClient, OpenWebUIError  # noqa: E402
from auto_evaluate.report import build_report  # noqa: E402
from auto_evaluate.skill_gate import evaluate_skill_gate  # noqa: E402


class _FakeResponse:
    def __init__(self, value):
        self.payload = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload

    def __iter__(self):
        return iter(self.payload.splitlines(keepends=True))


class _FakeStreamResponse(_FakeResponse):
    def __init__(self, payload: bytes):
        self.payload = payload


class PipelineTests(unittest.TestCase):
    def test_extract_score_points_parses_structured_trailer(self):
        response = """Final answer text.
[SCORE_POINTS_BEGIN]
{"task_type":"design_pressure","decision_variables":["feed_pressure_bar"],"fixed_inputs":{"temperature_c":32},"tool_calls":[],"constraint_checks":[],"final_answer":{"recommendation":"51 bar"}}
[SCORE_POINTS_END]
"""
        parsed = extract_score_points(response)
        self.assertEqual("ok", parsed["status"])
        self.assertEqual("design_pressure", parsed["data"]["task_type"])

    def test_cycle_waits_for_teacher_and_judge_then_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            benchmarks_dir = temp / "benchmarks" / "normalized"
            run_dir = temp / "runs" / "pilot-cycle"
            imported = import_all(ROOT / "configs/benchmarks.json", benchmarks_dir, ROOT)
            run_dir.mkdir(parents=True)
            benchmark = read_json(benchmarks_dir / f"{imported[0]['case_id']}.json")
            write_json(
                run_dir / "responses" / "q1__agent.json",
                {
                    "status": "success",
                    "case_id": benchmark["case_id"],
                    "system_id": "agent",
                    "display_name": "Agent",
                    "response_text": "A structured candidate response",
                    "latency_ms": 123,
                },
            )

            def fake_wait(path: Path, _poll_seconds: float) -> None:
                if path.name == "teacher_responses.jsonl":
                    write_jsonl(
                        path,
                        [
                            {
                                "task_id": f"teacher::{benchmark['case_id']}",
                                "case_id": benchmark["case_id"],
                                "system_id": "gpt-5.6-teacher",
                                "response_text": "Teacher answer",
                            }
                        ],
                    )
                    return
                if path.name == "ratings.jsonl":
                    tasks = read_jsonl(run_dir / "judge_batch.jsonl")
                    ratings = []
                    for task in tasks:
                        ratings.append(
                            {
                                "task_id": task["task_id"],
                                "case_id": task["case_id"],
                                "candidate_label": task["candidate_label"],
                                "total_score": 100,
                                "steps": [
                                    {
                                        "step_id": step["step_id"],
                                        "score": step["max_score"],
                                        "max_score": step["max_score"],
                                        "evidence": "evidence",
                                        "diagnosis": "complete",
                                        "failure_codes": [],
                                    }
                                    for step in task["rubric"]["steps"]
                                ],
                                "overall_diagnosis": "complete",
                                "skill_improvement_suggestions": [],
                            }
                        )
                    write_jsonl(
                        path,
                        ratings,
                    )
                    return
                raise AssertionError(f"Unexpected wait target: {path}")

            args = argparse.Namespace(
                run_id="pilot-cycle",
                benchmarks_dir="benchmarks/normalized",
                systems="configs/systems.json",
                force=False,
                seed=7,
                poll_seconds=0.0,
                output=None,
                config="configs/skill_promotion.json",
            )

            with (
                patch("auto_evaluate.cli._root", return_value=temp),
                patch("auto_evaluate.cli.command_probe", return_value=0),
                patch("auto_evaluate.cli.command_run", return_value=0),
                patch("auto_evaluate.cli.command_report", return_value=0),
                patch("auto_evaluate.cli.command_skill_gate", return_value=0),
                patch("auto_evaluate.cli._wait_for_file", side_effect=fake_wait) as wait_mock,
            ):
                result = cli.command_cycle(args)

            self.assertEqual(0, result)
            self.assertEqual(2, wait_mock.call_count)
            self.assertTrue((run_dir / "teacher_batch.jsonl").exists())
            self.assertTrue((run_dir / "judge_batch.jsonl").exists())
            self.assertTrue((run_dir / "teacher_responses.jsonl").exists())
            self.assertTrue((run_dir / "ratings.jsonl").exists())

    def test_cycle_resumes_without_waiting_when_manual_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "runs" / "pilot-resume"
            run_dir.mkdir(parents=True)
            write_jsonl(run_dir / "teacher_responses.jsonl", [{"case_id": "q1", "response_text": "done"}])
            write_jsonl(run_dir / "ratings.jsonl", [{"task_id": "judge::q1::response-a", "total_score": 100, "steps": []}])

            args = argparse.Namespace(
                run_id="pilot-resume",
                benchmarks_dir="benchmarks/normalized",
                systems="configs/systems.json",
                force=False,
                seed=7,
                poll_seconds=0.0,
                output=None,
                config="configs/skill_promotion.json",
            )

            with (
                patch("auto_evaluate.cli._root", return_value=temp),
                patch("auto_evaluate.cli.command_validate_benchmarks", return_value=0),
                patch("auto_evaluate.cli.command_probe", return_value=0),
                patch("auto_evaluate.cli.command_run", return_value=0),
                patch("auto_evaluate.cli.command_validate_ratings", return_value=0),
                patch("auto_evaluate.cli.command_report", return_value=0),
                patch("auto_evaluate.cli.command_skill_gate", return_value=0),
                patch("auto_evaluate.cli.prepare_teacher_tasks") as teacher_mock,
                patch("auto_evaluate.cli.prepare_judge_tasks") as judge_mock,
                patch("auto_evaluate.cli._wait_for_file") as wait_mock,
            ):
                result = cli.command_cycle(args)

            self.assertEqual(0, result)
            teacher_mock.assert_not_called()
            judge_mock.assert_not_called()
            wait_mock.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_openwebui_client_parses_streaming_completion(self, urlopen):
        urlopen.return_value = _FakeStreamResponse(
            b'data: {"choices":[{"delta":{"content":"O"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"K"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        client = OpenWebUIClient("https://example.test", "secret", timeout_seconds=5)
        result = client.chat_stream(
            model="agent",
            messages=[{"role": "user", "content": "question"}],
            generation={"temperature": 0, "top_p": 1, "max_tokens": 100},
        )
        self.assertEqual("OK", result.content)
        self.assertEqual(2, result.raw["event_count"])
        request = urlopen.call_args.args[0]
        self.assertEqual("text/event-stream", request.headers["Accept"])

    @patch("urllib.request.urlopen")
    def test_openwebui_wraps_remote_disconnect(self, urlopen):
        urlopen.side_effect = http.client.RemoteDisconnected("closed without response")
        client = OpenWebUIClient("https://example.test", "secret", timeout_seconds=5)
        with self.assertRaisesRegex(OpenWebUIError, "connection failed"):
            client.list_models()

    @patch("urllib.request.urlopen")
    def test_openwebui_client_parses_chat_completion(self, urlopen):
        urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "answer"}}], "usage": {"total_tokens": 9}}
        )
        client = OpenWebUIClient("https://example.test", "secret", timeout_seconds=5)
        result = client.chat(
            model="agent",
            messages=[{"role": "user", "content": "question"}],
            generation={"temperature": 0, "top_p": 1, "max_tokens": 100},
        )
        self.assertEqual("answer", result.content)
        request = urlopen.call_args.args[0]
        self.assertEqual("Bearer secret", request.headers["Authorization"])
        self.assertTrue(request.full_url.endswith("/api/chat/completions"))

    def test_teacher_judge_validation_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            benchmarks_dir = temp / "benchmarks"
            run_dir = temp / "runs" / "pilot"
            imported = import_all(ROOT / "configs/benchmarks.json", benchmarks_dir, ROOT)
            run_dir.mkdir(parents=True)
            write_json(run_dir / "manifest.json", {"run_id": "pilot"})

            benchmark = read_json(benchmarks_dir / f"{imported[0]['case_id']}.json")
            write_json(
                run_dir / "responses" / "q1__agent.json",
                {
                    "status": "success",
                    "case_id": benchmark["case_id"],
                    "system_id": "agent",
                    "display_name": "Agent",
                    "response_text": "A structured candidate response",
                    "latency_ms": 123,
                },
            )
            teacher_path = prepare_teacher_tasks(benchmarks_dir, run_dir)
            self.assertEqual(len(imported), len(read_jsonl(teacher_path)))

            judge_path = prepare_judge_tasks(benchmarks_dir, run_dir, seed=7)
            tasks = read_jsonl(judge_path)
            self.assertEqual(1, len(tasks))
            task = tasks[0]
            rating = {
                "task_id": task["task_id"],
                "case_id": task["case_id"],
                "candidate_label": task["candidate_label"],
                "total_score": 100,
                "steps": [
                    {
                        "step_id": step["step_id"],
                        "score": step["max_score"],
                        "max_score": step["max_score"],
                        "evidence": "evidence",
                        "diagnosis": "complete",
                        "failure_codes": [],
                    }
                    for step in task["rubric"]["steps"]
                ],
                "overall_diagnosis": "complete",
                "skill_improvement_suggestions": [],
            }
            write_jsonl(run_dir / "ratings.jsonl", [rating])
            self.assertEqual([], validate_ratings(run_dir))
            report = build_report(run_dir)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Agent", text)
            self.assertIn("100.0", text)
            self.assertIn("Step-level score", text)
            self.assertIn('class="benchmark-card"', text)
            self.assertIn("Scalable benchmark explorer", text)

    def test_skill_gate_requires_gain_on_every_case_and_no_argument_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            mapping = {
                "mapping": [
                    {"task_id": "base-q1", "case_id": "q1", "system_id": "agent-rag"},
                    {"task_id": "skill-q1", "case_id": "q1", "system_id": "agent-rag-skill"},
                    {"task_id": "base-q2", "case_id": "q2", "system_id": "agent-rag"},
                    {"task_id": "skill-q2", "case_id": "q2", "system_id": "agent-rag-skill"},
                ]
            }
            write_json(run_dir / "judge_mapping.json", mapping)
            write_jsonl(
                run_dir / "ratings.jsonl",
                [
                    {"task_id": "base-q1", "case_id": "q1", "total_score": 70, "steps": []},
                    {"task_id": "skill-q1", "case_id": "q1", "total_score": 71, "steps": []},
                    {"task_id": "base-q2", "case_id": "q2", "total_score": 60, "steps": []},
                    {"task_id": "skill-q2", "case_id": "q2", "total_score": 69, "steps": []},
                ],
            )
            result = evaluate_skill_gate(run_dir)
            self.assertTrue(result["passed"])
            self.assertEqual(5.0, result["mean_gain"])

            ratings = read_jsonl(run_dir / "ratings.jsonl")
            ratings[-1]["steps"] = [{"failure_codes": ["TOOL_ARGUMENT"]}]
            write_jsonl(run_dir / "ratings.jsonl", ratings)
            result = evaluate_skill_gate(run_dir)
            self.assertFalse(result["passed"])
            self.assertEqual({"TOOL_ARGUMENT": 1}, result["forbidden_failure_codes_found"])


if __name__ == "__main__":
    unittest.main()
