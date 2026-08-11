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
from auto_evaluate.runner import execute_run  # noqa: E402
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
    def test_prepare_benchmarks_uses_active_set_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source_config = temp / "configs" / "benchmarks_single.json"
            registry_path = temp / "configs" / "benchmark_sets.json"
            workbook = (ROOT / "benchmarks" / "Datasets Harness" / "D1" / "D1_1a .xlsx").resolve()

            source_config.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                source_config,
                {
                    "schema_version": "1.0",
                    "sources": [
                        {
                            "case_id": "D1-1a-swro-membrane-area-design",
                            "path": str(workbook),
                            "task_family": "d1_1a",
                        }
                    ],
                },
            )
            write_json(
                registry_path,
                {
                    "active_set": "single",
                    "sets": {
                        "single": {
                            "description": "single benchmark",
                            "source_config": "configs/benchmarks_single.json",
                            "normalized_dir": "benchmarks/normalized",
                        }
                    },
                },
            )

            args = argparse.Namespace(benchmarks_dir=None, benchmark_set=None)
            with patch("auto_evaluate.cli._root", return_value=temp):
                benchmarks_dir, selected = cli._prepare_benchmarks(args, sync=True)

            self.assertEqual("single", selected)
            self.assertTrue((benchmarks_dir / "index.json").exists())
            cases = read_json(benchmarks_dir / "index.json")["cases"]
            self.assertEqual(["D1-1a-swro-membrane-area-design"], [row["case_id"] for row in cases])

    def test_execute_run_honors_system_generation_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "runs" / "pilot"
            benchmark = {
                "case_id": "q1",
                "question_prompt": "question",
                "source": {"sha256": "case-sha"},
            }
            config = {
                "shared_system_prompt": "system",
                "generation": {
                    "stream": True,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 64,
                    "timeout_seconds": 10,
                    "max_retries": 0,
                },
                "systems": [
                    {
                        "id": "baseline",
                        "display_name": "Baseline",
                        "model_id": "baseline-model",
                        "generation_overrides": {"stream": False},
                    },
                    {
                        "id": "environment",
                        "display_name": "Environment",
                        "model_id": "environment-model",
                    },
                ],
            }

            class _Client:
                def __init__(self):
                    self.calls = []

                def chat(self, *, model, messages, generation):
                    self.calls.append(("chat", model, generation["stream"]))
                    return type("Result", (), {"content": "baseline answer", "raw": {"usage": None}, "latency_ms": 1})()

                def chat_stream(self, *, model, messages, generation):
                    self.calls.append(("chat_stream", model, generation["stream"]))
                    return type("Result", (), {"content": "environment answer", "raw": {"usage": None}, "latency_ms": 2})()

            client = _Client()
            with (
                patch("auto_evaluate.runner.load_systems", return_value=config),
                patch("auto_evaluate.runner.make_client", return_value=client),
                patch("auto_evaluate.runner.iter_benchmarks", return_value=iter([benchmark])),
            ):
                counts = execute_run(
                    benchmarks_dir=temp / "benchmarks",
                    systems_path=temp / "configs" / "systems.json",
                    run_dir=run_dir,
                )

            self.assertEqual({"success": 2, "error": 0, "skipped": 0}, counts)
            self.assertEqual(
                [("chat", "baseline-model", False), ("chat_stream", "environment-model", True)],
                client.calls,
            )
            baseline_record = read_json(run_dir / "responses" / "q1__baseline.json")
            environment_record = read_json(run_dir / "responses" / "q1__environment.json")
            self.assertFalse(baseline_record["generation"]["stream"])
            self.assertTrue(environment_record["generation"]["stream"])

    def test_extract_score_points_parses_structured_trailer(self):
        response = """Final answer text.
[SCORE_POINTS_BEGIN]
{"task_type":"design_pressure","decision_variables":["feed_pressure_bar"],"fixed_inputs":{"temperature_c":32},"tool_calls":[],"constraint_checks":[],"final_answer":{"recommendation":"51 bar"}}
[SCORE_POINTS_END]
"""
        parsed = extract_score_points(response)
        self.assertEqual("ok", parsed["status"])
        self.assertEqual("design_pressure", parsed["data"]["task_type"])

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
                    {"task_id": "base-q1", "case_id": "q1", "system_id": "environment"},
                    {"task_id": "skill-q1", "case_id": "q1", "system_id": "environment-skill"},
                    {"task_id": "base-q2", "case_id": "q2", "system_id": "environment"},
                    {"task_id": "skill-q2", "case_id": "q2", "system_id": "environment-skill"},
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
