from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate import cli  # noqa: E402
from auto_evaluate.codex_automation import (  # noqa: E402
    parse_json_object,
    run_codex_tasks,
    validate_judge_output,
)
from auto_evaluate.io_utils import read_jsonl, write_jsonl  # noqa: E402
from auto_evaluate.codex_worker import run as run_worker  # noqa: E402


def _teacher_task() -> dict:
    return {
        "task_id": "teacher::q1",
        "case_id": "q1",
        "question": "question",
        "expected_output": {},
    }


def _teacher_output() -> dict:
    return {
        "task_id": "teacher::q1",
        "case_id": "q1",
        "system_id": "gpt-5.6-teacher",
        "response_text": "complete answer",
    }


def _judge_task() -> dict:
    return {
        "task_id": "judge::q1::response-a",
        "case_id": "q1",
        "candidate_label": "Response A",
        "rubric": {
            "steps": [
                {"step_id": 1, "max_score": 40},
                {"step_id": 2, "max_score": 60},
            ]
        },
    }


def _rating() -> dict:
    return {
        "task_id": "judge::q1::response-a",
        "case_id": "q1",
        "candidate_label": "Response A",
        "total_score": 100,
        "steps": [
            {
                "step_id": 1,
                "score": 40,
                "max_score": 40,
                "evidence": "evidence",
                "diagnosis": "correct",
                "failure_codes": [],
            },
            {
                "step_id": 2,
                "score": 60,
                "max_score": 60,
                "evidence": "evidence",
                "diagnosis": "correct",
                "failure_codes": [],
            },
        ],
        "overall_diagnosis": "complete",
        "skill_improvement_suggestions": [],
    }


class CodexAutomationTests(unittest.TestCase):
    def test_parse_json_object_accepts_json_fence(self):
        self.assertEqual({"ok": True}, parse_json_object("```json\n{\"ok\": true}\n```"))

    def test_judge_validation_requires_every_rubric_step(self):
        rating = _rating()
        rating["steps"].pop()
        rating["total_score"] = 40
        errors = validate_judge_output(_judge_task(), rating)
        self.assertTrue(any("missing rubric step" in error for error in errors))

    def test_codex_task_cache_uses_input_hash(self):
        calls = []

        def fake_invoker(request, _root, _timeout):
            calls.append(request)
            return {
                "ok": True,
                "output": _teacher_output(),
                "thread_id": "thread-1",
                "attempts": 1,
                "latency_ms": 2,
            }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            run_dir = Path(tmp) / "runs" / "r1"
            kwargs = {
                "stage": "teacher",
                "tasks": [_teacher_task()],
                "run_dir": run_dir,
                "project_root": Path(tmp),
                "model": "gpt-5.6-terra",
                "invoker": fake_invoker,
            }
            first = run_codex_tasks(**kwargs)
            second = run_codex_tasks(**kwargs)

        self.assertEqual([_teacher_output()], first)
        self.assertEqual(first, second)
        self.assertEqual(1, len(calls))

    def test_worker_repairs_invalid_json_in_the_same_thread(self):
        expected = _teacher_output()

        class FakeThread:
            id = "thread-repair"

            def __init__(self):
                self.calls = []

            def run(self, prompt):
                self.calls.append(prompt)
                response = "not json" if len(self.calls) == 1 else __import__("json").dumps(expected)
                return type("Result", (), {"final_response": response})()

        thread = FakeThread()

        class FakeCodex:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def thread_start(self, *, model, sandbox):
                self.model = model
                self.sandbox = sandbox
                return thread

        fake_module = types.ModuleType("openai_codex")
        fake_module.Codex = FakeCodex
        fake_module.Sandbox = type("Sandbox", (), {"read_only": "read-only", "workspace_write": "workspace-write"})
        request = {
            "stage": "teacher",
            "task": _teacher_task(),
            "prompt": "initial",
            "retries": 2,
            "model": "gpt-5.6-terra",
        }
        with patch.dict(sys.modules, {"openai_codex": fake_module}):
            result = run_worker(request)

        self.assertEqual(expected, result["output"])
        self.assertEqual("thread-repair", result["thread_id"])
        self.assertEqual(2, result["attempts"])
        self.assertEqual(2, len(thread.calls))

    def test_auto_command_writes_teacher_and_rating_outputs(self):
        teacher_task = _teacher_task()
        judge_task = _judge_task()

        def prepare_teacher(_benchmarks, run_dir):
            path = run_dir / "teacher_batch.jsonl"
            write_jsonl(path, [teacher_task])
            return path

        def prepare_judge(_benchmarks, run_dir, seed):
            self.assertEqual(7, seed)
            path = run_dir / "judge_batch.jsonl"
            write_jsonl(path, [judge_task])
            return path

        def fake_codex(**kwargs):
            return [_teacher_output()] if kwargs["stage"] == "teacher" else [_rating()]

        args = argparse.Namespace(
            run_id="auto-test",
            benchmarks_dir=None,
            benchmark_set="single",
            systems="configs/systems.json",
            force=False,
            force_codex=False,
            seed=7,
            output=None,
            config="configs/skill_promotion.json",
            codex_model="gpt-5.6-terra",
            codex_concurrency=1,
            codex_retries=2,
            codex_timeout=60,
            fail_on_gate=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmarks = root / "benchmarks"
            benchmarks.mkdir()
            with (
                patch("auto_evaluate.cli._root", return_value=root),
                patch("auto_evaluate.cli._prepare_benchmarks", return_value=(benchmarks, "single")),
                patch("auto_evaluate.cli.command_validate_benchmarks", return_value=0),
                patch("auto_evaluate.cli.command_probe", return_value=0),
                patch("auto_evaluate.cli.command_run", return_value=0),
                patch("auto_evaluate.cli.prepare_teacher_tasks", side_effect=prepare_teacher),
                patch("auto_evaluate.cli.prepare_judge_tasks", side_effect=prepare_judge),
                patch("auto_evaluate.cli.run_codex_tasks", side_effect=fake_codex),
                patch("auto_evaluate.cli.command_validate_ratings", return_value=0),
                patch("auto_evaluate.cli.command_report", return_value=0),
                patch("auto_evaluate.cli.command_skill_gate", return_value=1),
            ):
                result = cli.command_auto(args)
            run_dir = root / "runs" / "auto-test"
            self.assertEqual([_teacher_output()], read_jsonl(run_dir / "teacher_responses.jsonl"))
            self.assertEqual([_rating()], read_jsonl(run_dir / "ratings.jsonl"))
            self.assertEqual(0, result)


if __name__ == "__main__":
    unittest.main()
