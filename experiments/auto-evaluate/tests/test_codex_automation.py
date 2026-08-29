from __future__ import annotations

import argparse
import base64
import json
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
    _REQUEST_B64_PREFIX,
    _RESULT_B64_PREFIX,
    build_task_prompt,
    invoke_codex_worker,
    parse_json_object,
    run_codex_tasks,
    validate_judge_output,
)
from auto_evaluate.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
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
        "tool_efficiency_rubric": {
            "total_points": 100,
            "dimensions": [
                {"dimension_id": "E1", "max_score": 40},
                {"dimension_id": "E2", "max_score": 60},
            ],
        },
        "observable_trajectory": {
            "source": "visible_response_transcript",
            "events": [
                {"event_id": "T001", "event_type": "tool_interaction"},
                {"event_id": "FINAL_RESPONSE", "event_type": "final_response"},
            ],
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
        "tool_efficiency_score": 100,
        "tool_efficiency_dimensions": [
            {
                "dimension_id": "E1",
                "score": 40,
                "max_score": 40,
                "evidence": "necessary tool call",
                "diagnosis": "efficient",
            },
            {
                "dimension_id": "E2",
                "score": 60,
                "max_score": 60,
                "evidence": "high information gain",
                "diagnosis": "efficient",
            },
        ],
        "tool_efficiency_overall_diagnosis": "efficient and sufficient",
        "trajectory_analysis": {
            "trajectory_source": "visible_response_transcript",
            "path_classification": "valid_alternative",
            "summary": "the observable path is valid",
            "first_error_event_id": None,
            "recovery_attempted": False,
            "recovery_succeeded": None,
            "event_assessments": [
                {
                    "event_id": "T001",
                    "verdict": "correct",
                    "failure_codes": [],
                    "primary_failure_code": None,
                    "evidence": "observable tool interaction",
                    "diagnosis": "correct action and observation",
                    "affected_rubric_steps": [1, 2],
                    "attributed_task_loss": 0,
                }
            ],
        },
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

    def test_judge_validation_accepts_research_causal_analysis(self):
        rating = _rating()
        rating["causal_analysis"] = {
            "first_error_step_id": 2,
            "root_cause": "threshold comparison failed",
            "error_propagation": ["wrong pass/fail", "wrong boundary"],
            "downstream_affected_steps": [2],
            "minimal_fix": "compare the unrounded value",
            "counterfactual_outcome": "the correct boundary would be searched",
            "evidence_strength": "direct",
        }
        rating["research_tags"] = ["threshold_comparison", "error_propagation"]
        self.assertEqual([], validate_judge_output(_judge_task(), rating))
    def test_new_judge_task_requires_research_diagnostics(self):
        task = _judge_task()
        task["expected_output"] = {"causal_analysis": {}, "research_tags": [], "trajectory_analysis": {}}
        rating = _rating()
        rating.pop("trajectory_analysis")
        errors = validate_judge_output(task, rating)
        self.assertIn("causal_analysis is required by this judge task", errors)
        self.assertIn("research_tags is required by this judge task", errors)
        self.assertIn("trajectory_analysis is required by this judge task", errors)
    def test_trajectory_loss_must_equal_task_loss(self):
        task = _judge_task()
        task["expected_output"] = {"trajectory_analysis": {}}
        rating = _rating()
        rating["total_score"] = 90
        rating["steps"][1]["score"] = 50
        errors = validate_judge_output(task, rating)
        self.assertTrue(any("trajectory attributed task loss" in error for error in errors))

    def test_trajectory_rejects_unknown_event_id(self):
        task = _judge_task()
        task["expected_output"] = {"trajectory_analysis": {}}
        rating = _rating()
        rating["trajectory_analysis"]["event_assessments"][0]["event_id"] = "INVENTED"
        errors = validate_judge_output(task, rating)
        self.assertTrue(any("observable event ID" in error for error in errors))

    def test_teacher_prompt_allows_required_watertap_mcp(self):
        task = {
            **_teacher_task(),
            "tool_policy": {
                "required": True,
                "mcp_server": "watertap",
                "require_observable_call": True,
            },
        }
        prompt = build_task_prompt("teacher", task)
        self.assertIn("allowed to call the configured MCP server `watertap`", prompt)
        self.assertNotIn(
            "Do not inspect files, run commands, browse, or call tools.",
            prompt,
        )

    def test_teacher_rejects_non_watertap_tool_trajectory(self):
        from auto_evaluate.codex_automation import validate_teacher_output

        task = {
            **_teacher_task(),
            "tool_policy": {
                "mcp_server": "watertap",
                "require_observable_call": True,
            },
        }
        output = {
            **_teacher_output(),
            "trajectory": {
                "events": [
                    {
                        "event_type": "tool_interaction",
                        "tool_name": "node_repl.evaluate",
                        "metadata": {"server": "node_repl"},
                    }
                ]
            },
        }
        errors = validate_teacher_output(task, output)
        self.assertIn(
            "teacher must make at least one observable watertap tool call before answering",
            errors,
        )
    def test_connected_app_does_not_require_project_token_env(self):
        task = {
            **_teacher_task(),
            "tool_policy": {
                "required": True,
                "mcp_server": "watertap",
                "require_observable_call": True,
            },
        }
        output = {
            **_teacher_output(),
            "trajectory": {
                "events": [
                    {
                        "event_type": "tool_interaction",
                        "tool_name": "watertap.simulate_ro",
                        "observation": {"permeate_flow": 6.2},
                        "status": "success",
                        "metadata": {"server": "watertap"},
                    }
                ]
            },
        }
        calls = []

        def fake_invoker(*_args):
            calls.append(True)
            return {
                "ok": True,
                "output": output,
                "thread_id": "thread-app",
                "attempts": 1,
                "latency_ms": 2,
            }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            result = run_codex_tasks(
                stage="teacher",
                tasks=[task],
                run_dir=Path(tmp) / "run",
                project_root=Path(tmp),
                model="gpt-5.6-sol",
                invoker=fake_invoker,
            )
        self.assertEqual([output], result)
        self.assertEqual([True], calls)

    def test_teacher_rejects_failed_watertap_call_without_observation(self):
        from auto_evaluate.codex_automation import validate_teacher_output

        task = {
            **_teacher_task(),
            "tool_policy": {
                "mcp_server": "watertap",
                "require_observable_call": True,
                "require_successful_observation": True,
            },
        }
        output = {
            **_teacher_output(),
            "trajectory": {
                "events": [
                    {
                        "event_type": "tool_interaction",
                        "tool_name": "watertap.equilibrate_feed",
                        "observation": None,
                        "status": "error",
                        "metadata": {
                            "server": "watertap",
                            "provider_server": "codex_apps",
                        },
                    }
                ]
            },
        }
        errors = validate_teacher_output(task, output)
        self.assertIn(
            "teacher must complete at least one successful watertap tool call "
            "with a non-empty observation before answering",
            errors,
        )

    def test_cli_jsonl_normalizes_completed_codex_app_call(self):
        from auto_evaluate.codex_worker import _parse_codex_cli_jsonl

        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-cli"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item-1",
                            "type": "mcp_tool_call",
                            "server": "codex_apps",
                            "tool": "watertap.equilibrate_feed",
                            "arguments": {"water_recovery": 0.1},
                            "result": {"content": [{"type": "text", "text": "SI=-0.2"}]},
                            "error": None,
                            "status": "completed",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item-2",
                            "type": "agent_message",
                            "text": json.dumps(_teacher_output()),
                        },
                    }
                ),
            ]
        )
        parsed = _parse_codex_cli_jsonl(stdout)
        self.assertEqual("thread-cli", parsed["thread_id"])
        self.assertEqual(json.dumps(_teacher_output()), parsed["final_response"])
        self.assertEqual(1, len(parsed["tool_events"]))
        event = parsed["tool_events"][0]
        self.assertEqual("watertap.equilibrate_feed", event["tool_name"])
        self.assertEqual("watertap", event["metadata"]["server"])
        self.assertEqual("codex_apps", event["metadata"]["provider_server"])
        self.assertEqual("success", event["status"])

    def test_codex_cli_path_falls_back_to_bundled_sdk_runtime(self):
        from auto_evaluate.codex_worker import _codex_cli_path

        with tempfile.TemporaryDirectory() as tmp:
            bundled_cli = Path(tmp) / "codex.exe"
            bundled_cli.touch()
            runtime_module = types.ModuleType("codex_cli_bin")
            runtime_module.bundled_codex_path = lambda: bundled_cli
            with patch.dict(os.environ, {}, clear=True), patch(
                "auto_evaluate.codex_worker.shutil.which", return_value=None
            ), patch.dict(sys.modules, {"codex_cli_bin": runtime_module}):
                self.assertEqual(str(bundled_cli), _codex_cli_path())

    def test_required_tools_teacher_uses_cli_auto_approval(self):
        task = {
            **_teacher_task(),
            "tool_policy": {
                "mcp_server": "watertap",
                "require_observable_call": True,
                "require_successful_observation": True,
            },
        }
        cli_stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-cli"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item-1",
                            "type": "mcp_tool_call",
                            "server": "codex_apps",
                            "tool": "watertap.equilibrate_feed",
                            "arguments": {"water_recovery": 0.1},
                            "result": {"content": [{"type": "text", "text": "SI=-0.2"}]},
                            "error": None,
                            "status": "completed",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item-2",
                            "type": "agent_message",
                            "text": json.dumps(_teacher_output()),
                        },
                    }
                ),
            ]
        )
        completed = types.SimpleNamespace(returncode=0, stdout=cli_stdout, stderr="")
        request = {
            "stage": "teacher",
            "task": task,
            "prompt": "initial",
            "retries": 0,
            "model": "gpt-5.6-sol",
        }
        with patch("auto_evaluate.codex_worker._codex_cli_path", return_value="codex"), patch(
            "auto_evaluate.codex_worker.subprocess.run", return_value=completed
        ) as run_process, patch("auto_evaluate.codex_worker._best_effort_cli_archive"):
            result = run_worker(request)

        command = run_process.call_args.args[0]
        self.assertIn("--approve-for-me", command)
        self.assertNotIn("--sandbox", command)
        self.assertEqual("codex_cli_auto_approve", result["diagnostics"]["executor"])
        trajectory = result["output"]["trajectory"]
        self.assertEqual("success", trajectory["events"][0]["status"])
        self.assertEqual("watertap.equilibrate_feed", trajectory["events"][0]["tool_name"])
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

            def thread_start(self, *, model, sandbox, ephemeral=False):
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

        self.assertEqual(expected, {key: result["output"][key] for key in expected})
        self.assertEqual("final_response_only", result["output"]["trajectory"]["source"])
        self.assertEqual("thread-repair", result["thread_id"])
        self.assertEqual(2, result["attempts"])
        self.assertEqual(2, len(thread.calls))

    def test_session_audit_records_nested_codex_app_tool_call(self):
        from auto_evaluate.codex_worker import _codex_session_tool_events

        record = {
            "type": "event_msg",
            "payload": {
                "type": "mcp_tool_call_end",
                "call_id": "exec-1",
                "invocation": {
                    "server": "codex_apps",
                    "tool": "watertap.simulate_ro",
                    "arguments": {"membrane_area_m2": 400},
                },
                "app_name": "waterTAP",
                "action_name": "simulate_ro",
                "result": {"Ok": {"content": [{"type": "text", "text": "Qp=6.2"}]}},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rollout-thread-nested.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            seen = set()
            events = _codex_session_tool_events(
                "thread-nested",
                seen_call_ids=seen,
                sessions_root=root,
            )
            repeated = _codex_session_tool_events(
                "thread-nested",
                seen_call_ids=seen,
                sessions_root=root,
            )
        self.assertEqual(1, len(events))
        self.assertEqual([], repeated)
        self.assertEqual("watertap.simulate_ro", events[0]["tool_name"])
        self.assertEqual("watertap", events[0]["metadata"]["server"])
        self.assertEqual(400, events[0]["arguments"]["membrane_area_m2"])
    def test_worker_records_mcp_call_as_teacher_trajectory(self):
        expected = _teacher_output()
        task = {
            **_teacher_task(),
            "tool_policy": {
                "required": True,
                "mcp_server": "watertap",
                "require_observable_call": True,
            },
        }

        class FakeThread:
            id = "thread-mcp"

            def run(self, _prompt):
                item = {
                    "type": "mcpToolCall",
                    "server": "watertap",
                    "tool": "simulate_ro",
                    "arguments": {"membrane_area_m2": 400},
                    "result": {"content": [{"type": "text", "text": "Qp=6.2"}]},
                    "status": "completed",
                }
                return type(
                    "Result",
                    (),
                    {"final_response": json.dumps(expected), "items": [item]},
                )()

        class FakeCodex:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def thread_start(self, *, model, sandbox, ephemeral=False):
                return FakeThread()

        fake_module = types.ModuleType("openai_codex")
        fake_module.Codex = FakeCodex
        fake_module.Sandbox = type(
            "Sandbox",
            (),
            {"read_only": "read-only", "workspace_write": "workspace-write"},
        )
        request = {
            "stage": "teacher",
            "task": task,
            "prompt": "initial",
            "retries": 0,
            "model": "gpt-5.6-sol",
        }
        with patch.dict(sys.modules, {"openai_codex": fake_module}), patch.dict(
            os.environ, {"AE_CODEX_TOOL_TEACHER_EXECUTOR": "sdk"}
        ):
            result = run_worker(request)

        trajectory = result["output"]["trajectory"]
        self.assertEqual("api_structured_and_transcript", trajectory["source"])
        self.assertEqual(1, trajectory["summary"]["tool_interactions"])
        self.assertEqual("watertap.simulate_ro", trajectory["events"][0]["tool_name"])
        self.assertEqual(
            400,
            trajectory["events"][0]["arguments"]["membrane_area_m2"],
        )
    @patch("auto_evaluate.codex_automation.subprocess.run")
    def test_invoker_uses_ascii_safe_worker_protocol(self, run_process):
        worker_result = {
            "ok": True,
            "output": {
                **_teacher_output(),
                "response_text": "中文答案\n含引号：\"可行\"",
            },
            "thread_id": "thread-unicode",
            "attempts": 1,
            "latency_ms": 2,
        }
        encoded_result = base64.b64encode(
            json.dumps(worker_result, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        run_process.return_value = type(
            "Completed",
            (),
            {
                "stdout": _RESULT_B64_PREFIX + encoded_result + "\n",
                "stderr": "",
                "returncode": 0,
            },
        )()

        result = invoke_codex_worker(
            {"stage": "teacher", "task": {"question": "中文问题"}},
            ROOT,
            60,
        )

        self.assertEqual(worker_result, result)
        request_payload = run_process.call_args.kwargs["input"]
        self.assertTrue(request_payload.startswith(_REQUEST_B64_PREFIX))
        decoded_request = json.loads(
            base64.b64decode(
                request_payload[len(_REQUEST_B64_PREFIX) :],
                validate=True,
            ).decode("utf-8")
        )
        self.assertEqual("中文问题", decoded_request["task"]["question"])
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
            codex_model="gpt-5.6-terra",
            codex_concurrency=1,
            codex_retries=2,
            codex_timeout=60,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmarks = root / "benchmarks"
            benchmarks.mkdir()
            write_json(
                root / "configs" / "evaluation_profiles.json",
                {
                    "default_profile": "single",
                    "profiles": {
                        "single": {
                            "system_ids": ["baseline"],
                            "teachers": [
                                {
                                    "id": "tools",
                                    "system_id": "gpt-5.6-teacher",
                                    "display_name": "GPT-5.6 Teacher",
                                    "tools_enabled": True,
                                }
                            ],
                            "comparisons": [],
                        }
                    },
                },
            )
            with (
                patch("auto_evaluate.cli._root", return_value=root),
                patch("auto_evaluate.cli._prepare_benchmarks", return_value=(benchmarks, "single")),
                patch("auto_evaluate.cli.command_validate_benchmarks", return_value=0),
                patch("auto_evaluate.cli.command_probe", return_value=0),
                patch("auto_evaluate.cli.command_run", return_value=1),
                patch("auto_evaluate.cli.prepare_teacher_tasks", side_effect=prepare_teacher),
                patch("auto_evaluate.cli.prepare_judge_tasks", side_effect=prepare_judge),
                patch("auto_evaluate.cli.run_codex_tasks", side_effect=fake_codex),
                patch("auto_evaluate.cli.command_validate_ratings", return_value=0),
                patch("auto_evaluate.cli.command_report", return_value=0),
            ):
                result = cli.command_auto(args)
            run_dir = root / "runs" / "auto-test"
            self.assertEqual([_teacher_output()], read_jsonl(run_dir / "teacher_responses.jsonl"))
            self.assertEqual([_rating()], read_jsonl(run_dir / "ratings.jsonl"))
            self.assertEqual(0, result)



    def test_cleanup_codex_threads_dry_run_lists_recorded_threads(self):
        from auto_evaluate import cli

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "demo"
            records = run_dir / "codex" / "records" / "judge"
            records.mkdir(parents=True)
            (records / "a.json").write_text(
                json.dumps(
                    {"stage": "judge", "task_id": "judge::q1::response-a", "thread_id": "thr-1"}
                ),
                encoding="utf-8",
            )
            (records / "b.json").write_text(
                json.dumps(
                    {"stage": "judge", "task_id": "judge::q1::response-b", "thread_id": "thr-2"}
                ),
                encoding="utf-8",
            )
            with patch("auto_evaluate.cli._root", return_value=root):
                code = cli.command_cleanup_codex_threads(
                    argparse.Namespace(run_id="demo", dry_run=True)
                )
            self.assertEqual(0, code)


if __name__ == "__main__":
    unittest.main()
