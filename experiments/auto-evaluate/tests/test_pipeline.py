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
from auto_evaluate.openwebui import OpenWebUIAgentError, OpenWebUIClient, OpenWebUIError  # noqa: E402
from auto_evaluate.report import build_report  # noqa: E402
from auto_evaluate.runner import (  # noqa: E402
    _context_recovery_excerpt,
    execute_run,
    parse_rag_route,
    required_tool_call_error,
    response_completion_error,
    summarize_run_completeness,
)


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
    def test_context_recovery_excerpt_keeps_bounded_head_and_tail(self):
        text = "A" * 2000 + "B" * 2000
        excerpt, truncated = _context_recovery_excerpt(text, 1000)
        self.assertTrue(truncated)
        self.assertEqual(1000, len(excerpt))
        self.assertTrue(excerpt.startswith("A"))
        self.assertTrue(excerpt.endswith("B"))
        self.assertIn("middle of partial execution omitted", excerpt)

    def test_parse_rag_route_validates_router_contract(self):
        route = parse_rag_route(
            '```json\n{"action":"use_rag","reason_code":"MISSING_DOMAIN_KNOWLEDGE",'
            '"confidence":0.8,"retrieval_need":"carbonate scaling interpretation"}\n```'
        )
        self.assertEqual("use_rag", route["action"])
        self.assertEqual(0.8, route["confidence"])
        with self.assertRaisesRegex(ValueError, "retrieval_need=null"):
            parse_rag_route(
                '{"action":"skip_rag","reason_code":"FULLY_SPECIFIED_NUMERIC_TASK",'
                '"confidence":0.9,"retrieval_need":"unnecessary"}'
            )

    def test_prepare_benchmarks_uses_active_set_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source_config = temp / "configs" / "benchmarks_single.json"
            registry_path = temp / "configs" / "benchmark_sets.json"
            workbook = (ROOT / "benchmarks" / "Datasets Harness" / "D1" / "D1_1a_EN.xlsx").resolve()

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
                    system_concurrency=1,
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

    def test_execute_run_recovers_context_error_with_tool_free_finalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "runs" / "recovered"
            benchmark = {
                "case_id": "q-recovery",
                "question_prompt": "Find the observed safe recovery boundary.",
                "source": {"sha256": "case-sha"},
            }
            recovery = {
                "enabled": True,
                "policy_version": "context-reset-finalizer@0.2.0",
                "trigger_error_types": [
                    "context_window_exceeded",
                    "incomplete_response",
                    "output_budget_exhausted",
                ],
                "finalizer_model_id": "baseline-model",
                "require_partial_response": True,
                "max_partial_response_chars": 12000,
                "max_prompt_chars": 24000,
                "system_prompt": "Use observed evidence only and finalize without tools.",
                "generation": {
                    "stream": True,
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "max_tokens": 256,
                    "enable_thinking": False,
                    "max_retries": 0,
                },
            }
            config = {
                "shared_system_prompt": "Return a concise and complete natural-language answer.",
                "generation": {
                    "stream": True,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 256,
                    "max_retries": 0,
                },
                "context_recovery": recovery,
                "systems": [
                    {
                        "id": "tools",
                        "display_name": "Tools",
                        "model_id": "tools-model",
                        "tools_enabled": True,
                        "rag_enabled": False,
                    }
                ],
            }

            class _Client:
                def __init__(self):
                    self.calls = []

                def chat_stream(self, *, model, messages, generation):
                    self.calls.append((model, messages, generation))
                    if model == "tools-model":
                        raise OpenWebUIAgentError(
                            "maximum context length is 16384 tokens",
                            response_text=(
                                "Observed R=0.496: SI=-0.0006. "
                                + ("intermediate simulator output " * 1500)
                                + "The next tool call failed before R=0.497 returned."
                            ),
                            raw_response={"usage": None, "finish_reason": "stop"},
                            latency_ms=5,
                        )
                    return type(
                        "Result",
                        (),
                        {
                            "content": "The observed safe point is R=0.496; R=0.497 remains unverified.",
                            "raw": {"usage": {"total_tokens": 80}, "finish_reason": "stop"},
                            "latency_ms": 7,
                        },
                    )()

            client = _Client()
            with (
                patch("auto_evaluate.runner.load_systems", return_value=config),
                patch("auto_evaluate.runner.make_client", return_value=client),
                patch(
                    "auto_evaluate.runner.iter_benchmarks",
                    side_effect=lambda _: iter([benchmark]),
                ),
            ):
                counts = execute_run(
                    benchmarks_dir=temp / "benchmarks",
                    systems_path=temp / "configs" / "systems.json",
                    run_dir=run_dir,
                    system_concurrency=1,
                )

            self.assertEqual({"success": 1, "error": 0, "skipped": 0}, counts)
            self.assertEqual(["tools-model", "baseline-model"], [row[0] for row in client.calls])
            finalizer_call = client.calls[1]
            self.assertIn("ORIGINAL QUESTION", finalizer_call[1][1]["content"])
            self.assertIn("Observed R=0.496", finalizer_call[1][1]["content"])
            self.assertFalse(finalizer_call[2]["enable_thinking"])

            record = read_json(run_dir / "responses" / "q-recovery__tools.json")
            self.assertEqual("success", record["status"])
            self.assertEqual("error", record["native_status"])
            self.assertEqual("context_window_exceeded", record["native_error_type"])
            self.assertEqual("context_reset_finalizer", record["completion_mode"])
            self.assertEqual("success", record["recovery"]["status"])
            self.assertTrue(record["recovery"]["partial_response_truncated"])
            self.assertLessEqual(record["recovery"]["included_partial_response_chars"], 12000)
            self.assertLessEqual(record["recovery"]["prompt_chars"], 24000)
            self.assertEqual(12, record["latency_ms"])
            self.assertIn("R=0.497 remains unverified", record["response_text"])
            self.assertIn("next tool call failed", record["partial_response_text"])

    def test_execute_run_recovers_tool_free_incomplete_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "runs" / "baseline-error"
            benchmark = {
                "case_id": "q-baseline",
                "question_prompt": "question",
                "source": {"sha256": "case-sha"},
            }
            config = {
                "shared_system_prompt": "system",
                "generation": {"stream": True, "max_retries": 0},
                "context_recovery": {
                    "enabled": True,
                    "policy_version": "context-reset-finalizer@0.2.0",
                    "trigger_error_types": [
                        "context_window_exceeded",
                        "incomplete_response",
                        "output_budget_exhausted",
                    ],
                    "finalizer_model_id": "baseline-model",
                    "require_partial_response": True,
                    "max_partial_response_chars": 12000,
                    "max_prompt_chars": 24000,
                    "system_prompt": "finalize",
                    "generation": {"stream": True},
                },
                "systems": [
                    {
                        "id": "baseline",
                        "display_name": "Baseline",
                        "model_id": "baseline-model",
                        "tools_enabled": False,
                    }
                ],
            }

            class _Client:
                def __init__(self):
                    self.calls = 0

                def chat_stream(self, *, model, messages, generation):
                    self.calls += 1
                    if self.calls == 1:
                        raise OpenWebUIAgentError(
                            "Assistant response incomplete: finish_reason='length'",
                            response_text="partial baseline calculation",
                            raw_response={"usage": None, "finish_reason": "length"},
                            latency_ms=3,
                        )
                    return type(
                        "Result",
                        (),
                        {
                            "content": "The final baseline decision is feasible.",
                            "raw": {"usage": None, "finish_reason": "stop"},
                            "latency_ms": 2,
                        },
                    )()

            client = _Client()
            with (
                patch("auto_evaluate.runner.load_systems", return_value=config),
                patch("auto_evaluate.runner.make_client", return_value=client),
                patch(
                    "auto_evaluate.runner.iter_benchmarks",
                    side_effect=lambda _: iter([benchmark]),
                ),
            ):
                counts = execute_run(
                    benchmarks_dir=temp / "benchmarks",
                    systems_path=temp / "configs" / "systems.json",
                    run_dir=run_dir,
                )

            self.assertEqual({"success": 1, "error": 0, "skipped": 0}, counts)
            self.assertEqual(2, client.calls)
            record = read_json(run_dir / "responses" / "q-baseline__baseline.json")
            self.assertEqual("context_reset_finalizer", record["completion_mode"])
            self.assertEqual("incomplete_response", record["native_error_type"])
            self.assertEqual("success", record["recovery"]["status"])
            self.assertEqual("The final baseline decision is feasible.", record["response_text"])

    def test_execute_run_preserves_native_error_when_finalizer_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "runs" / "recovery-error"
            benchmark = {
                "case_id": "q-recovery-error",
                "question_prompt": "question",
                "source": {"sha256": "case-sha"},
            }
            config = {
                "shared_system_prompt": "system",
                "generation": {"stream": True, "max_retries": 0},
                "context_recovery": {
                    "enabled": True,
                    "policy_version": "context-reset-finalizer@0.2.0",
                    "trigger_error_types": [
                        "context_window_exceeded",
                        "incomplete_response",
                        "output_budget_exhausted",
                    ],
                    "finalizer_model_id": "baseline-model",
                    "require_partial_response": True,
                    "max_partial_response_chars": 12000,
                    "max_prompt_chars": 24000,
                    "system_prompt": "finalize",
                    "generation": {"stream": True},
                },
                "systems": [
                    {
                        "id": "tools",
                        "display_name": "Tools",
                        "model_id": "tools-model",
                        "tools_enabled": True,
                    }
                ],
            }

            class _Client:
                def chat_stream(self, *, model, messages, generation):
                    if model == "tools-model":
                        raise OpenWebUIAgentError(
                            "maximum context length is 16384 tokens",
                            response_text="partial tool evidence",
                            raw_response={"usage": None},
                            latency_ms=3,
                        )
                    raise OpenWebUIError("OpenWebUI connection failed: offline")

            with (
                patch("auto_evaluate.runner.load_systems", return_value=config),
                patch("auto_evaluate.runner.make_client", return_value=_Client()),
                patch("auto_evaluate.runner.iter_benchmarks", return_value=iter([benchmark])),
            ):
                counts = execute_run(
                    benchmarks_dir=temp / "benchmarks",
                    systems_path=temp / "configs" / "systems.json",
                    run_dir=run_dir,
                )

            self.assertEqual({"success": 0, "error": 1, "skipped": 0}, counts)
            record = read_json(run_dir / "responses" / "q-recovery-error__tools.json")
            self.assertEqual("context_window_exceeded", record["error_type"])
            self.assertEqual("error", record["recovery"]["status"])
            self.assertEqual("connection_failure", record["recovery"]["error_type"])

    def test_execute_run_routes_to_rag_preset_and_records_end_to_end_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "runs" / "adaptive"
            benchmark = {
                "case_id": "q-route",
                "question_prompt": "Interpret an unspecified mineral scaling mechanism.",
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
                        "id": "adaptive",
                        "display_name": "Adaptive",
                        "model_id": "no-rag-model",
                        "tools_enabled": True,
                        "rag_enabled": True,
                        "rag_policy": "two_stage_skill_router",
                        "adaptive_rag": {
                            "router_model_id": "router-model",
                            "no_rag_model_id": "no-rag-model",
                            "rag_model_id": "rag-model",
                            "router_prompt": "route only",
                            "router_skill_version": "swro-rag-router@0.1.0",
                            "fallback_action": "skip_rag",
                        },
                    }
                ],
            }

            class _Client:
                def __init__(self):
                    self.calls = []

                def chat_stream(self, *, model, messages, generation):
                    if model == "router-model":
                        self.calls.append(("router", model, generation["stream"]))
                        content = json.dumps(
                            {
                                "action": "use_rag",
                                "reason_code": "MISSING_DOMAIN_KNOWLEDGE",
                                "confidence": 0.75,
                                "retrieval_need": "mineral scaling interpretation",
                            }
                        )
                        return type(
                            "Result",
                            (),
                            {"content": content, "raw": {"usage": None}, "latency_ms": 3},
                        )()
                    self.calls.append(("solver", model, generation["stream"]))
                    return type(
                        "Result", (), {"content": "complete answer", "raw": {"usage": None}, "latency_ms": 7}
                    )()

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

            self.assertEqual({"success": 1, "error": 0, "skipped": 0}, counts)
            self.assertEqual(
                [("router", "router-model", True), ("solver", "rag-model", True)],
                client.calls,
            )
            record = read_json(run_dir / "responses" / "q-route__adaptive.json")
            self.assertEqual("use_rag", record["routing"]["action"])
            self.assertTrue(record["rag_enabled"])
            self.assertEqual("rag-model", record["model_id"])
            self.assertEqual(10, record["latency_ms"])
            self.assertEqual(7, record["solver_latency_ms"])

    def test_execute_run_policy_replay_selects_existing_rag_arm_without_second_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "runs" / "policy-replay"
            benchmark = {
                "case_id": "q-replay",
                "question_prompt": "Apply a missing external operating rule.",
                "source": {"sha256": "case-sha"},
            }
            config = {
                "shared_system_prompt": "system",
                "generation": {
                    "stream": True,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 64,
                    "max_retries": 0,
                },
                "systems": [
                    {
                        "id": "tools",
                        "display_name": "Tools",
                        "model_id": "tools-model",
                        "tools_enabled": True,
                        "rag_enabled": False,
                    },
                    {
                        "id": "tools-rag",
                        "display_name": "Tools + RAG",
                        "model_id": "rag-model",
                        "tools_enabled": True,
                        "rag_enabled": True,
                    },
                    {
                        "id": "adaptive",
                        "display_name": "Adaptive",
                        "model_id": "tools-model",
                        "tools_enabled": True,
                        "rag_enabled": True,
                        "rag_policy": "two_stage_skill_router",
                        "adaptive_rag": {
                            "execution_mode": "policy_replay",
                            "no_rag_system_id": "tools",
                            "rag_system_id": "tools-rag",
                            "router_model_id": "router-model",
                            "no_rag_model_id": "tools-model",
                            "rag_model_id": "rag-model",
                            "router_prompt": "route only",
                            "router_skill_version": "swro-rag-router@0.1.2",
                            "fallback_action": "skip_rag",
                        },
                    },
                ],
            }

            class _Client:
                def __init__(self):
                    self.models = []

                def chat_stream(self, *, model, messages, generation):
                    self.models.append(model)
                    if model == "router-model":
                        content = json.dumps(
                            {
                                "action": "use_rag",
                                "reason_code": "MISSING_DOMAIN_KNOWLEDGE",
                                "confidence": 0.9,
                                "retrieval_need": "external operating rule",
                            }
                        )
                        return type(
                            "Result",
                            (),
                            {"content": content, "raw": {"usage": None}, "latency_ms": 2},
                        )()
                    return type(
                        "Result",
                        (),
                        {
                            "content": "rag answer" if model == "rag-model" else "tools answer",
                            "raw": {"usage": None, "finish_reason": "stop"},
                            "latency_ms": 5,
                        },
                    )()

            client = _Client()
            with (
                patch("auto_evaluate.runner.load_systems", return_value=config),
                patch("auto_evaluate.runner.make_client", return_value=client),
                patch(
                    "auto_evaluate.runner.iter_benchmarks",
                    side_effect=lambda _: iter([benchmark]),
                ),
            ):
                counts = execute_run(
                    benchmarks_dir=temp / "benchmarks",
                    systems_path=temp / "configs" / "systems.json",
                    run_dir=run_dir,
                    system_concurrency=2,
                )
                completeness = summarize_run_completeness(
                    benchmarks_dir=temp / "benchmarks",
                    run_dir=run_dir,
                    system_ids=["tools", "tools-rag", "adaptive"],
                )

            self.assertEqual({"success": 3, "error": 0, "skipped": 0}, counts)
            self.assertEqual(2, completeness["native_success"])
            self.assertEqual(1, completeness["policy_replay_success"])
            self.assertEqual(1, client.models.count("tools-model"))
            self.assertEqual(1, client.models.count("rag-model"))
            self.assertEqual(1, client.models.count("router-model"))
            record = read_json(run_dir / "responses" / "q-replay__adaptive.json")
            self.assertEqual("policy_replay", record["completion_mode"])
            self.assertEqual("not_executed", record["native_status"])
            self.assertEqual("tools-rag", record["selected_arm_system_id"])
            self.assertEqual("rag answer", record["response_text"])
            self.assertTrue(record["rag_enabled"])
            self.assertEqual(7, record["latency_ms"])

    def test_prepare_judge_skips_duplicate_policy_replay_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            benchmarks_dir = temp / "benchmarks"
            run_dir = temp / "runs" / "judge-replay"
            imported = import_all(ROOT / "configs" / "benchmarks.json", benchmarks_dir, ROOT)
            case_id = imported[0]["case_id"]
            write_json(
                run_dir / "responses" / f"{case_id}__tools.json",
                {
                    "status": "success",
                    "case_id": case_id,
                    "system_id": "tools",
                    "display_name": "Tools",
                    "response_text": "physical answer",
                },
            )
            write_json(
                run_dir / "responses" / f"{case_id}__adaptive.json",
                {
                    "status": "success",
                    "case_id": case_id,
                    "system_id": "tools-adaptive-rag",
                    "display_name": "Adaptive",
                    "completion_mode": "policy_replay",
                    "selected_arm_system_id": "tools",
                    "response_text": "physical answer",
                },
            )
            write_jsonl(run_dir / "teacher_responses.jsonl", [])

            tasks = [
                task
                for task in read_jsonl(prepare_judge_tasks(benchmarks_dir, run_dir, seed=1))
                if task["case_id"] == case_id
            ]
            self.assertEqual(1, len(tasks))
            mapping = read_json(run_dir / "judge_mapping.json")["mapping"]
            mapped = [row for row in mapping if row["case_id"] == case_id]
            self.assertEqual(["tools"], [row["system_id"] for row in mapped])

    def test_extract_score_points_parses_structured_trailer(self):
        response = """Final answer text.
[SCORE_POINTS_BEGIN]
{"task_type":"design_pressure","decision_variables":["feed_pressure_bar"],"fixed_inputs":{"temperature_c":32},"tool_calls":[],"constraint_checks":[],"final_answer":{"recommendation":"51 bar"}}
[SCORE_POINTS_END]
"""
        parsed = extract_score_points(response)
        self.assertEqual("ok", parsed["status"])
        self.assertEqual("design_pressure", parsed["data"]["task_type"])

    def test_extracts_observable_tool_trajectory_from_visible_transcript(self):
        from auto_evaluate.trajectory import extract_observable_trajectory

        response = '''Plan
`🔧 ro-chem-simulate_ro({"membrane_area_m2": 400})`
`↳ {"performance": {"salt_rejection_pct": 98.8}}`
Final answer'''
        trace = extract_observable_trajectory(response, tools_enabled=True, rag_enabled=True)
        self.assertEqual("visible_response_transcript", trace["source"])
        self.assertEqual(1, trace["summary"]["tool_interactions"])
        self.assertEqual("ro-chem-simulate_ro", trace["events"][0]["tool_name"])
        self.assertEqual(400, trace["events"][0]["arguments"]["membrane_area_m2"])

    def test_extracts_parallel_batched_tool_transcript(self):
        from auto_evaluate.trajectory import extract_observable_trajectory

        response = '''Plan
`🔧 ro-chem-simulate_ro({"case":"A","area":400})`
`🔧 ro-chem-simulate_ro({"case":"A","area":400})`
`🔧 ro-chem-simulate_ro({"case":"B","area":400})`
`↳ {"case":"A","product":5.2}`
`↳ {"case":"B","product":4.8}`
Final answer'''
        trace = extract_observable_trajectory(response, tools_enabled=True)

        self.assertEqual(2, trace["summary"]["tool_interactions"])
        self.assertEqual("A", trace["events"][0]["arguments"]["case"])
        self.assertEqual("B", trace["events"][1]["arguments"]["case"])
        self.assertEqual(4.8, trace["events"][1]["observation"]["product"])

    def test_completion_contract_rejects_missing_or_malformed_trailer(self):
        prompt = "Append [SCORE_POINTS_BEGIN] JSON [SCORE_POINTS_END]."
        self.assertIn(
            "trailer is missing",
            response_completion_error("analysis only", {"finish_reason": "stop"}, prompt),
        )
        self.assertIn(
            "not valid JSON",
            response_completion_error(
                "answer [SCORE_POINTS_BEGIN]{bad}[SCORE_POINTS_END]",
                {"finish_reason": "stop"},
                prompt,
            ),
        )
        complete = {
            "task_type": "boundary",
            "decision_variables": [],
            "fixed_inputs": {},
            "tool_calls": [],
            "constraint_checks": [],
            "final_answer": {},
        }
        text = (
            "answer [SCORE_POINTS_BEGIN]"
            + json.dumps(complete)
            + "[SCORE_POINTS_END]"
        )
        self.assertIsNone(response_completion_error(text, {"finish_reason": "stop"}, prompt))

    def test_completion_contract_rejects_length_finish(self):
        self.assertIn(
            "finish_reason='length'",
            response_completion_error("partial", {"finish_reason": "length"}, "system"),
        )

    def test_completion_contract_accepts_natural_answer_when_trailer_not_requested(self):
        prompt = "Return a concise and complete natural-language answer."
        self.assertIsNone(
            response_completion_error(
                "The selected design passes all stated constraints.",
                {"finish_reason": "stop"},
                prompt,
            )
        )

    def test_required_tool_contract_rejects_final_answer_only(self):
        from auto_evaluate.trajectory import extract_observable_trajectory

        no_tools = extract_observable_trajectory(
            "A simulated-looking answer with no observed call.",
            tools_enabled=True,
        )
        self.assertIn(
            "observable tool call is missing",
            required_tool_call_error(no_tools, required=True),
        )
        self.assertIsNone(required_tool_call_error(no_tools, required=False))

        with_tool = extract_observable_trajectory(
            '`🔧 ro-chem-simulate_ro({"area":400})`\n'
            '`↳ {"flux":15.7}`\nFinal answer',
            tools_enabled=True,
        )
        self.assertIsNone(required_tool_call_error(with_tool, required=True))

    @patch("urllib.request.urlopen")
    def test_openwebui_preserves_native_tool_events(self, urlopen):
        urlopen.return_value = _FakeStreamResponse(
            b'data: {"choices":[{"delta":{"tool_calls":[{"id":"c1","function":{"name":"simulate_ro"}}]},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        client = OpenWebUIClient("https://example.test", "secret", timeout_seconds=5)
        result = client.chat_stream(
            model="agent",
            messages=[{"role": "user", "content": "question"}],
            generation={"temperature": 0, "top_p": 1},
        )
        self.assertEqual("tool_calls", result.raw["trajectory_events"][0]["field"])

    @patch("urllib.request.urlopen")
    def test_openwebui_agent_error_preserves_partial_response(self, urlopen):
        urlopen.return_value = _FakeStreamResponse(
            b'data: {"choices":[{"delta":{"content":"partial tool result\\n"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"**[agent error]** maximum context length"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        client = OpenWebUIClient("https://example.test", "secret", timeout_seconds=5)
        with self.assertRaises(OpenWebUIAgentError) as caught:
            client.chat_stream(
                model="agent",
                messages=[{"role": "user", "content": "question"}],
                generation={"temperature": 0.7, "top_p": 0.8},
            )
        self.assertIn("partial tool result", caught.exception.response_text)
        self.assertEqual(2, caught.exception.raw_response["event_count"])

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

    def test_probe_chat_has_no_client_token_cap_by_default(self):
        args = cli.build_parser().parse_args(["probe-chat"])
        self.assertIsNone(args.max_tokens)
        self.assertFalse(args.no_thinking)

        args = cli.build_parser().parse_args(["probe-chat", "--no-thinking"])
        self.assertTrue(args.no_thinking)

    @patch("urllib.request.urlopen")
    def test_openwebui_stream_omits_max_tokens_when_unspecified(self, urlopen):
        urlopen.return_value = _FakeStreamResponse(
            b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        client = OpenWebUIClient("https://example.test", "secret", timeout_seconds=5)
        client.chat_stream(
            model="agent",
            messages=[{"role": "user", "content": "question"}],
            generation={"temperature": 0.7, "top_p": 0.8, "enable_thinking": False},
        )
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(
            {"enable_thinking": False}, payload["chat_template_kwargs"]
        )

    @patch("urllib.request.urlopen")
    def test_openwebui_empty_stream_reports_reasoning_and_finish_reason(self, urlopen):
        urlopen.return_value = _FakeStreamResponse(
            b'data: {"choices":[{"delta":{"reasoning":"work"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        client = OpenWebUIClient("https://example.test", "secret", timeout_seconds=5)
        with self.assertRaisesRegex(
            OpenWebUIError,
            r"finish_reason='length'.*reasoning_events=1.*reasoning_chars=4",
        ):
            client.chat_stream(
                model="agent",
                messages=[{"role": "user", "content": "question"}],
                generation={"temperature": 0.7, "top_p": 0.8},
            )

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

    def test_prepare_judge_includes_failed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            benchmarks_dir = temp / "benchmarks"
            run_dir = temp / "runs" / "failed"
            imported = import_all(ROOT / "configs" / "benchmarks.json", benchmarks_dir, ROOT)
            benchmark = imported[0]
            write_json(
                run_dir / "responses" / "failed.json",
                {
                    "status": "error",
                    "error_type": "context_window_exceeded",
                    "error": "maximum context length",
                    "case_id": benchmark["case_id"],
                    "system_id": "environment",
                    "display_name": "Environment",
                    "response_text": "partial calculation without final answer",
                    "trajectory": {
                        "source": "visible_response_transcript",
                        "events": [],
                        "summary": {},
                    },
                },
            )
            write_jsonl(run_dir / "teacher_responses.jsonl", [])
            tasks = read_jsonl(prepare_judge_tasks(benchmarks_dir, run_dir, seed=1))
            failed = [task for task in tasks if task["case_id"] == benchmark["case_id"]]
            self.assertEqual(1, len(failed))
            self.assertEqual("error", failed[0]["candidate_execution_status"])
            self.assertEqual(
                "context_window_exceeded", failed[0]["candidate_execution_error_type"]
            )

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
                    "tools_enabled": False,
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
                "tool_efficiency_score": 100,
                "tool_efficiency_dimensions": [
                    {
                        "dimension_id": dimension["dimension_id"],
                        "score": dimension["max_score"],
                        "max_score": dimension["max_score"],
                        "evidence": "efficient tool use",
                        "diagnosis": "complete",
                    }
                    for dimension in task["tool_efficiency_rubric"]["dimensions"]
                ],
                "tool_efficiency_overall_diagnosis": "efficient and sufficient",
                "trajectory_analysis": {
                    "trajectory_source": task["observable_trajectory"]["source"],
                    "path_classification": "insufficient_trace",
                    "summary": "only the final answer is observable",
                    "first_error_event_id": None,
                    "recovery_attempted": False,
                    "recovery_succeeded": None,
                    "event_assessments": [
                        {
                            "event_id": task["observable_trajectory"]["events"][-1]["event_id"],
                            "verdict": "correct",
                            "failure_codes": [],
                            "primary_failure_code": None,
                            "evidence": "final answer",
                            "diagnosis": "complete answer",
                            "affected_rubric_steps": [step["step_id"] for step in task["rubric"]["steps"]],
                            "attributed_task_loss": 0,
                        }
                    ],
                },
                "causal_analysis": {
                    "first_error_step_id": None,
                    "root_cause": "no observed failure",
                    "error_propagation": [],
                    "downstream_affected_steps": [],
                    "minimal_fix": "none required",
                    "counterfactual_outcome": "the outcome remains correct",
                    "evidence_strength": "direct",
                },
                "research_tags": ["complete"],
                "skill_improvement_suggestions": [],
            }
            write_jsonl(run_dir / "ratings.jsonl", [rating])
            self.assertEqual([], validate_ratings(run_dir))
            report = build_report(run_dir)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Agent", text)
            self.assertIn("100.0", text)
            self.assertIn("报告主线与阅读顺序", text)
            self.assertIn('class="benchmark-card"', text)
            self.assertIn("逐题 Rubric、Loss 与轨迹证据", text)
            self.assertIn("评分失分归因（不是训练 loss）", text)
            self.assertIn("Task loss", text)
            self.assertIn("任务质量 Rubric", text)
            self.assertIn("Tool 效率 Rubric", text)
            self.assertIn("轨迹与替代路径", text)
            self.assertIn("有效替代路径", text)
            self.assertIn("评分证据与诊断", text)
            self.assertIn("中文说明", text)
            self.assertIn("Benchmark 诊断", text)
            self.assertIn("轨迹来源", text)
            self.assertIn("不适用：该条件没有工具权限", text)
            self.assertIn("失分 0.0", text)
            first_step_label = benchmark["rubric"]["steps"][0]["step_label"]
            first_efficiency_label = benchmark["tool_efficiency_rubric"]["dimensions"][0]["dimension_label"]
            self.assertNotIn(f'S1<small>{first_step_label}', text)
            self.assertNotIn(f'E1<small>{first_efficiency_label}', text)
            self.assertIn("端到端均分", text)
            self.assertIn("成功回答质量均分", text)
            self.assertIn("完成率 1/1", text)
            self.assertIn("主要待测系统", text)
            self.assertIn("多维科研图形", text)
            self.assertIn("report schema 5.0", text)
            self.assertTrue((run_dir / "figures" / "main-scores.svg").exists())
            self.assertTrue((run_dir / "figures" / "quality-efficiency.svg").exists())

    def test_execute_run_supports_independent_parallel_requests(self):
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            run_dir = temp / "run"
            benchmark = {
                "case_id": "q-parallel",
                "question_prompt": "question",
                "source": {"sha256": "abc"},
            }
            config = {
                "generation": {"stream": True, "max_retries": 0},
                "shared_system_prompt": "system",
                "systems": [
                    {"id": "a", "display_name": "A", "model_id": "model-a"},
                    {"id": "b", "display_name": "B", "model_id": "model-b"},
                ],
            }
            barrier = threading.Barrier(2)

            class _Client:
                def chat_stream(self, *, model, messages, generation):
                    barrier.wait(timeout=3)
                    return type(
                        "Result",
                        (),
                        {"content": model, "raw": {"usage": None}, "latency_ms": 1},
                    )()

            with (
                patch("auto_evaluate.runner.load_systems", return_value=config),
                patch("auto_evaluate.runner.make_client", return_value=_Client()),
                patch("auto_evaluate.runner.iter_benchmarks", return_value=iter([benchmark])),
            ):
                counts = execute_run(
                    benchmarks_dir=temp / "benchmarks",
                    systems_path=temp / "systems.json",
                    run_dir=run_dir,
                    system_concurrency=2,
                )
            self.assertEqual({"success": 2, "error": 0, "skipped": 0}, counts)


if __name__ == "__main__":
    unittest.main()
