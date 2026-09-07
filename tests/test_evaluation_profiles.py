from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_evaluate.cli import _probe_binding_expectations, build_parser
from auto_evaluate.codex_automation import validate_teacher_output
from auto_evaluate.evaluation import load_profiles
from auto_evaluate.io_utils import read_json
from auto_evaluate.runner import (
    format_execution_error_for_log,
    retry_limit_for_error,
)


class EvaluationProfileTests(unittest.TestCase):
    def test_solver_sampling_is_low_variance_and_router_stays_deterministic(self):
        systems_config = read_json(ROOT / "configs" / "systems.json")
        self.assertEqual(0.2, systems_config["generation"]["temperature"])
        self.assertEqual(0.9, systems_config["generation"]["top_p"])

        adaptive_system = next(
            system
            for system in systems_config["systems"]
            if system["id"] == "tools-adaptive-rag"
        )
        self.assertEqual(
            0.0,
            adaptive_system["adaptive_rag"]["generation_overrides"]["temperature"],
        )

    def test_context_recovery_is_tool_free_single_pass_and_versioned(self):
        systems_config = read_json(ROOT / "configs" / "systems.json")
        recovery = systems_config["context_recovery"]
        self.assertTrue(recovery["enabled"])
        self.assertEqual(
            "context-reset-finalizer@0.2.0", recovery["policy_version"]
        )
        self.assertEqual(
            [
                "context_window_exceeded",
                "incomplete_response",
                "output_budget_exhausted",
            ],
            recovery["trigger_error_types"],
        )
        self.assertEqual("OPENWEBUI_MODEL_BASELINE", recovery["finalizer_model_env"])
        self.assertEqual(12000, recovery["max_partial_response_chars"])
        self.assertEqual(24000, recovery["max_prompt_chars"])
        self.assertEqual(0, recovery["generation"]["max_retries"])
        self.assertFalse(recovery["generation"]["enable_thinking"])

    def test_solver_prompt_preserves_agent_tool_entry_contract(self):
        systems_config = read_json(ROOT / "configs" / "systems.json")
        prompt = systems_config["shared_system_prompt"]
        self.assertIn("show the calculation path", prompt)
        self.assertIn("concise and complete natural-language answer", prompt)
        self.assertNotIn("decision or conclusion first", prompt)
        self.assertNotIn("Do not narrate internal planning", prompt)

        systems = {row["id"]: row for row in systems_config["systems"]}
        self.assertTrue(systems["tools"]["require_observable_tool_call"])
        self.assertTrue(systems["tools-rag"]["require_observable_tool_call"])
        self.assertNotIn("require_observable_tool_call", systems["baseline"])

    def test_all_9b_requests_disable_explicit_thinking(self):
        systems_config = read_json(ROOT / "configs" / "systems.json")
        self.assertIs(False, systems_config["generation"]["enable_thinking"])

        adaptive_system = next(
            system
            for system in systems_config["systems"]
            if system["id"] == "tools-adaptive-rag"
        )
        self.assertIs(
            False,
            adaptive_system["adaptive_rag"]["generation_overrides"][
                "enable_thinking"
            ],
        )

    def test_profiles_define_requested_candidate_matrices(self):
        profiles = load_profiles(ROOT)["profiles"]
        self.assertEqual(
            ["baseline", "tools", "tools-rag", "tools-adaptive-rag"],
            profiles["d1_d6"]["system_ids"],
        )
        self.assertEqual(
            ["baseline", "tools", "tools-rag", "tools-adaptive-rag"],
            profiles["d7"]["system_ids"],
        )
        self.assertEqual(
            "skip_rag",
            profiles["d1_d6"]["adaptive_rag_analysis"]["default_expected_route"],
        )
        self.assertIsNone(
            profiles["d7"]["adaptive_rag_analysis"]["default_expected_route"]
        )
        self.assertNotIn("skill_gate", profiles["d1_d6"])
        for profile_id in ("d1_d6", "d7"):
            self.assertEqual(
                [False, True],
                [teacher["tools_enabled"] for teacher in profiles[profile_id]["teachers"]],
            )
        self.assertEqual(
            {
                "solver_skill_side",
                "d1_d6",
                "d1_d6_repeatability",
                "d7",
                "d7_mock",
            },
            set(profiles),
        )
        self.assertEqual(
            ["baseline", "tools", "tools-rag"],
            profiles["d1_d6_repeatability"]["system_ids"],
        )
        self.assertEqual([], profiles["d1_d6_repeatability"]["teachers"])
        self.assertEqual(
            "skip_rag",
            profiles["d7_mock"]["adaptive_rag_analysis"]["default_expected_route"],
        )
        self.assertEqual(
            ["tools-rag-skill-control", "tools-rag-solver-v060", "tools-rag-solver-v089"],
            profiles["solver_skill_side"]["system_ids"],
        )

    def test_repeatability_selection_is_balanced_and_hash_determined(self):
        selection = read_json(ROOT / "configs" / "d1_d6_repeatability_26.json")
        selected = selection["case_ids"]
        self.assertEqual(26, len(selected))
        self.assertEqual(26, len(set(selected)))

        index = read_json(ROOT / "benchmarks" / "normalized" / "index.json")
        family_by_case = {row["case_id"]: row["task_family"] for row in index["cases"]}
        grouped = {}
        for case_id, family in family_by_case.items():
            key = hashlib.sha256(
                f"{selection['selection_id']}|{case_id}".encode()
            ).hexdigest()
            grouped.setdefault(family, []).append((key, case_id))
        expected = {
            case_id
            for rows in grouped.values()
            for _, case_id in sorted(rows)[: selection["cases_per_family"]]
        }
        self.assertEqual(13, len(grouped))
        self.assertEqual(expected, set(selected))
        counts = {}
        for case_id in selected:
            family = family_by_case[case_id]
            counts[family] = counts.get(family, 0) + 1
        self.assertEqual({2}, set(counts.values()))

    def test_solver_skill_side_uses_distinct_remote_skill_ids(self):
        systems = {
            row["id"]: row
            for row in read_json(ROOT / "configs" / "systems.json")["systems"]
        }
        self.assertEqual(
            "swro-watertap-v060", systems["tools-rag-solver-v060"]["remote_skill_id"]
        )
        self.assertEqual(
            "swro-watertap-v089", systems["tools-rag-solver-v089"]["remote_skill_id"]
        )
        self.assertEqual(
            {"temperature": 0.0}, systems["tools-rag-skill-control"]["generation_overrides"]
        )
        self.assertEqual(
            systems["tools-rag-skill-control"]["generation_overrides"],
            systems["tools-rag-solver-v060"]["generation_overrides"],
        )
        self.assertEqual(
            systems["tools-rag-skill-control"]["generation_overrides"],
            systems["tools-rag-solver-v089"]["generation_overrides"],
        )

    def test_tool_free_teacher_rejects_observable_tool_event(self):
        task = {
            "task_id": "teacher-general::q1",
            "case_id": "q1",
            "model": "gpt-5.6-teacher-general",
            "expected_output": {"system_id": "gpt-5.6-teacher-general"},
            "tool_policy": {"forbid_observable_calls": True},
        }
        output = {
            "task_id": task["task_id"],
            "case_id": "q1",
            "system_id": "gpt-5.6-teacher-general",
            "response_text": "answer",
            "trajectory": {
                "events": [
                    {"event_type": "tool_interaction", "tool_name": "watertap.solve"}
                ]
            },
        }
        self.assertIn(
            "tool-free teacher must not make observable tool calls",
            validate_teacher_output(task, output),
        )

    def test_stage_and_concurrency_defaults(self):
        args = build_parser().parse_args(["auto", "--run-id", "x"])
        self.assertEqual("all", args.stage)
        self.assertEqual(2, args.teacher_general_concurrency)
        self.assertEqual(1, args.teacher_tools_concurrency)
        self.assertEqual(4, args.judge_concurrency)
        self.assertEqual(2, args.system_concurrency)
        self.assertFalse(args.require_complete_systems)

    def test_router_eval_exposes_zero_solver_pilot(self):
        args = build_parser().parse_args(
            ["router-eval", "--benchmark-set", "d1_d6", "--pilot"]
        )
        self.assertTrue(args.pilot)
        self.assertEqual("configs/router_evaluation.json", args.config)

    def test_paper_adaptive_system_does_not_mount_solver_skill(self):
        systems = {
            row["id"]: row
            for row in read_json(ROOT / "configs" / "systems.json")["systems"]
        }
        adaptive = systems["tools-adaptive-rag"]
        self.assertIsNone(adaptive["skill_version"])
        self.assertEqual("policy_replay", adaptive["adaptive_rag"]["execution_mode"])
        self.assertEqual("tools", adaptive["adaptive_rag"]["no_rag_system_id"])
        self.assertEqual("tools-rag", adaptive["adaptive_rag"]["rag_system_id"])
        self.assertEqual("OPENWEBUI_MODEL_TOOLS", adaptive["adaptive_rag"]["no_rag_model_env"])
        self.assertEqual(
            "OPENWEBUI_MODEL_TOOLS_RAG", adaptive["adaptive_rag"]["rag_model_env"]
        )
        self.assertEqual(
            "swro-rag-router@0.1.2",
            adaptive["adaptive_rag"]["router_skill_version"],
        )

    def test_retry_policy_does_not_repeat_permanent_requests(self):
        self.assertEqual(0, retry_limit_for_error("authentication_failure", 2))
        self.assertEqual(0, retry_limit_for_error("invalid_request", 2))
        self.assertEqual(0, retry_limit_for_error("context_window_exceeded", 2))
        self.assertEqual(0, retry_limit_for_error("output_budget_exhausted", 2))
        self.assertEqual(0, retry_limit_for_error("required_tool_call_missing", 2))
        self.assertEqual(2, retry_limit_for_error("connection_failure", 2))

    def test_required_tool_error_log_explains_that_an_answer_was_returned(self):
        self.assertEqual(
            "required_tool_call_missing: answer returned without a successful "
            "observable tool call/result",
            format_execution_error_for_log("required_tool_call_missing"),
        )
        self.assertEqual(
            "connection_failure",
            format_execution_error_for_log("connection_failure"),
        )

    def test_probe_resolves_adaptive_system_to_physical_presets(self):
        config = {
            "systems": [
                {
                    "id": "tools",
                    "model_id": "tools-model",
                    "rag_enabled": False,
                },
                {
                    "id": "tools-rag",
                    "model_id": "tools-rag-model",
                    "rag_enabled": True,
                },
                {
                    "id": "adaptive",
                    "adaptive_rag": {
                        "router_model_id": "baseline-model",
                        "no_rag_model_id": "tools-model",
                        "rag_model_id": "tools-rag-model",
                    },
                },
            ]
        }
        expectations = _probe_binding_expectations(config)
        self.assertEqual(
            {"baseline-model", "tools-model", "tools-rag-model"},
            set(expectations),
        )
        self.assertFalse(expectations["tools-model"]["rag_enabled"])
        self.assertTrue(expectations["tools-rag-model"]["rag_enabled"])
        self.assertEqual(
            ["tools", "adaptive:skip_rag"],
            expectations["tools-model"]["aliases"],
        )

    def test_probe_uses_remote_skill_id_for_versioned_skill_clones(self):
        config = {
            "systems": [
                {
                    "id": "skill-v1",
                    "model_id": "preset-v1",
                    "rag_enabled": True,
                    "skill_version": "shared-skill@1.0.0",
                    "remote_skill_id": "shared-skill-v1",
                }
            ]
        }
        expectations = _probe_binding_expectations(config)
        self.assertEqual(
            "shared-skill-v1", expectations["preset-v1"]["expected_skill"]
        )

    def test_probe_allows_prompt_skill_and_control_to_share_physical_preset(self):
        config = {
            "systems": [
                {
                    "id": "c00-r1",
                    "model_id": "tools",
                    "rag_enabled": False,
                },
                {
                    "id": "c10-r1",
                    "model_id": "tools",
                    "rag_enabled": False,
                    "skill_version": "swro-parallel-compare@0.1.0",
                    "skill_delivery": "prompt",
                },
            ]
        }
        expectations = _probe_binding_expectations(config)
        self.assertEqual({"tools"}, set(expectations))
        self.assertEqual(["c00-r1", "c10-r1"], expectations["tools"]["aliases"])
        self.assertIsNone(expectations["tools"]["expected_skill"])


if __name__ == "__main__":
    unittest.main()
