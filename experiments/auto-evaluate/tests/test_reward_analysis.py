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
from auto_evaluate.cli import build_parser  # noqa: E402
from auto_evaluate.reward_analysis import build_reward_analysis, build_router_update_plan


class RewardAnalysisTests(unittest.TestCase):
    def test_paired_rewards_route_regret_and_router_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "reward-run"
            run_dir.mkdir(parents=True)
            systems = ["tools", "tools-rag", "adaptive"]
            write_json(
                run_dir / "evaluation_profile.json",
                {
                    "profile_id": "research",
                    "system_ids": systems,
                    "comparisons": [
                        {
                            "id": "rag_gain",
                            "baseline_system": "tools",
                            "candidate_system": "tools-rag",
                        }
                    ],
                    "adaptive_rag_analysis": {
                        "no_rag_system": "tools",
                        "always_rag_system": "tools-rag",
                        "adaptive_system": "adaptive",
                        "minimum_gain_for_rag": 0,
                        "use_benchmark_expected_route": True,
                        "default_expected_route": "skip_rag",
                        "default_rag_need": "R0",
                    },
                },
            )
            write_json(
                run_dir / "manifest.json",
                {
                    "benchmark_cases": ["q1"],
                    "systems": [
                        {"id": "tools"},
                        {"id": "tools-rag"},
                        {"id": "adaptive"},
                    ],
                },
            )
            write_json(
                run_dir / "benchmarks" / "q1.json",
                {
                    "case_id": "q1",
                    "benchmark_view": {"rag_need": "R2", "expected_route": "use_rag"},
                    "rubric": {
                        "steps": [
                            {"step_id": 1, "step_label": "Plan"},
                            {"step_id": 2, "step_label": "Search"},
                        ]
                    },
                },
            )
            mapping = []
            ratings = []
            scores = {"tools": 50, "tools-rag": 80, "adaptive": 70}
            for system_id, score in scores.items():
                task_id = f"{system_id}::q1"
                mapping.append({"task_id": task_id, "case_id": "q1", "system_id": system_id})
                ratings.append(
                    {
                        "task_id": task_id,
                        "case_id": "q1",
                        "total_score": score,
                        "tool_efficiency_score": 60,
                        "steps": [
                            {
                                "step_id": 1,
                                "score": score / 2,
                                "max_score": 50,
                                "failure_codes": ["SEARCH_STRATEGY"] if system_id != "tools-rag" else [],
                            },
                            {
                                "step_id": 2,
                                "score": score / 2,
                                "max_score": 50,
                                "failure_codes": ["SEARCH_STRATEGY"],
                            },
                        ],
                        "skill_improvement_suggestions": ["Reduce redundant search"],
                    }
                )
                response = {
                    "case_id": "q1",
                    "system_id": system_id,
                    "status": "success",
                    "trajectory": {"summary": {"tool_interactions": 4}},
                    "latency_ms": 100,
                    "rag_enabled": system_id == "tools-rag",
                }
                if system_id == "adaptive":
                    response["routing"] = {
                        "action": "skip_rag",
                        "reason_code": "FULLY_SPECIFIED_NUMERIC_TASK",
                        "confidence": 0.8,
                        "status": "success",
                        "router_skill_version": "swro-rag-router@0.1.1",
                    }
                write_json(run_dir / "responses" / f"q1__{system_id}.json", response)
            write_json(run_dir / "judge_mapping.json", {"mapping": mapping})
            write_jsonl(run_dir / "ratings.jsonl", ratings)

            analysis = build_reward_analysis(run_dir)
            self.assertEqual(30.0, analysis["comparisons"]["rag_gain"]["mean_total_gain"])
            self.assertEqual("use_rag", analysis["adaptive_rag"]["cases"][0]["optimal_action"])
            self.assertFalse(analysis["adaptive_rag"]["cases"][0]["routing_correct"])
            self.assertFalse(analysis["adaptive_rag"]["cases"][0]["policy_routing_correct"])
            self.assertEqual("benchmark_expected_route", analysis["adaptive_rag"]["routing_accuracy_basis"])
            # Router selected the no-RAG physical arm (score 50), so policy
            # regret is 80 - 50.  The independently sampled adaptive score 70
            # is retained only as an execution-stability diagnostic.
            self.assertEqual(30.0, analysis["adaptive_rag"]["mean_routing_regret"])
            route_row = analysis["adaptive_rag"]["cases"][0]
            self.assertEqual(50.0, route_row["policy_replay_score"])
            self.assertEqual(70.0, route_row["independent_adaptive_score"])
            self.assertEqual(20.0, route_row["independent_execution_gap"])
            self.assertEqual(
                1,
                analysis["adaptive_rag"]["policy_routing_classification"]
                ["confusion_matrix"]["true_use_pred_skip"],
            )
            self.assertEqual(30.0, analysis["adaptive_rag"]["by_rag_need"]["R2"]["mean_rag_gain"])
            self.assertEqual([30.0, 30.0], analysis["comparisons"]["rag_gain"]["mean_total_gain_ci95"])
            self.assertEqual(100.0, analysis["systems"]["tools"]["mean_latency_ms"])
            self.assertEqual(1.0, analysis["systems"]["tools-rag"]["rag_activation_rate"])

            # Per-case labels take precedence over profile defaults.
            self.assertEqual("use_rag", analysis["adaptive_rag"]["cases"][0]["expected_action"])
            self.assertEqual("R2", analysis["adaptive_rag"]["cases"][0]["rag_need"])

            plan = build_router_update_plan(run_dir, analysis)
            self.assertNotIn("solver_skill_targets", plan)
            self.assertEqual(1, plan["router_skill_target"]["n_misroutes"])
            self.assertEqual("swro-rag-router@0.1.1", plan["router_skill_target"]["router_skill_version"])

    def test_policy_replay_uses_rag_arm_score_without_duplicate_adaptive_rating(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "replay-r2"
            run_dir.mkdir(parents=True)
            write_json(
                run_dir / "evaluation_profile.json",
                {
                    "profile_id": "d7",
                    "system_ids": ["tools", "tools-rag", "tools-adaptive-rag"],
                    "comparisons": [],
                    "adaptive_rag_analysis": {
                        "no_rag_system": "tools",
                        "always_rag_system": "tools-rag",
                        "adaptive_system": "tools-adaptive-rag",
                        "minimum_gain_for_rag": 0,
                        "use_benchmark_expected_route": True,
                        "default_expected_route": None,
                        "default_rag_need": None,
                    },
                },
            )
            write_json(
                run_dir / "manifest.json",
                {
                    "benchmark_cases": ["d7-r2"],
                    "systems": [
                        {"id": "tools"},
                        {"id": "tools-rag"},
                        {"id": "tools-adaptive-rag"},
                    ],
                },
            )
            write_json(
                run_dir / "benchmarks" / "d7-r2.json",
                {
                    "case_id": "d7-r2",
                    "benchmark_view": {
                        "rag_need": "R2",
                        "expected_route": "use_rag",
                    },
                    "rubric": {"steps": []},
                },
            )
            mapping = []
            ratings = []
            for system_id, score in (("tools", 45), ("tools-rag", 85)):
                task_id = f"{system_id}::d7-r2"
                mapping.append(
                    {"task_id": task_id, "case_id": "d7-r2", "system_id": system_id}
                )
                ratings.append(
                    {
                        "task_id": task_id,
                        "case_id": "d7-r2",
                        "total_score": score,
                        "steps": [],
                    }
                )
                write_json(
                    run_dir / "responses" / f"d7-r2__{system_id}.json",
                    {
                        "case_id": "d7-r2",
                        "system_id": system_id,
                        "status": "success",
                        "completion_mode": "native",
                        "rag_enabled": system_id == "tools-rag",
                    },
                )
            write_json(
                run_dir / "responses" / "d7-r2__tools-adaptive-rag.json",
                {
                    "case_id": "d7-r2",
                    "system_id": "tools-adaptive-rag",
                    "status": "success",
                    "completion_mode": "policy_replay",
                    "selected_arm_system_id": "tools-rag",
                    "rag_enabled": True,
                    "routing": {
                        "action": "use_rag",
                        "reason_code": "MISSING_DOMAIN_KNOWLEDGE",
                        "confidence": 0.95,
                        "status": "success",
                    },
                },
            )
            write_json(run_dir / "judge_mapping.json", {"mapping": mapping})
            write_jsonl(run_dir / "ratings.jsonl", ratings)

            analysis = build_reward_analysis(run_dir)
            route_row = analysis["adaptive_rag"]["cases"][0]
            self.assertEqual(85.0, route_row["policy_replay_score"])
            self.assertIsNone(route_row["independent_adaptive_score"])
            self.assertEqual(0.0, route_row["routing_regret"])
            self.assertTrue(route_row["policy_routing_correct"])
            self.assertEqual("R2", route_row["rag_need"])

    def test_analysis_without_judge_scores_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(run_dir / "manifest.json", {"benchmark_cases": ["q1"], "systems": [{"id": "tools"}]})
            analysis = build_reward_analysis(run_dir)
            plan = build_router_update_plan(run_dir, analysis)
            self.assertIsNone(analysis["systems"]["tools"]["mean_task_reward"])
            self.assertEqual("awaiting_routing_evidence", plan["status"])

    def test_profile_default_route_scores_raw_benchmark_before_judging(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(
                run_dir / "evaluation_profile.json",
                {
                    "profile_id": "d1_d6",
                    "system_ids": ["tools-adaptive-rag"],
                    "comparisons": [],
                    "adaptive_rag_analysis": {
                        "no_rag_system": "tools",
                        "always_rag_system": "tools-rag",
                        "adaptive_system": "tools-adaptive-rag",
                        "use_benchmark_expected_route": True,
                        "default_expected_route": "skip_rag",
                        "default_rag_need": "R0",
                    },
                },
            )
            write_json(
                run_dir / "manifest.json",
                {"benchmark_cases": ["q1"], "systems": [{"id": "tools-adaptive-rag"}]},
            )
            write_json(
                run_dir / "benchmarks" / "q1.json",
                {"case_id": "q1", "rubric": {"steps": []}},
            )
            write_json(
                run_dir / "responses" / "q1__tools-adaptive-rag.json",
                {
                    "case_id": "q1",
                    "system_id": "tools-adaptive-rag",
                    "status": "success",
                    "routing": {
                        "action": "skip_rag",
                        "reason_code": "FULLY_SPECIFIED_NUMERIC_TASK",
                        "confidence": 0.9,
                        "status": "success",
                    },
                },
            )

            analysis = build_reward_analysis(run_dir)

            self.assertEqual(1.0, analysis["adaptive_rag"]["routing_accuracy"])
            self.assertEqual(
                "benchmark_expected_route",
                analysis["adaptive_rag"]["routing_accuracy_basis"],
            )
            self.assertEqual("skip_rag", analysis["adaptive_rag"]["cases"][0]["expected_action"])
            self.assertEqual("R0", analysis["adaptive_rag"]["cases"][0]["rag_need"])
            self.assertIsNone(analysis["adaptive_rag"]["mean_routing_regret"])

    def test_cli_exposes_reward_analysis_command(self):
        args = build_parser().parse_args(["reward-analysis", "--run-id", "x"])
        self.assertEqual("x", args.run_id)
        self.assertIsNone(args.output)
        plot_args = build_parser().parse_args(
            ["plot", "--run-id", "x", "--figure", "main-scores"]
        )
        self.assertEqual("main-scores", plot_args.figure)


if __name__ == "__main__":
    unittest.main()
