from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from auto_evaluate.io_utils import read_json, write_json  # noqa: E402
from auto_evaluate.router_evaluation import execute_router_evaluation  # noqa: E402


class RouterEvaluationTests(unittest.TestCase):
    def test_zero_shot_and_skill_are_scored_without_solver_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs" / "router_evaluation.json"
            write_json(
                config_path,
                {
                    "generation": {
                        "stream": True,
                        "max_tokens": 64,
                        "enable_thinking": False,
                    },
                    "variants": [
                        {
                            "id": "zero-shot",
                            "display_name": "Zero",
                            "model_id": "router-model",
                            "prompt": "zero prompt",
                        },
                        {
                            "id": "router-skill",
                            "display_name": "Skill",
                            "model_id": "router-model",
                            "prompt": "skill prompt",
                        },
                    ],
                },
            )
            benchmark = {
                "case_id": "q1",
                "question_prompt": "A fully specified numeric task",
                "source": {"sha256": "case-sha"},
            }

            class _Client:
                def __init__(self):
                    self.calls = []

                def chat_stream(self, *, model, messages, generation):
                    self.calls.append((model, messages[0]["content"]))
                    self.assert_generation = generation
                    action = "skip_rag" if messages[0]["content"] == "skill prompt" else "use_rag"
                    need = None if action == "skip_rag" else "external limit"
                    content = (
                        '{"action":"' + action
                        + '","reason_code":"FULLY_SPECIFIED_NUMERIC_TASK",'
                        + '"confidence":0.9,"retrieval_need":'
                        + ("null" if need is None else '"external limit"')
                        + "}"
                    )
                    return type(
                        "Result",
                        (),
                        {"content": content, "raw": {"usage": None}, "latency_ms": 5},
                    )()

            client = _Client()
            with (
                patch(
                    "auto_evaluate.router_evaluation.iter_benchmarks",
                    return_value=iter([benchmark]),
                ),
                patch(
                    "auto_evaluate.router_evaluation.make_client",
                    return_value=client,
                ),
            ):
                result = execute_router_evaluation(
                    benchmarks_dir=root / "benchmarks",
                    config_path=config_path,
                    run_dir=root / "runs" / "router-test",
                    benchmark_set="d1_d6",
                    default_expected_route="skip_rag",
                )

            self.assertEqual(2, result["success"])
            self.assertEqual(2, len(client.calls))
            self.assertFalse(client.assert_generation["enable_thinking"])
            summary = read_json(root / "runs" / "router-test" / "router_summary.json")
            self.assertEqual(0.0, summary["variants"]["zero-shot"]["routing_accuracy"])
            self.assertEqual(1.0, summary["variants"]["router-skill"]["routing_accuracy"])
            self.assertEqual(1.0, summary["skill_effect"]["routing_accuracy_gain"])
            classification = summary["variants"]["zero-shot"]["routing_classification"]
            self.assertEqual(1, classification["confusion_matrix"]["true_skip_pred_use"])
            self.assertEqual(1.0, classification["false_positive_rate"])
            paired = summary["skill_effect"]["paired_mcnemar"]
            self.assertEqual(1, paired["baseline_wrong_candidate_right"])
            self.assertEqual(1.0, paired["exact_two_sided_p"])


if __name__ == "__main__":
    unittest.main()
