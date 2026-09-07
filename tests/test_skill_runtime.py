from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")) if str(ROOT / "src") not in sys.path else None

from auto_evaluate.skill_runtime import ContractError, SkillRuntime, ToolResult, public_question
from auto_evaluate.skill_runtime.artifacts import load_skill
from auto_evaluate.skill_runtime.cli import main
from auto_evaluate.skill_runtime.contracts import digest, validate_plan, validate_task
from auto_evaluate.skill_runtime.replay import FixtureBackend, fixture_tool_specs


EXAMPLES = ROOT / "examples" / "executable_skills"
SKILL = ROOT / "skills" / "swro-parallel-compare" / "v0.1.0"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tools = fixture_tool_specs()
        self.task = read(EXAMPLES / "parallel_task.json")
        self.fixture = read(EXAMPLES / "parallel_fixture.json")
        self.skill = load_skill(SKILL, self.tools)
        self.plan = copy.deepcopy(self.skill["plan"])
        self.backend = FixtureBackend(self.fixture)

    def runtime(self, **kwargs):
        return SkillRuntime(self.task, self.plan, self.tools, self.backend, **kwargs)

    def test_parallel_complete_eight_calls_and_a_selected(self):
        runtime = self.runtime()
        result = runtime.run()
        self.assertEqual("completed", result["status"])
        self.assertEqual([{"candidate_id": "A", "scenario_id": "design"}], result["selected"])
        self.assertEqual(8, len(self.backend.calls))
        self.assertEqual([True, True, False, False], [r["feasible"] for r in result["candidates"]])
        self.assertIsNone(result["quality_score"])
        self.assertFalse(result["model_executed"])

    def test_missing_pressure_blocked_before_backend(self):
        del self.plan["nodes"][1]["arguments"]["p1_pressure_bar"]
        result = self.runtime().execute("plant", "A", "design")
        self.assertEqual("MISSING_REQUIRED_BINDING", result["code"])
        self.assertEqual([], self.backend.calls)

    def test_module_area_cannot_replace_plant_area(self):
        self.plan["nodes"][1]["arguments"]["ro_area_m2"] = {"input": "membrane.membrane_area_m2"}
        result = self.runtime().execute("plant", "A", "design")
        self.assertEqual("INPUT_DRIFT", result["code"])
        self.assertEqual([], self.backend.calls)

    def test_wrong_tool_cannot_be_substituted(self):
        self.plan["nodes"][1].update(tool="simulate_ro", arguments={})
        result = self.runtime().execute("plant", "A", "design")
        self.assertEqual("blocked", result["type"])
        self.assertFalse(self.backend.calls)

    def test_no_decision_with_missing_plant_branch(self):
        self.plan["nodes"] = self.plan["nodes"][:1]
        result = self.runtime().run()
        self.assertEqual("insufficient_evidence", result["status"])
        self.assertFalse(result["selected"])
        self.assertEqual(4, len(result["missing"]))

    def test_task_rejects_hidden_gold_fields(self):
        for key in ("reference_answer", "rubric", "case_id", "task_family"):
            task = copy.deepcopy(self.task)
            task[key] = "SECRET"
            with self.assertRaises(ContractError):
                validate_task(task, self.tools)

    def test_public_question_does_not_leak_hidden_fields(self):
        benchmark = {"question_prompt": "Public question", "reference_answer": "SECRET_ANSWER",
                     "rubric": {"secret": "SECRET_RUBRIC"}, "case_id": "SECRET_ID"}
        self.assertEqual({"question": "Public question"}, public_question(benchmark))
        self.assertNotIn("SECRET", json.dumps(public_question(benchmark)))

    def test_cycle_rejected_before_any_calls(self):
        self.plan["nodes"][0]["needs"] = ["plant"]
        self.plan["nodes"][1]["needs"] = ["membrane"]
        with self.assertRaisesRegex(ContractError, "CYCLIC_PLAN"):
            self.runtime()

    def test_duplicate_ids_and_unknown_dependencies_rejected(self):
        broken = copy.deepcopy(self.plan)
        broken["nodes"].append(copy.deepcopy(broken["nodes"][0]))
        with self.assertRaisesRegex(ContractError, "DUPLICATE_NODE"):
            validate_plan(broken, self.tools)
        broken = copy.deepcopy(self.plan)
        broken["nodes"][0]["needs"] = ["does-not-exist"]
        with self.assertRaisesRegex(ContractError, "UNKNOWN_REFERENCE"):
            validate_plan(broken, self.tools)

    def test_arbitrary_expression_and_nan_rejected(self):
        self.plan["nodes"][0]["arguments"]["feed_pressure_bar"] = {"eval": "open('secret')"}
        with self.assertRaises(ContractError):
            self.runtime()
        self.task["constraints"][0]["threshold"] = float("nan")
        with self.assertRaises(ContractError):
            validate_task(self.task, self.tools)

    def test_units_and_boolean_thresholds_rejected(self):
        self.task["constraints"][0]["unit"] = "m3/s"
        with self.assertRaisesRegex(ContractError, "UNIT_MISMATCH"):
            validate_task(self.task, self.tools)
        self.task["constraints"][0]["unit"] = "kg/s"
        self.task["constraints"][0]["threshold"] = True
        with self.assertRaises(ContractError):
            validate_task(self.task, self.tools)

    def test_replanning_invalidates_evidence_and_preserves_spend(self):
        runtime = self.runtime(max_tool_calls=1)
        record = runtime.execute("membrane", "A", "design")
        runtime.replace_plan(self.plan)
        with self.assertRaisesRegex(ContractError, "STALE_EVIDENCE"):
            runtime.read_evidence(record["evidence_id"])
        self.assertEqual("BUDGET_EXHAUSTED", runtime.execute("membrane", "A", "design")["code"])
        self.assertEqual(1, runtime.decision()["spent"]["tool_calls"])

    def test_returned_objects_cannot_mutate_private_state(self):
        runtime = self.runtime()
        record = runtime.execute("membrane", "A", "design")
        record["data"]["water_kg_s"] = 900
        fetched = runtime.read_evidence("E000001")
        self.assertNotEqual(900, fetched["data"]["water_kg_s"])
        fetched["data"]["water_kg_s"] = 901
        self.assertNotEqual(901, runtime.read_evidence("E000001", "water_kg_s"))
        self.task["candidates"][0]["inputs"]["plant"]["p1_pressure_bar"] = 99
        self.assertEqual("success", runtime.execute("plant", "A", "design")["status"])

    def test_corruption_is_detected_before_decision(self):
        runtime = self.runtime()
        runtime.run()
        runtime._records["E000001"]["data"]["water_kg_s"] = 99
        self.assertEqual("insufficient_evidence", runtime.decision()["status"])
        self.assertEqual("CORRUPT_EVIDENCE", runtime.decision()["missing"][0]["code"])

    def test_wrong_candidate_record_cannot_be_joined(self):
        runtime = self.runtime()
        runtime.run()
        runtime._current[("A", "design", "membrane")] = "E000002"
        self.assertEqual("insufficient_evidence", runtime.decision()["status"])
        self.assertEqual("LINEAGE_MISMATCH", runtime.decision()["missing"][0]["code"])

    def test_budget_exhaustion_not_infeasibility(self):
        result = self.runtime(max_tool_calls=3).run()
        self.assertEqual("budget_exhausted", result["status"])
        self.assertEqual(3, len(self.backend.calls))
        self.assertFalse(result["selected"])

    def test_duplicate_success_is_not_reexecuted(self):
        runtime = self.runtime()
        runtime.execute("membrane", "A", "design")
        self.assertEqual("ALREADY_EXECUTED", runtime.execute("membrane", "A", "design")["code"])
        self.assertEqual(1, len(self.backend.calls))

    def test_invalid_attempts_are_bounded(self):
        del self.plan["nodes"][1]["arguments"]["p1_pressure_bar"]
        runtime = self.runtime(max_attempts_per_node=2)
        runtime.execute("plant", "A", "design")
        runtime.execute("plant", "A", "design")
        self.assertEqual("ATTEMPTS_EXHAUSTED", runtime.execute("plant", "A", "design")["code"])
        self.assertFalse(self.backend.calls)

    def test_backend_failures_count_and_are_retained(self):
        class Fail:
            def execute(self, tool, arguments):
                raise RuntimeError("solver did not converge")
        self.backend = Fail()
        runtime = self.runtime(max_tool_calls=2)
        first = runtime.execute("membrane", "A", "design")
        self.assertEqual("error", first["status"])
        runtime.execute("membrane", "A", "design")
        self.assertEqual("BUDGET_EXHAUSTED", runtime.execute("membrane", "A", "design")["code"])
        self.assertEqual(2, runtime.decision()["spent"]["tool_calls"])

    def test_full_raw_result_saved_and_existing_directory_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            runtime = self.runtime(run_dir=output)
            runtime.run()
            saved = read(output / "E000001.json")
            self.assertEqual(self.fixture["records"][0], saved["raw"])
            self.assertTrue((output / "events.jsonl").exists())
            with self.assertRaises(FileExistsError):
                self.runtime(run_dir=output)

    def test_missing_required_report_field_fails_evidence(self):
        del self.fixture["records"][0]["data"]["water_lmh"]
        self.backend = FixtureBackend(self.fixture)
        runtime = self.runtime()
        record = runtime.execute("membrane", "A", "design")
        self.assertEqual("error", record["status"])
        self.assertIn("raw", record)
        self.assertEqual("insufficient_evidence", runtime.decision()["status"])

    def test_strict_inequality_and_all_infeasible_scope(self):
        self.task["constraints"][2].update(op="<", threshold=1.271138)
        result = self.runtime().run()
        self.assertEqual("infeasible_supported", result["status"])
        self.assertEqual("supplied_finite_candidate_scenario_rows_only", result["claim_scope"])

    def test_candidate_renaming_and_order_do_not_change_physical_choice(self):
        for candidate in self.task["candidates"]:
            candidate["id"] = "new_" + candidate["id"]
        self.task["candidates"].reverse()
        result = self.runtime().run()
        self.assertEqual("new_A", result["selected"][0]["candidate_id"])

    def test_constraint_change_changes_selected_candidate(self):
        self.task["constraints"][1]["threshold"] = 310
        result = self.runtime().run()
        self.assertEqual("B", result["selected"][0]["candidate_id"])

    def test_ties_preserved_without_invented_tie_break(self):
        self.task["constraints"][-1]["threshold"] = 3
        self.task["objective"].update(node="membrane", path="water_kg_s", unit="kg/s", direction="max")
        result = self.runtime().run()
        self.assertEqual(["B", "D"], [r["candidate_id"] for r in result["selected"]])

    def test_context_has_refs_not_raw_and_never_silent_truncation(self):
        runtime = self.runtime()
        runtime.run()
        snapshot = runtime.context_snapshot()
        self.assertEqual(8, len(snapshot["evidence"]))
        self.assertNotIn("data", json.dumps(snapshot))
        with self.assertRaisesRegex(ContractError, "CONTEXT_BUDGET_EXCEEDED"):
            runtime.context_snapshot(max_chars=1)

    def test_observe_records_drift_without_faking_completed_evidence(self):
        self.plan["nodes"][1]["arguments"]["p1_pressure_bar"] = {"literal": 70}
        for row in self.fixture["records"]:
            if row["tool"] == "simulate_swro_system":
                row["arguments"]["p1_pressure_bar"] = 70
        self.backend = FixtureBackend(self.fixture)
        runtime = self.runtime(enforcement="observe")
        record = runtime.execute("plant", "A", "design")
        self.assertEqual("success", record["status"])
        self.assertFalse(record["task_valid"])
        self.assertEqual(1, len(self.backend.calls))

    def serial(self):
        task = {"schema_version": "1.0", "question": "Synthetic lineage test, not chemistry validation",
                "origin": "synthetic_fixture",
                "branches": {"ro": {"tool": "simulate_ro", "required_outputs": ["water_recovery_pct"]},
                             "chem": {"tool": "equilibrate_feed", "required_outputs": ["gypsum_si", "ph"]}},
                "candidates": [{"id": "x", "scenario_id": "s", "inputs": {"ro": {"feed_pressure_bar": 58},
                                                                                 "chem": {"ph": 7}}}],
                "constraints": [{"id": "si", "node": "chem", "path": "gypsum_si", "op": "<", "threshold": 0,
                                 "unit": "SI", "source": "synthetic test"}],
                "objective": {"node": "chem", "path": "ph", "direction": "max", "unit": "pH", "source": "synthetic test"},
                "links": [{"target_node": "chem", "target_argument": "water_recovery", "target_unit": "fraction",
                           "source": "synthetic test", "binding": {"result": {"node": "ro", "path": "water_recovery_pct"},
                                                                     "conversion": "percent_to_fraction"}}]}
        plan = {"schema_version": "1.0", "nodes": [
            {"id": "ro", "tool": "simulate_ro", "needs": [], "arguments": {"feed_pressure_bar": {"input": "ro.feed_pressure_bar"}}},
            {"id": "chem", "tool": "equilibrate_feed", "needs": ["ro"], "arguments": {"ph": {"input": "chem.ph"},
             "water_recovery": copy.deepcopy(task["links"][0]["binding"])}}]}
        fixture = {"fixture_only": True, "source": "synthetic, not a physical calculation", "records": [
            {"tool": "simulate_ro", "arguments": {"feed_pressure_bar": 58}, "data": {"water_recovery_pct": 40}},
            {"tool": "equilibrate_feed", "arguments": {"ph": 7, "water_recovery": 0.4}, "data": {"gypsum_si": -0.1, "ph": 7}}]}
        return task, plan, FixtureBackend(fixture)

    def test_serial_actual_result_converted_and_linked(self):
        task, plan, backend = self.serial()
        runtime = SkillRuntime(task, plan, self.tools, backend)
        result = runtime.run()
        self.assertEqual("completed", result["status"])
        self.assertEqual(0.4, backend.calls[1]["arguments"]["water_recovery"])
        self.assertEqual([{"evidence_id": "E000001", "node": "ro", "path": "water_recovery_pct",
                           "conversion": "percent_to_fraction"}],
                         runtime.read_evidence("E000002")["lineage"]["water_recovery"])

    def test_same_number_from_wrong_source_field_is_not_valid_lineage(self):
        task, plan, _ = self.serial()
        class WrongField:
            def execute(self, tool, arguments):
                return ToolResult({"water_recovery_pct": 40, "water_kg_s": 0.4}, arguments, "test", {})
        plan["nodes"][1]["arguments"]["water_recovery"] = {"result": {"node": "ro", "path": "water_kg_s"}}
        runtime = SkillRuntime(task, plan, self.tools, WrongField())
        runtime.execute("ro", "x", "s")
        self.assertEqual("LINEAGE_MISMATCH", runtime.execute("chem", "x", "s")["code"])

    def test_malformed_enums_return_contract_errors(self):
        for parent, key in ((None, "origin"), ("constraint", "node"), ("constraint", "op"), ("objective", "direction")):
            task = copy.deepcopy(self.task)
            target = task if parent is None else task["constraints"][0] if parent == "constraint" else task["objective"]
            target[key] = []
            with self.assertRaises(ContractError):
                validate_task(task, self.tools)

    def test_boolean_cannot_masquerade_as_numeric_effective_input(self):
        delegate = self.backend
        class BadEcho:
            def execute(self, tool, arguments):
                result = delegate.execute(tool, arguments)
                effective = dict(result.effective_inputs)
                effective["feed_flow_mass_kg_s"] = True
                return ToolResult(result.data, effective, result.version, result.raw)
        self.backend = BadEcho()
        record = self.runtime().execute("membrane", "A", "design")
        self.assertEqual("EFFECTIVE_INPUT_MISMATCH", record["error"]["code"])

    def test_raw_retained_when_normalized_result_has_nan(self):
        class BadProjection:
            def execute(self, tool, arguments):
                return ToolResult({"water_kg_s": float("nan")}, arguments, "test", {"text": "native response"})
        self.backend = BadProjection()
        record = self.runtime().execute("membrane", "A", "design")
        self.assertEqual("error", record["status"])
        self.assertEqual({"text": "native response"}, record["raw"])

    def test_serial_literal_cannot_fake_actual_result_even_if_value_matches(self):
        task, plan, backend = self.serial()
        plan["nodes"][1]["arguments"]["water_recovery"] = {"literal": 0.4}
        runtime = SkillRuntime(task, plan, self.tools, backend)
        runtime.run()
        self.assertEqual(1, len(backend.calls))
        self.assertEqual("LINEAGE_MISMATCH", runtime.decision()["last_errors"][0]["code"])

    def test_serial_dependency_not_ready_blocks(self):
        task, plan, backend = self.serial()
        runtime = SkillRuntime(task, plan, self.tools, backend)
        self.assertEqual("DEPENDENCY_NOT_READY", runtime.execute("chem", "x", "s")["code"])
        self.assertFalse(backend.calls)

    def test_undeclared_serial_reference_rejected(self):
        task, plan, backend = self.serial()
        plan["nodes"][1]["needs"] = []
        with self.assertRaisesRegex(ContractError, "UNDECLARED_DEPENDENCY"):
            SkillRuntime(task, plan, self.tools, backend)

    def test_cli_validation_and_fixture_never_overwrite(self):
        base = ["--task", str(EXAMPLES / "parallel_task.json"), "--skill", str(SKILL)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(0, main(["validate", *base]))
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "run"
                args = ["run-fixture", *base, "--fixture", str(EXAMPLES / "parallel_fixture.json"), "--out", str(out)]
                self.assertEqual(0, main(args))
                original = (out / "result.json").read_bytes()
                self.assertEqual(2, main(args))
                self.assertEqual(original, (out / "result.json").read_bytes())
                self.assertTrue(read(out / "result.json")["fixture_only"])


if __name__ == "__main__":
    unittest.main()
