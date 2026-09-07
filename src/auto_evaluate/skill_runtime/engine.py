"""Small step executor with private evidence, explicit lineage and finite budgets."""
from __future__ import annotations

import copy
import json
import operator
import time
from pathlib import Path
from typing import Any, Protocol

from .contracts import (
    CONVERSIONS, ContractError, ToolResult, ToolSpec, digest, field, json_copy,
    number, require, same_value, validate_plan, validate_task,
)


class Backend(Protocol):
    def execute(self, tool: str, arguments: dict) -> ToolResult: ...


class RunStore:
    """New directory only. Persist each call, not merely the final compact summary."""
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)

    def write(self, name: str, value: Any) -> None:
        with (self.directory / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")

    def event(self, value: dict) -> None:
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


class SkillRuntime:
    """Execute one tool node for one candidate at a time.

    Task extraction is supplied by the caller; validation is NOT proof that the
    extraction matches the question. No teacher/gold/rubric is accepted here.
    A backend adapter is trusted infrastructure, never supplied by a model plan.
    """
    VERSION = "executable-skill-runtime@0.1.0"

    def __init__(self, task: dict, plan: dict, tools: dict[str, ToolSpec], backend: Backend,
                 *, enforcement: str = "enforce", max_tool_calls: int = 24,
                 max_actions: int = 64, max_attempts_per_node: int = 3,
                 run_dir: Path | None = None, artifact: dict | None = None):
        require(enforcement in {"enforce", "observe"}, "INVALID_MODE", enforcement)
        for value in (max_tool_calls, max_actions, max_attempts_per_node):
            require(type(value) is int and value > 0, "INVALID_BUDGET", "Positive integers required")
        self._tools = copy.deepcopy(tools)
        self._task = validate_task(task, self._tools)
        self._plan, self._order = validate_plan(plan, self._tools)
        self._check_nodes(self._plan)
        self._backend = backend
        self.enforcement = enforcement
        self._revision = 1
        self._records: dict[str, dict] = {}
        self._current: dict[tuple[str, str, str], str] = {}
        self._attempts: dict[tuple[str, str, str], int] = {}
        self._events: list[dict] = []
        self._last_errors: dict[tuple[str, str, str], str] = {}
        self._tool_calls = 0
        self._actions = 0
        self._max_tool_calls = max_tool_calls
        self._max_actions = max_actions
        self._max_attempts = max_attempts_per_node
        self._store = RunStore(run_dir) if run_dir else None
        self._task_hash = digest(self._task)
        self._artifact = json_copy(artifact or {})
        if self._store:
            self._store.write("manifest.json", self.manifest())
            self._store.write("task.json", self._task)
            self._store.write("plan-001.json", self._plan)

    def _check_nodes(self, plan: dict) -> None:
        require({n["id"] for n in plan["nodes"]} <= self._task["branches"].keys(),
                "UNKNOWN_BRANCH", "Plan nodes must name extracted task branches")

    def manifest(self) -> dict:
        return {"runtime_version": self.VERSION, "task_hash": self._task_hash,
                "plan_hash": digest(self._plan), "revision": self._revision,
                "origin": self._task["origin"], "enforcement": self.enforcement,
                "artifact": json_copy(self._artifact), "model_executed": False,
                "budgets": {"tool_calls": self._max_tool_calls, "actions": self._max_actions,
                            "attempts_per_node": self._max_attempts}}

    def _emit(self, event: dict) -> None:
        event = json_copy({"sequence": len(self._events) + 1, "revision": self._revision, **event})
        self._events.append(event)
        if self._store:
            self._store.event(event)

    def _candidate(self, candidate_id: str, scenario_id: str) -> dict:
        for candidate in self._task["candidates"]:
            if (candidate["id"], candidate["scenario_id"]) == (candidate_id, scenario_id):
                return candidate
        raise ContractError("UNKNOWN_CANDIDATE", "Unknown candidate/scenario pair")

    def _verified_record(self, evidence_id: str) -> dict:
        require(evidence_id in self._records, "UNKNOWN_EVIDENCE", evidence_id)
        record = self._records[evidence_id]
        payload = {k: v for k, v in record.items() if k != "sha256"}
        require(digest(payload) == record["sha256"], "CORRUPT_EVIDENCE", evidence_id)
        require(record["revision"] == self._revision, "STALE_EVIDENCE", evidence_id)
        require(record["status"] == "success", "FAILED_EVIDENCE", evidence_id)
        return record

    def _resolve(self, binding: dict, candidate: dict) -> tuple[Any, list[dict]]:
        if "literal" in binding:
            return json_copy(binding["literal"]), []
        if "input" in binding:
            return field(candidate["inputs"], binding["input"]), []
        source = binding["result"]
        key = (candidate["id"], candidate["scenario_id"], source["node"])
        require(key in self._current, "DEPENDENCY_NOT_READY", source["node"])
        record = self._verified_record(self._current[key])
        require(record["task_hash"] == self._task_hash
                and (record["candidate_id"], record["scenario_id"], record["node"])
                == key, "LINEAGE_MISMATCH", source["node"])
        require(record["task_valid"], "INVALID_DEPENDENCY", source["node"])
        value = field(record["data"], source["path"])
        conversion = binding.get("conversion", "identity")
        from_unit, _, factor = CONVERSIONS[conversion]
        spec = self._tools[record["tool"]]
        require(source["path"] in spec.output_units, "UNKNOWN_OUTPUT", source["path"])
        if from_unit:
            require(spec.output_units[source["path"]] == from_unit and number(value),
                    "UNIT_MISMATCH", source["path"])
            value *= factor
        return value, [{"evidence_id": record["evidence_id"], "node": source["node"],
                        "path": source["path"], "conversion": conversion}]

    def _expected(self, node_id: str, candidate: dict) -> tuple[dict, dict]:
        arguments = json_copy(candidate["inputs"][node_id])
        lineage = {key: [] for key in arguments}
        for link in self._task.get("links", []):
            if link["target_node"] == node_id:
                value, sources = self._resolve(link["binding"], candidate)
                arguments[link["target_argument"]] = value
                lineage[link["target_argument"]] = sources
        return arguments, lineage

    @staticmethod
    def _input_violations(actual: dict, expected: dict) -> list[dict]:
        violations = []
        for key, expected_value in expected.items():
            if key not in actual:
                violations.append({"code": "MISSING_REQUIRED_BINDING", "field": key})
            elif not same_value(actual[key], expected_value):
                violations.append({"code": "INPUT_DRIFT", "field": key})
        for key in actual.keys() - expected.keys():
            violations.append({"code": "UNDECLARED_INPUT", "field": key})
        return violations

    def execute(self, node_id: str, candidate_id: str, scenario_id: str) -> dict:
        """One action only. Failures remain visible and count against finite budgets."""
        key = (candidate_id, scenario_id, node_id)
        try:
            require(self._actions < self._max_actions, "BUDGET_EXHAUSTED", "Action budget")
            self._actions += 1
            candidate = self._candidate(candidate_id, scenario_id)
            nodes = {node["id"]: node for node in self._plan["nodes"]}
            require(node_id in nodes, "UNKNOWN_NODE", node_id)
            require(key not in self._current, "ALREADY_EXECUTED", node_id)
            require(self._attempts.get(key, 0) < self._max_attempts,
                    "ATTEMPTS_EXHAUSTED", node_id)
            self._attempts[key] = self._attempts.get(key, 0) + 1
            node = nodes[node_id]
            for dependency in node["needs"]:
                ref_key = (candidate_id, scenario_id, dependency)
                require(ref_key in self._current, "DEPENDENCY_NOT_READY", dependency)
                self._verified_record(self._current[ref_key])
            arguments, lineage = {}, {}
            for name, binding in node["arguments"].items():
                arguments[name], lineage[name] = self._resolve(binding, candidate)
            expected, expected_lineage = self._expected(node_id, candidate)
            violations = self._input_violations(arguments, expected)
            if node["tool"] != self._task["branches"][node_id]["tool"]:
                violations.append({"code": "TOOL_MISMATCH", "field": node_id})
            for name, sources in expected_lineage.items():
                if sources and lineage.get(name) != sources:
                    violations.append({"code": "LINEAGE_MISMATCH", "field": name})
            if violations and self.enforcement == "enforce":
                raise ContractError(violations[0]["code"], json.dumps(violations))
            spec = self._tools[node["tool"]]
            spec.check_inputs(arguments)
            require(self._tool_calls < self._max_tool_calls, "BUDGET_EXHAUSTED", "Tool budget")
        except ContractError as exc:
            self._last_errors[key] = exc.code
            event = {"type": "blocked", "node": node_id, "candidate_id": candidate_id,
                     "scenario_id": scenario_id, "code": exc.code, "detail": str(exc)}
            self._emit(event)
            return json_copy(event)

        self._tool_calls += 1
        evidence_id = f"E{self._tool_calls:06d}"
        record = {"evidence_id": evidence_id, "task_hash": self._task_hash,
                  "revision": self._revision, "candidate_id": candidate_id,
                  "scenario_id": scenario_id, "node": node_id, "tool": node["tool"],
                  "arguments": json_copy(arguments), "input_hash": digest(arguments),
                  "lineage": lineage, "violations": violations, "status": "error",
                  "task_valid": False}
        self._emit({"type": "tool_started", "evidence_id": evidence_id,
                    "node": node_id, "candidate_id": candidate_id, "scenario_id": scenario_id,
                    "arguments": arguments, "input_hash": digest(arguments)})
        started = time.monotonic()
        try:
            result = self._backend.execute(node["tool"], json_copy(arguments))
            require(isinstance(result, ToolResult), "INVALID_TOOL_RESULT", node["tool"])
            # Preserve the adapter's full result before checking its normalized projection.
            try:
                record["raw"] = json_copy(result.raw)
            except ContractError:
                record["raw_invalid_json_repr"] = repr(result.raw)
                raise
            record["data"] = json_copy(result.data)
            record["effective_inputs"] = json_copy(result.effective_inputs)
            require(isinstance(result.version, str) and bool(result.version),
                    "MISSING_TOOL_VERSION", node["tool"])
            record["tool_version"] = result.version
            require(isinstance(result.data, dict) and isinstance(result.effective_inputs, dict),
                    "INVALID_TOOL_RESULT", node["tool"])
            for name, value in arguments.items():
                require(name in result.effective_inputs and same_value(result.effective_inputs[name], value),
                        "EFFECTIVE_INPUT_MISMATCH", name)
            for path in self._task["branches"][node_id]["required_outputs"]:
                require(path in spec.output_units and number(field(result.data, path)),
                        "INVALID_OUTPUT", path)
            # Missing contract inputs may have been supplied as backend defaults in observe mode.
            for name, value in expected.items():
                if name not in result.effective_inputs or not same_value(result.effective_inputs[name], value):
                    record["violations"].append({"code": "EFFECTIVE_INPUT_MISMATCH", "field": name})
            record["status"] = "success"
            record["task_valid"] = not record["violations"]
        except Exception as exc:
            record["error"] = {"code": exc.code if isinstance(exc, ContractError) else "TOOL_FAILURE",
                               "message": str(exc)}
            self._last_errors[key] = record["error"]["code"]
        record["latency_ms"] = round((time.monotonic() - started) * 1000)
        record["sha256"] = digest(record)
        self._records[evidence_id] = record
        if record["status"] == "success":
            self._current[key] = evidence_id
            self._last_errors.pop(key, None)
        if self._store:
            self._store.write(f"{evidence_id}.json", record)
        self._emit({"type": "tool_finished", "evidence_id": evidence_id,
                    "status": record["status"], "task_valid": record["task_valid"]})
        return json_copy(record)

    def replace_plan(self, plan: dict) -> None:
        """Conservative invalidation. Never reset spent budgets on replanning."""
        plan, order = validate_plan(plan, self._tools)
        self._check_nodes(plan)
        self._plan, self._order = plan, order
        self._revision += 1
        self._current.clear()
        self._attempts.clear()
        self._last_errors.clear()
        self._emit({"type": "plan_replaced", "plan_hash": digest(plan), "invalidated_all": True})
        if self._store:
            self._store.write(f"plan-{self._revision:03d}.json", self._plan)

    def read_evidence(self, evidence_id: str, path: str | None = None) -> Any:
        record = self._verified_record(evidence_id)
        return field(record["data"], path) if path else json_copy(record)

    def context_snapshot(self, *, max_chars: int = 6000) -> dict:
        """Bounded structural state, NOT a tokenizer estimate. Raw data stays retrievable."""
        require(type(max_chars) is int and max_chars > 0, "INVALID_BUDGET", "max_chars")
        snapshot = {"revision": self._revision, "task_hash": self._task_hash,
                    "spent": {"tool_calls": self._tool_calls, "actions": self._actions},
                    "evidence": [{"id": eid, "candidate_id": key[0], "scenario_id": key[1],
                                  "node": key[2]} for key, eid in self._current.items()]}
        require(len(json.dumps(snapshot, ensure_ascii=False)) <= max_chars,
                "CONTEXT_BUDGET_EXCEEDED", "Select fewer fields; do not truncate evidence")
        return snapshot

    def decision(self) -> dict:
        comparisons = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
                       ">=": operator.ge, "==": operator.eq}
        rows, missing = [], []
        for candidate in self._task["candidates"]:
            evidence = {}
            for node in self._task["branches"]:
                key = (candidate["id"], candidate["scenario_id"], node)
                try:
                    require(key in self._current, "MISSING_EVIDENCE", node)
                    record = self._verified_record(self._current[key])
                    require(record["task_valid"], "INVALID_EVIDENCE", record["evidence_id"])
                    require((record["candidate_id"], record["scenario_id"], record["node"]) == key,
                            "LINEAGE_MISMATCH", record["evidence_id"])
                    evidence[node] = record
                except ContractError as exc:
                    missing.append({"candidate_id": key[0], "scenario_id": key[1],
                                    "node": node, "code": exc.code})
            if len(evidence) != len(self._task["branches"]):
                continue
            try:
                checks = []
                for rule in self._task["constraints"]:
                    value = field(evidence[rule["node"]]["data"], rule["path"])
                    require(number(value), "INVALID_OUTPUT", rule["path"])
                    checks.append({"id": rule["id"], "value": value, "op": rule["op"],
                                   "threshold": rule["threshold"], "unit": rule["unit"],
                                   "pass": comparisons[rule["op"]](value, rule["threshold"]),
                                   "evidence_id": evidence[rule["node"]]["evidence_id"]})
                objective = self._task["objective"]
                value = field(evidence[objective["node"]]["data"], objective["path"])
                require(number(value), "INVALID_OUTPUT", objective["path"])
                rows.append({"candidate_id": candidate["id"], "scenario_id": candidate["scenario_id"],
                             "checks": checks, "feasible": all(c["pass"] for c in checks),
                             "objective": value, "evidence_ids": [r["evidence_id"] for r in evidence.values()]})
            except ContractError as exc:
                missing.append({"candidate_id": candidate["id"], "scenario_id": candidate["scenario_id"],
                                "code": exc.code})
        eligible = [r for r in rows if r["feasible"]]
        selected = []
        if missing:
            status = "budget_exhausted" if "BUDGET_EXHAUSTED" in self._last_errors.values() else "insufficient_evidence"
        elif not eligible:
            status = "infeasible_supported"
        else:
            best = (min if self._task["objective"]["direction"] == "min" else max)(r["objective"] for r in eligible)
            selected = [{"candidate_id": r["candidate_id"], "scenario_id": r["scenario_id"]}
                        for r in eligible if r["objective"] == best]
            status = "completed"
        return {"status": status, "claim_scope": "supplied_finite_candidate_scenario_rows_only",
                "selected": selected, "candidates": rows, "missing": missing,
                "quality_score": None, "model_executed": False, "revision": self._revision,
                "spent": {"tool_calls": self._tool_calls, "actions": self._actions},
                "last_errors": [{"candidate_id": k[0], "scenario_id": k[1], "node": k[2], "code": v}
                                for k, v in self._last_errors.items()]}

    def run(self) -> dict:
        for node in self._order:
            for candidate in self._task["candidates"]:
                self.execute(node, candidate["id"], candidate["scenario_id"])
        result = self.decision()
        self._emit({"type": "decision", "result": result})
        return result

    def events(self) -> list[dict]:
        return json_copy(self._events)
