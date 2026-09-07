"""Strict, JSON-only contracts. No eval, template execution or benchmark lookup."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ContractError(code, message)


def json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_JSON", "Expected finite JSON data") from exc


def digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def exact_keys(value: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    require(isinstance(value, dict), "INVALID_SCHEMA", "Expected an object")
    require(required <= value.keys(), "INVALID_SCHEMA", f"Missing fields: {required - value.keys()}")
    require(value.keys() <= required | optional, "INVALID_SCHEMA",
            f"Unknown fields: {value.keys() - required - optional}")


def identifier(value: Any) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is not None,
            "INVALID_ID", "IDs must be 1-80 letters, numbers, underscores or hyphens")


def number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def same_value(left: Any, right: Any) -> bool:
    """Numeric spelling is irrelevant, but booleans are not numbers."""
    if number(left) and number(right):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_value(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    return left == right


def field(data: dict, path: str) -> Any:
    require(isinstance(path, str) and bool(path), "INVALID_PATH", "Expected a field path")
    current: Any = data
    for part in path.split("."):
        require(isinstance(current, dict) and part in current, "MISSING_FIELD", path)
        current = current[part]
    return copy.deepcopy(current)


def public_question(benchmark: dict) -> dict[str, str]:
    """Allowlist, never a filtered copy of a full answer/rubric-bearing object."""
    text = benchmark.get("question_prompt")
    require(isinstance(text, str) and bool(text.strip()), "INVALID_QUESTION", "Question is required")
    return {"question": text}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_names: frozenset[str]
    output_units: dict[str, str]
    required_inputs: frozenset[str] = frozenset()
    validate_inputs: Callable[[dict], None] | None = None

    def check_inputs(self, arguments: dict) -> None:
        require(isinstance(arguments, dict), "INVALID_ARGUMENTS", self.name)
        json_copy(arguments)
        require(arguments.keys() <= self.input_names, "UNKNOWN_ARGUMENT", self.name)
        require(self.required_inputs <= arguments.keys(), "MISSING_TOOL_ARGUMENT", self.name)
        if self.validate_inputs:
            self.validate_inputs(copy.deepcopy(arguments))


@dataclass(frozen=True)
class ToolResult:
    """Trusted adapter output, including the full native payload and effective defaults.

    data uses the adapter's declared field/unit contract. Units are not inferred by
    the LLM. A successful HTTP call alone must not produce this success object.
    """
    data: dict
    effective_inputs: dict
    version: str
    raw: Any


CONVERSIONS = {
    "identity": (None, None, 1.0),
    "percent_to_fraction": ("%", "fraction", 0.01),
}


def validate_binding(binding: Any, node_ids: set[str]) -> set[str]:
    require(isinstance(binding, dict), "INVALID_BINDING", "Bindings must be explicit objects")
    if set(binding) == {"literal"}:
        json_copy(binding["literal"])
        return set()
    if set(binding) == {"input"}:
        require(isinstance(binding["input"], str) and "." in binding["input"],
                "INVALID_BINDING", "Input reference requires branch.field")
        return set()
    exact_keys(binding, {"result"}, {"conversion"})
    exact_keys(binding["result"], {"node", "path"})
    require(isinstance(binding["result"]["node"], str) and binding["result"]["node"] in node_ids,
            "UNKNOWN_REFERENCE", "Unknown source node")
    require(isinstance(binding["result"]["path"], str) and bool(binding["result"]["path"]),
            "INVALID_BINDING", "Result path is required")
    require(isinstance(binding.get("conversion", "identity"), str)
            and binding.get("conversion", "identity") in CONVERSIONS, "UNKNOWN_CONVERSION", "Not allowlisted")
    return {binding["result"]["node"]}


def validate_task(task: dict, tools: dict[str, ToolSpec]) -> dict:
    task = json_copy(task)
    exact_keys(task, {"schema_version", "question", "origin", "candidates", "branches",
                      "constraints", "objective"}, {"links"})
    require(task["schema_version"] == "1.0", "INVALID_SCHEMA", "Task version must be 1.0")
    require(isinstance(task["origin"], str)
            and task["origin"] in {"model_extraction", "human_reviewed_development", "synthetic_fixture"},
            "INVALID_ORIGIN", "Extraction origin must be disclosed")
    require(isinstance(task["question"], str) and bool(task["question"].strip()),
            "INVALID_QUESTION", "Question is required")
    branches = task["branches"]
    require(isinstance(branches, dict) and bool(branches), "INVALID_SCHEMA", "Branches required")
    for name, branch in branches.items():
        identifier(name)
        exact_keys(branch, {"tool", "required_outputs"})
        require(isinstance(branch["tool"], str) and branch["tool"] in tools, "UNKNOWN_TOOL", str(branch["tool"]))
        required = branch["required_outputs"]
        require(isinstance(required, list) and bool(required) and all(isinstance(p, str) for p in required),
                "INVALID_SCHEMA", "Required output paths must be a nonempty list")
        require(set(required) <= tools[branch["tool"]].output_units.keys(),
                "UNKNOWN_OUTPUT", name)
    require(isinstance(task["candidates"], list) and bool(task["candidates"]),
            "INVALID_SCHEMA", "Candidates required")
    identities = set()
    for candidate in task["candidates"]:
        exact_keys(candidate, {"id", "scenario_id", "inputs"})
        identifier(candidate["id"])
        identifier(candidate["scenario_id"])
        identity = (candidate["id"], candidate["scenario_id"])
        require(identity not in identities, "DUPLICATE_CANDIDATE", str(identity))
        identities.add(identity)
        require(isinstance(candidate["inputs"], dict) and set(candidate["inputs"]) == set(branches),
                "INVALID_SCHEMA", "Every candidate needs an input object for every branch")
        for name, inputs in candidate["inputs"].items():
            require(isinstance(inputs, dict), "INVALID_SCHEMA", "Inputs must be objects")
            require(inputs.keys() <= tools[branches[name]["tool"]].input_names,
                    "UNKNOWN_ARGUMENT", name)
    require(isinstance(task["constraints"], list) and bool(task["constraints"]),
            "INVALID_SCHEMA", "Constraints required")
    constraint_ids = set()
    for constraint in task["constraints"]:
        exact_keys(constraint, {"id", "node", "path", "op", "threshold", "unit", "source"})
        identifier(constraint["id"])
        require(constraint["id"] not in constraint_ids, "DUPLICATE_CONSTRAINT", constraint["id"])
        constraint_ids.add(constraint["id"])
        require(isinstance(constraint["op"], str)
                and constraint["op"] in {"<", "<=", ">", ">=", "=="} and number(constraint["threshold"]),
                "INVALID_CONSTRAINT", constraint["id"])
        require(isinstance(constraint["source"], str) and bool(constraint["source"].strip()),
                "MISSING_SOURCE", constraint["id"])
        validate_metric(constraint, branches, tools)
    objective = task["objective"]
    exact_keys(objective, {"node", "path", "unit", "direction", "source"})
    require(isinstance(objective["direction"], str) and objective["direction"] in {"min", "max"},
            "INVALID_OBJECTIVE", "min or max required")
    require(isinstance(objective["source"], str) and bool(objective["source"].strip()),
            "MISSING_SOURCE", "Objective source required")
    validate_metric(objective, branches, tools)
    links = task.get("links", [])
    require(isinstance(links, list), "INVALID_SCHEMA", "Links must be a list")
    targets = set()
    for link in links:
        exact_keys(link, {"target_node", "target_argument", "binding", "source", "target_unit"})
        require(isinstance(link["target_node"], str) and link["target_node"] in branches,
                "UNKNOWN_REFERENCE", "Unknown target branch")
        require(isinstance(link["target_argument"], str), "UNKNOWN_ARGUMENT", "Expected argument name")
        target = (link["target_node"], link["target_argument"])
        require(target not in targets, "DUPLICATE_LINK", str(target))
        targets.add(target)
        require(link["target_argument"] in tools[branches[link["target_node"]]["tool"]].input_names,
                "UNKNOWN_ARGUMENT", link["target_argument"])
        require(isinstance(link["source"], str) and bool(link["source"].strip()), "MISSING_SOURCE", str(target))
        deps = validate_binding(link["binding"], set(branches))
        require(bool(deps), "INVALID_LINK", "Task links must reference a result")
        source = link["binding"]["result"]
        source_spec = tools[branches[source["node"]]["tool"]]
        require(source["path"] in source_spec.output_units, "UNKNOWN_OUTPUT", source["path"])
        conversion = link["binding"].get("conversion", "identity")
        from_unit, to_unit, _ = CONVERSIONS[conversion]
        actual_unit = source_spec.output_units[source["path"]]
        require((from_unit is None or actual_unit == from_unit)
                and link["target_unit"] == (to_unit or actual_unit), "UNIT_MISMATCH", str(target))
        for candidate in task["candidates"]:
            require(link["target_argument"] not in candidate["inputs"][link["target_node"]],
                    "AMBIGUOUS_BINDING", "Cannot provide both a fixed input and a link")
    return task


def validate_metric(metric: dict, branches: dict, tools: dict[str, ToolSpec]) -> None:
    require(isinstance(metric["node"], str) and metric["node"] in branches,
            "UNKNOWN_REFERENCE", str(metric["node"]))
    units = tools[branches[metric["node"]]["tool"]].output_units
    require(isinstance(metric["path"], str) and metric["path"] in units, "UNKNOWN_OUTPUT", str(metric["path"]))
    require(units[metric["path"]] == metric["unit"], "UNIT_MISMATCH", metric["path"])


def validate_plan(plan: dict, tools: dict[str, ToolSpec]) -> tuple[dict, list[str]]:
    plan = json_copy(plan)
    exact_keys(plan, {"schema_version", "nodes"}, {"skill_id"})
    require(plan["schema_version"] == "1.0", "INVALID_SCHEMA", "Plan version must be 1.0")
    require(isinstance(plan["nodes"], list) and bool(plan["nodes"]), "INVALID_PLAN", "Nodes required")
    nodes = {}
    for node in plan["nodes"]:
        exact_keys(node, {"id", "tool", "arguments", "needs"})
        identifier(node["id"])
        require(node["id"] not in nodes, "DUPLICATE_NODE", node["id"])
        require(isinstance(node["tool"], str) and node["tool"] in tools, "UNKNOWN_TOOL", str(node["tool"]))
        require(isinstance(node["arguments"], dict), "INVALID_PLAN", "Arguments must be an object")
        require(node["arguments"].keys() <= tools[node["tool"]].input_names, "UNKNOWN_ARGUMENT", node["id"])
        require(isinstance(node["needs"], list) and all(isinstance(n, str) for n in node["needs"]),
                "INVALID_PLAN", "needs must be a list of node IDs")
        require(len(node["needs"]) == len(set(node["needs"])), "INVALID_PLAN", "Duplicate dependencies")
        nodes[node["id"]] = node
    pending = {}
    for name, node in nodes.items():
        deps = set(node["needs"])
        require(deps <= nodes.keys(), "UNKNOWN_REFERENCE", name)
        for binding in node["arguments"].values():
            refs = validate_binding(binding, set(nodes))
            require(refs <= deps, "UNDECLARED_DEPENDENCY", name)
        pending[name] = deps
    order = []
    while pending:
        ready = [name for name, deps in pending.items() if deps <= set(order)]
        require(bool(ready), "CYCLIC_PLAN", "Dependency cycle")
        for name in ready:
            order.append(name)
            del pending[name]
    return plan, order
