"""Evidence-recording subset used by ELS. Standard library only.

Methods are copied unchanged from the development runtime; older
registration and skill-discovery APIs are intentionally not exposed.
This is an in-process helper, not a security sandbox.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math


class SkillError(ValueError):
    pass



def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise SkillError("Expected a finite number")
    return value



def field_at(payload, path):
    for name in path.split("."):
        if not isinstance(payload, dict) or name not in payload:
            raise SkillError("Missing returned field: " + path)
        payload = payload[name]
    return payload



def convert(value, source, target):
    # Same physical dimension only. kg/s <-> m3/s needs an explicit density
    # operation and is deliberately unsupported by this first prototype.
    groups = [{"m3/s": 1, "m3/h": 1 / 3600, "m3/day": 1 / 86400},
              {"Pa": 1, "bar": 100000, "MPa": 1000000}]
    number(value)
    if source == target:
        return value
    for group in groups:
        if source in group and target in group:
            return value * group[source] / group[target]
    raise SkillError("Unsupported or dimensionally incompatible conversion")



class ReviewSession:
    def __init__(self, contract, tool_specs, execute=None):
        self._contract = deepcopy(contract)
        self._specs = deepcopy(tool_specs)
        self._execute = execute
        self._events = []
        self._metrics = {}
        self._calls = 0
        self._declared = set()
        self._successful_inputs = {}


    def _candidate(self, candidate):
        if not isinstance(candidate, str) or not candidate.strip():
            raise SkillError("A nonempty agent-chosen candidate ID is required")


    def check_inputs(self, tool, arguments):
        if tool not in self._specs:
            raise SkillError("Unknown tool")
        spec = self._specs[tool]
        issues = []
        for key in spec.get("required", []):
            if key not in arguments:
                issues.append("Missing parameter: " + key)
        for key, value in arguments.items():
            if key not in spec["properties"]:
                issues.append("Unknown parameter: " + key)
                continue
            rule = spec["properties"][key]
            if rule.get("type") == "number":
                try:
                    number(value)
                    if "minimum" in rule and value < rule["minimum"]:
                        issues.append("Below minimum: " + key)
                    if "maximum" in rule and value > rule["maximum"]:
                        issues.append("Above maximum: " + key)
                except SkillError:
                    issues.append("Not a finite number: " + key)
            elif rule.get("type") == "string" and not isinstance(value, str):
                issues.append("Not a string: " + key)
            if "enum" in rule and value not in rule["enum"]:
                issues.append("Invalid enum: " + key)
        fixed = self._contract.get("fixed_inputs", {}).get(tool, {})
        for key, value in fixed.items():
            if key not in arguments or arguments[key] != value:
                issues.append("Public fixed input missing/changed: " + key)
        for rule in spec.get("conditional_rules", []):
            if all(arguments.get(k) == v for k, v in rule["when"].items()):
                for key in rule.get("forbid", []):
                    if key in arguments:
                        issues.append("Inapplicable parameter: " + key)
                for key in rule.get("require", []):
                    if key not in arguments:
                        issues.append("Conditionally required parameter: " + key)
        return {"valid": not issues, "issues": issues}


    def call_tool(self, candidate, tool, arguments, bindings=None):
        """Called only when the agent chooses a physical tool action.

        execute(tool, args) is a trusted host callback returning
        {status: 'success'|'error', data: {...}}. It must unwrap domain errors,
        not merely report HTTP/transport success. Failed attempts cost budget.
        """
        self._candidate(candidate)
        args = deepcopy(arguments)
        for parameter, reference in (bindings or {}).items():
            metric = self._get_metric(candidate, reference)
            expected_unit = self._specs.get(tool, {}).get("properties", {}).get(parameter, {}).get("unit")
            if not expected_unit or metric["unit"] != expected_unit:
                raise SkillError("Bound metric and input units are undeclared or incompatible")
            if parameter in args and args[parameter] != metric["value"]:
                raise SkillError("Bound value differs from supplied argument")
            args[parameter] = metric["value"]
        check = self.check_inputs(tool, args)
        if not check["valid"]:
            self._events.append({"kind": "rejected_call", "candidate": candidate,
                                 "tool": tool, "arguments": args, "issues": check["issues"]})
            raise SkillError("; ".join(check["issues"]))
        previous = self._successful_inputs.get((candidate, tool))
        if previous is not None and previous != args:
            self._events.append({"kind": "rejected_call", "candidate": candidate,
                                 "tool": tool, "arguments": args,
                                 "issues": ["Candidate inputs changed after a successful observation"]})
            raise SkillError("Create a new candidate/scenario revision for changed design inputs")
        if self._execute is None:
            raise SkillError("No host tool executor attached; cannot run physics")
        if self._calls >= self._contract.get("max_calls", 20):
            raise SkillError("Frozen tool-call budget exhausted")
        self._calls += 1
        try:
            output = self._execute(tool, deepcopy(args))
            if not isinstance(output, dict) or output.get("status") not in {"success", "error"}:
                raise SkillError("Host callback returned an invalid status envelope")
            if output["status"] == "success" and not isinstance(output.get("data"), dict):
                raise SkillError("Successful callback must contain structured data")
            json.dumps(output, allow_nan=False)
        except Exception as exc:
            output = {"status": "error", "data": {}, "error": str(exc)}
        event_id = "T%03d" % self._calls
        payload = {"event_id": event_id, "kind": "physical_tool", "candidate": candidate,
                   "tool": tool, "arguments": args, "bindings": deepcopy(bindings or {}),
                   "result": deepcopy(output), "execution_origin": self._contract.get("execution_origin", "host")}
        payload["sha256"] = sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self._events.append(payload)
        self._declared.add(candidate)
        if output["status"] == "success":
            self._successful_inputs[candidate, tool] = deepcopy(args)
        return deepcopy(payload)


    def _get_metric(self, candidate, name):
        if (candidate, name) not in self._metrics:
            raise SkillError("Missing evidence-backed metric: " + name)
        return deepcopy(self._metrics[candidate, name])


    def _store(self, candidate, name, metric):
        if (candidate, name) in self._metrics:
            raise SkillError("Metric is immutable; choose a new name or candidate revision")
        self._metrics[candidate, name] = deepcopy(metric)
        return deepcopy(metric)


    def record_metric(self, candidate, name, event_id, field):
        event = next((e for e in self._events if e.get("event_id") == event_id), None)
        if not event or event["candidate"] != candidate or event["result"]["status"] != "success":
            raise SkillError("Only successful evidence from the same candidate is admissible")
        output_spec = self._specs[event["tool"]].get("output_fields", {})
        if field not in output_spec:
            raise SkillError("Output field/unit must be declared by the host tool adapter")
        meta = output_spec[field]
        value = number(field_at(event["result"]["data"], field))
        return self._store(candidate, name, {"value": value, "unit": meta["unit"],
            "basis": meta.get("basis"), "source_events": [event_id], "field": field})


    def export_trace(self):
        return deepcopy({"events": self._events, "physical_attempts": self._calls,
                         "execution_origin": self._contract.get("execution_origin", "host")})
