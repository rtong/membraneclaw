"""Bounded executable workflow tool for D1-D5 public task contracts.

The model supplies a small, structured contract.  This module owns all physical
solver calls, caching, numeric checks, aggregation, and the verified evidence
report.  It never reads benchmark answers or rubrics.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import json
import math
import operator
import re
from typing import Any, Callable


SOLVER_TOOLS = {
    "simulate_ro",
    "simulate_swro_system",
    "equilibrate_feed",
    "analyze_ro_scaling",
}
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
_ARGUMENT_ALIASES = {
    "simulate_ro": {
        "A_comp_m_s_Pa": "A_comp", "B_comp_m_s": "B_comp",
        "NaCl_mass_fraction": "feed_nacl_mass_frac",
        "feed_NaCl_mass_frac": "feed_nacl_mass_frac",
        "feed_mass_fraction_NaCl": "feed_nacl_mass_frac",
        "feed_salinity_mass_frac": "feed_nacl_mass_frac",
        "temperature_C": "feed_temperature_c",
        "Temperature_C": "feed_temperature_c",
        "feed_temp_C": "feed_temperature_c",
        "feed_temperature_C": "feed_temperature_c",
        "feed_mass_flow_kg_s": "feed_flow_mass_kg_s",
        "feed_flow_kg_s": "feed_flow_mass_kg_s",
        "active_membrane_area_m2": "membrane_area_m2",
        "membrane_area": "membrane_area_m2",
        "permeate_side_pressure_bar": "permeate_pressure_bar",
        "module_pressure_drop_bar": "pressure_drop_bar",
        "equivalent_channel_length_m": "module_length_m",
        "feed.temperature": "feed_temperature_c",
        "feed.solute_mass_fraction": "feed_nacl_mass_frac",
        "feed.pressure": "feed_pressure_bar",
        "feed.flow": "feed_flow_mass_kg_s",
        "membrane.A_comp": "A_comp",
        "membrane.B_comp": "B_comp",
        "membrane.area": "membrane_area_m2",
        "permeate.pressure": "permeate_pressure_bar",
        "module.pressure_drop": "pressure_drop_bar",
        "channel.height": "channel_height_m",
        "channel.spacer_porosity": "spacer_porosity",
        "channel.length": "module_length_m",
    },
    "simulate_swro_system": {
        "feed_flow": "feed_flow_m3_s",
        "feed_concentration_TDS_g_L": "feed_tds_g_L",
        "feed_concentration_TSS_g_L": "feed_tss_g_L",
        "TDS": "feed_tds_g_L", "TSS": "feed_tss_g_L",
        "temperature": "feed_temperature_c",
        "feed_temperature_C": "feed_temperature_c",
        "temperature_C": "feed_temperature_c",
        "active_ro_area_m2": "ro_area_m2",
        "active_membrane_area": "ro_area_m2",
        "P1_outlet_pressure": "p1_pressure_bar",
        "p1_outlet_pressure_bar": "p1_pressure_bar",
        "P1_pressure_bar": "p1_pressure_bar",
        "P1_efficiency": "p1_efficiency",
        "PX_efficiency": "pxr_efficiency",
        "P2_efficiency": "p2_efficiency",
        "ERD_efficiency": "erd_efficiency",
        "energy_recovery": "erd_type",
        "membrane_area_m2": "ro_area_m2",
        "active_membrane_area_m2": "ro_area_m2",
        "membrane.water_permeability_A": "A_comp",
    },
    "equilibrate_feed": {
        "T": "temperature_c", "P": "pressure_bar",
        "temperature": "temperature_c", "pressure": "pressure_bar",
        "temperature_C": "temperature_c", "pH": "ph",
        "recovery": "water_recovery",
        "HCl_dose_mol_s": "acid_addition_mol_s",
        "HCl_dose": "acid_addition_mol_s",
        "NaOH_dose_mol_s": "base_addition_mol_s",
        "feed": "composition_mol_s",
    },
    "analyze_ro_scaling": {
        "temperature_C": "feed_temperature_c", "pH": "ph",
        "feed_mass_flow_kg_s": "feed_flow_mass_kg_s",
        "active_membrane_area_m2": "membrane_area_m2",
        "HCl_dose_mol_s": "acid_addition_mol_s",
        "NaOH_dose_mol_s": "base_addition_mol_s",
    },
}
_ALLOWED_ARGUMENTS = {
    "simulate_ro": {
        "feed_flow_mass_kg_s", "feed_nacl_mass_frac", "feed_pressure_bar",
        "feed_temperature_c", "membrane_area_m2", "A_comp", "B_comp",
        "permeate_pressure_bar", "pressure_drop_bar", "channel_height_m",
        "spacer_porosity", "module_length_m", "cp_modulus",
        "mass_transfer_coeff", "concentration_polarization",
        "mass_transfer_coefficient",
    },
    "simulate_swro_system": {
        "erd_type", "feed_flow_m3_s", "feed_tds_g_L", "feed_tss_g_L",
        "feed_temperature_c", "ro_area_m2", "A_comp", "B_comp",
        "p1_pressure_bar", "p1_efficiency", "pxr_efficiency",
        "p2_efficiency", "erd_efficiency",
    },
    "equilibrate_feed": {
        "composition_mol_s", "temperature_c", "pressure_bar", "ph", "pe",
        "water_recovery", "acid_addition_mol_s", "base_addition_mol_s",
        "minerals",
    },
    "analyze_ro_scaling": {
        "composition_mol_s", "feed_pressure_bar", "membrane_area_m2",
        "feed_flow_mass_kg_s", "feed_temperature_c", "ph",
        "acid_addition_mol_s", "base_addition_mol_s", "minerals",
    },
}
_METRIC_ALIASES = {
    "permeate_flow_m3_h": ("permeate.water_kg_s", 3.6),
    "permeate.flow_m3_h": ("permeate.water_kg_s", 3.6),
    "permeate.flow": ("permeate.water_kg_s", 3.6),
    "recovery_pct": ("performance.water_recovery_pct", 1.0),
    "flux_LMH": ("flux.water_LMH", 1.0),
    "rejection_pct": ("performance.salt_rejection_pct", 1.0),
    "salt_rejection_pct": ("performance.salt_rejection_pct", 1.0),
    "permeate_NaCl_mg_L": ("permeate.nacl_mg_L", 1.0),
    "performance.product_flow_m3_d": ("performance.product_flow_m3_s", 86400.0),
    "performance.sec_kwh_m3": ("costing.specific_energy_kWh_m3", 1.0),
    "performance.p2_power_kw": ("desalination.p2_power_kW", 1.0),
}
_CHEMISTRY_SPECIES = {
    "H2O", "Na", "K", "Ca", "Mg", "Ba", "Sr", "Li", "Cl", "SO4",
    "HCO3", "CO3", "F", "Br", "B", "SiO2",
}
_TASK_ONLY_ARGUMENTS = {
    "erd_inlet_loss_bar", "erd_outlet_pressure_bar", "operating_days_y",
    "target_flow_m3_d", "margin_pct",
}
_SCALABLE_METRICS = {
    "performance.product_flow_m3_s",
    "costing.total_capital_cost_usd",
    "costing.total_operating_cost_usd_year",
}


def _fold_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


_ALIAS_INDEX = {
    tool: {
        **{_fold_name(name): name for name in _ALLOWED_ARGUMENTS[tool]},
        **{_fold_name(alias): target for alias, target in aliases.items()},
    }
    for tool, aliases in _ARGUMENT_ALIASES.items()
}
_METRIC_ALIAS_INDEX = {_fold_name(name): value for name, value in _METRIC_ALIASES.items()}


class ProgramToolError(ValueError):
    pass


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProgramToolError(f"{label} must be a finite number")
    return float(value)


def _field(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ProgramToolError(f"tool result missing {path}")
        current = current[part]
    return current


def _set_path(payload: dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ProgramToolError(f"variable path conflicts with non-object: {path}")
        current = child
    current[parts[-1]] = value


def _merge(*parts: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in parts:
        if not isinstance(part, dict):
            raise ProgramToolError("arguments must be objects")
        for key, value in part.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def _canonical_argument_name(tool: str, original: str) -> str | None:
    return _ALIAS_INDEX.get(tool, {}).get(_fold_name(original))


def _normalize_arguments(tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    allowed = _ALLOWED_ARGUMENTS.get(tool, set())
    normalized: dict[str, Any] = {}
    ignored: list[str] = []
    species_index = {_fold_name(name): name for name in _CHEMISTRY_SPECIES}
    species: dict[str, Any] = {}
    for original, value in arguments.items():
        folded_original = _fold_name(original)
        if tool == "simulate_ro" and folded_original == _fold_name("cp_mass_transfer"):
            for name in ("concentration_polarization", "mass_transfer_coefficient"):
                normalized[name] = copy.deepcopy(value)
            continue
        species_name = species_index.get(folded_original) if tool in {"equilibrate_feed", "analyze_ro_scaling"} else None
        if species_name is not None:
            species[species_name] = copy.deepcopy(value)
            continue
        name = _canonical_argument_name(tool, original)
        if name == "erd_type" and isinstance(value, str):
            folded = value.strip().casefold().replace("-", "_").replace(" ", "_")
            value = "pressure_exchanger" if folded in {"px", "pxr", "pressure_exchanger"} else "pump_as_turbine" if folded in {"pat", "pump_as_turbine"} else value
        if name not in allowed:
            task_only = folded_original in {_fold_name(item) for item in _TASK_ONLY_ARGUMENTS}
            looks_like_limit = any(token in original.casefold() for token in ("_min", "_max", "target", "limit", "requirement"))
            if task_only or looks_like_limit or folded_original in _METRIC_ALIAS_INDEX:
                ignored.append(original)
                continue
            raise ProgramToolError(f"{tool} has unknown task input {original!r}")
        if name == "composition_mol_s":
            if not isinstance(value, dict):
                raise ProgramToolError(f"{tool} composition must be an object")
            species.update(copy.deepcopy(value))
            continue
        if name in normalized and normalized[name] != value:
            raise ProgramToolError(f"{tool} has conflicting aliases for {name}")
        normalized[name] = copy.deepcopy(value)
    if species:
        normalized["composition_mol_s"] = species
    if tool == "simulate_swro_system":
        erd_type = normalized.get("erd_type", "pressure_exchanger")
        if erd_type == "pressure_exchanger" and "erd_efficiency" in normalized:
            normalized.setdefault("pxr_efficiency", normalized["erd_efficiency"])
            normalized.pop("erd_efficiency", None)
        elif erd_type == "pump_as_turbine" and "pxr_efficiency" in normalized:
            normalized.setdefault("erd_efficiency", normalized["pxr_efficiency"])
            normalized.pop("pxr_efficiency", None)
            normalized.pop("p2_efficiency", None)
    return normalized, ignored


def _normalize_variable(tool: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ProgramToolError("variable.path is required")
    normalized = copy.deepcopy(value)
    canonical = _canonical_argument_name(tool, normalized["path"])
    if canonical is None:
        raise ProgramToolError(f"{tool} has unknown variable path {normalized['path']!r}")
    normalized["path"] = canonical
    numeric_fields = ("lower", "upper", "resolution")
    numeric_values = [_finite(normalized.get(name), f"variable.{name}") for name in numeric_fields]
    probes = normalized.get("probe_values") or []
    if not isinstance(probes, list):
        raise ProgramToolError("variable.probe_values must be an array")
    probe_values = [_finite(item, "probe value") for item in probes]
    if canonical == "A_comp" and max(abs(item) for item in numeric_values + probe_values) > 1e-6:
        raise ProgramToolError("A_comp bounds/resolution must use m/s/Pa, e.g. 4.2e-12 and 0.01e-12; no implicit unit guessing")
    for name, item in zip(numeric_fields, numeric_values):
        normalized[name] = item
    normalized["probe_values"] = probe_values
    return normalized


def _normalize_constraints(value: Any, *, field: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProgramToolError(f"{field} string is not valid JSON") from exc
    if not isinstance(value, list) or not value:
        raise ProgramToolError(f"{field} must be a non-empty array")
    normalized = []
    for index, rule in enumerate(value):
        if not isinstance(rule, dict):
            raise ProgramToolError(f"{field}[{index}] must be an object")
        path, op = rule.get("path"), rule.get("op")
        if not isinstance(path, str) or not path or op not in OPS:
            raise ProgramToolError(f"{field}[{index}] has invalid path/op")
        scale = _finite(rule.get("scale", 1.0), f"{field}[{index}].scale")
        threshold = _finite(rule.get("threshold"), f"{field}[{index}].threshold")
        scenarios = rule.get("scenario_ids")
        if scenarios is not None and (
            not isinstance(scenarios, list)
            or any(not isinstance(item, str) or not item for item in scenarios)
        ):
            raise ProgramToolError(f"{field}[{index}].scenario_ids is invalid")
        normalized.append({
            "id": str(rule.get("id") or f"C{index + 1}"),
            "path": path,
            "op": op,
            "threshold": threshold,
            "scale": scale,
            "unit": str(rule.get("unit") or ""),
            "scenario_ids": scenarios,
        })
    return normalized


def _augment_required_arguments(tool: str, arguments: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    augmented = copy.deepcopy(arguments)
    if tool in {"equilibrate_feed", "analyze_ro_scaling"}:
        minerals = {
            rule["path"].split(".")[-1]
            for rule in rules
            if ".saturation_index." in f".{rule['path']}"
        }
        if minerals:
            existing = augmented.get("minerals") or []
            if isinstance(existing, list):
                augmented["minerals"] = sorted(set(existing) | minerals)
    return augmented


def validate_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate every row before the first expensive physical call."""
    from .program_contract import FAMILY_GUIDANCE
    result = copy.deepcopy(spec)
    errors = []
    if result.get("task_family") not in FAMILY_GUIDANCE:
        errors.append("task_family must match the supplied public family")
    mode = result.get("mode")
    if mode not in {"finite_candidates", "one_dimensional_search"}:
        errors.append("mode must be finite_candidates or one_dimensional_search")
    budget = result.get("max_physical_calls", 32)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 64:
        errors.append("max_physical_calls must be an integer in 1..64")
    tools = {r.get("tool") for r in (result.get("rows") or []) if isinstance(r, dict) and r.get("tool")}
    tool = result.get("solver_tool") or (next(iter(tools)) if len(tools) == 1 else None)
    if tool not in SOLVER_TOOLS or tools - {tool}:
        errors.append("one program contract must use one supported solver_tool")
    result["solver_tool"] = tool
    rules = []
    for field in ("constraints", "recommended_constraints"):
        if field == "recommended_constraints" and not result.get(field): continue
        try:
            normalized = _normalize_constraints(result.get(field), field=field)
            result[field] = normalized
            rules.extend(normalized)
        except ProgramToolError as exc:
            errors.append(str(exc))
    scenarios = result.get("scenarios") or []
    # Preserve actual scenario IDs supplied in row-form search requests.
    if not scenarios and mode == "one_dimensional_search":
        grouped = {}
        for row in result.get("rows") or []:
            if isinstance(row, dict): grouped.setdefault(str(row.get("scenario_id") or "design"), []).append(row.get("arguments") or {})
        variable_name = _canonical_argument_name(tool, (result.get("variable") or {}).get("path", ""))
        for sid, args in grouped.items():
            canonical = [_normalize_arguments(tool, a)[0] for a in args]
            common = {k:v for k,v in canonical[0].items() if k != variable_name and all(a.get(k)==v for a in canonical)}
            scenarios.append({"id":sid,"arguments":common})
        if not scenarios: scenarios = [{"id":"design","arguments":{}}]
    ids = [str(x.get("id", "")) for x in scenarios if isinstance(x,dict)]
    if not isinstance(scenarios, list) or len(ids)!=len(scenarios) or any(not x for x in ids) or len(set(ids))!=len(ids):
        errors.append("scenarios require unique non-empty IDs")
    result["scenarios"] = scenarios
    rows = result.get("rows") or []
    if mode == "finite_candidates" and (not isinstance(rows,list) or not 1 <= len(rows) <= 64):
        errors.append("finite_candidates requires 1..64 rows")
    scenario_ids = set(ids) if ids else {str(r.get("scenario_id") or "design") for r in rows if isinstance(r,dict)}
    for rule in rules:
        unknown = set(rule.get("scenario_ids") or []) - scenario_ids
        if unknown: errors.append(f"constraint {rule['id']} refers to unknown scenarios {sorted(unknown)}")
        if rule["path"] in _ALLOWED_ARGUMENTS.get(tool,set()):
            errors.append(f"constraint {rule['id']} uses an input as an output metric: {rule['path']}; do not invent a design constraint")
    for sid in scenario_ids:
        if not any(r.get("scenario_ids") is None or sid in r["scenario_ids"] for r in (result.get("constraints") or []) if isinstance(r,dict)):
            errors.append(f"no hard constraints apply to scenario {sid}")
    if tool in SOLVER_TOOLS:
        try:
            result["base_arguments"] = _normalize_arguments(tool, result.get("base_arguments") or {})[0]
            for scenario in scenarios:
                scenario["arguments"] = _normalize_arguments(tool, scenario.get("arguments") or {})[0]
            for row in rows:
                row["arguments"] = _normalize_arguments(tool, row.get("arguments") or {})[0]
                row.setdefault("tool",tool)
            for row in result.get("diagnostic_rows") or []:
                row["arguments"] = _normalize_arguments(tool, row["arguments"])[0]
                if row.get("scenario_id") not in scenario_ids: errors.append("diagnostic row has unknown scenario_id")
            if mode == "one_dimensional_search": result["variable"] = _normalize_variable(tool,result.get("variable"))
        except (ProgramToolError, AttributeError, TypeError, KeyError) as exc:
            errors.append(str(exc))
    if mode == "finite_candidates":
        seen = set()
        for row in rows:
            if not isinstance(row,dict): errors.append("row must be an object"); continue
            rid = row.get("id")
            if not rid or rid in seen: errors.append("row IDs must be unique and non-empty")
            seen.add(rid)
            if row.get("scenario_id", "design") not in scenario_ids: errors.append(f"row {rid} has unknown scenario_id")
            if result.get("task_family") == "d4_4a" and tool == "simulate_swro_system" and "multiplicity" not in row:
                errors.append(f"row {rid} must explicitly state multiplicity (1 for a single train)")
            try: _row_multiplicity(row, str(row.get("candidate_id")), str(result.get("task_family")))
            except ProgramToolError as exc: errors.append(str(exc))
    if errors:
        raise ProgramToolError("contract errors before simulation: " + "; ".join(dict.fromkeys(errors)))
    return result


class _Executor:
    def __init__(self, call: Callable[[str, dict[str, Any]], str], budget: int):
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 64:
            raise ProgramToolError("max_physical_calls must be 1..64")
        self.call = call
        self.budget = budget
        self.cache: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []

    def run(self, tool: str, arguments: dict[str, Any], *, row_id: str) -> dict[str, Any]:
        if tool not in SOLVER_TOOLS:
            raise ProgramToolError(f"unsupported solver tool: {tool}")
        arguments, ignored = _normalize_arguments(tool, arguments)
        signature = json.dumps([tool, arguments], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature in self.cache:
            cached = self.cache[signature]
            self.events.append({
                "field": "multifamily_executable_skill",
                "event_type": "cache_interaction",
                "tool_name": tool,
                "arguments": copy.deepcopy(arguments),
                "observation": copy.deepcopy(cached),
                "status": "success",
                "metadata": {"row_id": row_id, "physical_call": False, "ignored_task_fields": ignored},
            })
            return copy.deepcopy(cached)
        physical = sum(e.get("metadata", {}).get("physical_call") is True for e in self.events)
        if physical >= self.budget:
            raise ProgramToolError(f"physical call budget exhausted ({self.budget})")
        try:
            raw = self.call(tool, copy.deepcopy(arguments))
            result = json.loads(raw)
            if not isinstance(result, dict):
                result = {"error": "solver returned a non-object JSON result", "raw": result}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        status = "error" if result.get("error") else "success"
        event = {
            "field": "multifamily_executable_skill",
            "event_type": "tool_interaction",
            "tool_name": tool,
            "arguments": copy.deepcopy(arguments),
            "observation": copy.deepcopy(result),
            "raw_observation": copy.deepcopy(result),
            "status": status,
            "metadata": {"row_id": row_id, "physical_call": True, "ignored_task_fields": ignored},
        }
        self.events.append(event)
        if status != "success":
            raise ProgramToolError(f"{tool} failed for {row_id}: {result.get('error', result)}")
        self.cache[signature] = copy.deepcopy(result)
        return result


def _check(result: dict[str, Any], rules: list[dict[str, Any]], scenario_id: str,
           *, quantity_multiplier: float = 1.0) -> list[dict[str, Any]]:
    checks = []
    for rule in rules:
        if rule["scenario_ids"] is not None and scenario_id not in rule["scenario_ids"]:
            continue
        path, alias_scale = _METRIC_ALIASES.get(
            rule["path"], _METRIC_ALIAS_INDEX.get(_fold_name(rule["path"]), (rule["path"], 1.0))
        )
        raw = _field(result, path)
        normalized_unit = rule["unit"].casefold().replace("³", "3")
        unit_scale = 86400.0 if (
            path == "performance.product_flow_m3_s"
            and normalized_unit in {"m3/d", "m3/day"}
            and alias_scale == 1.0
        ) else 1.0
        aggregate_scale = quantity_multiplier if path in _SCALABLE_METRICS else 1.0
        value = _finite(raw, path) * rule["scale"] * alias_scale * unit_scale * aggregate_scale
        checks.append({
            "id": rule["id"], "path": path, "value": value,
            "op": rule["op"], "threshold": rule["threshold"], "unit": rule["unit"],
            "quantity_multiplier": aggregate_scale,
            "pass": bool(OPS[rule["op"]](value, rule["threshold"])),
        })
    return checks


def _evaluate_row(executor: _Executor, *, row_id: str, candidate_id: str,
                  scenario_id: str, tool: str, arguments: dict[str, Any],
                  decision_value: float | None, rules: list[dict[str, Any]],
                  quantity_multiplier: float = 1.0) -> dict[str, Any]:
    arguments = _augment_required_arguments(tool, arguments, rules)
    result = executor.run(tool, arguments, row_id=row_id)
    checks = _check(result, rules, scenario_id, quantity_multiplier=quantity_multiplier)
    row = {
        "row_id": row_id,
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "tool": tool,
        "arguments": copy.deepcopy(arguments),
        "decision_value": decision_value,
        "quantity_multiplier": quantity_multiplier,
        "checks": checks,
        "feasible": bool(checks) and all(item["pass"] for item in checks),
        "result": result,
    }
    executor.rows.append(row)
    return row


def _row_multiplicity(row: dict[str, Any], candidate_id: str, family: str) -> float:
    value = row.get("multiplicity", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProgramToolError("multiplicity must be an explicit positive integer; never infer it from a candidate name")
    return float(value)


def _finite_mode(spec: dict[str, Any], executor: _Executor) -> dict[str, Any]:
    rows = spec.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) > 64:
        raise ProgramToolError("finite_candidates requires 1..64 rows")
    rules = _normalize_constraints(spec.get("constraints"), field="constraints")
    base_arguments = spec.get("base_arguments") or {}
    scenarios = spec.get("scenarios") or []
    if not isinstance(base_arguments, dict) or not isinstance(scenarios, list):
        raise ProgramToolError("finite base_arguments/scenarios are invalid")
    scenario_arguments = {
        str(item.get("id")): item.get("arguments", {})
        for item in scenarios
        if isinstance(item, dict) and isinstance(item.get("arguments", {}), dict)
    }
    family = str(spec.get("task_family") or "")
    prune_failed = bool(spec.get("prune_failed_candidates", family == "d1_1b"))
    candidate_alive: dict[str, bool] = {}
    seen = set()
    evaluated = []
    skipped = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProgramToolError(f"rows[{index}] must be an object")
        row_id = str(row.get("id") or f"R{index + 1}")
        if row_id in seen:
            raise ProgramToolError(f"duplicate row id: {row_id}")
        seen.add(row_id)
        candidate_id = str(row.get("candidate_id") or row_id)
        if prune_failed and candidate_alive.get(candidate_id) is False:
            skipped.append({"row_id": row_id, "candidate_id": candidate_id, "reason": "prior mandatory row failed"})
            continue
        scenario_id = str(row.get("scenario_id") or "design")
        tool = str(row.get("tool") or spec.get("solver_tool") or "")
        row_arguments = row.get("arguments")
        if not isinstance(row_arguments, dict):
            raise ProgramToolError(f"{row_id}.arguments must be an object")
        arguments = _merge(
            base_arguments,
            scenario_arguments.get(scenario_id, {}),
            row_arguments,
        )
        decision_value = row.get("decision_value")
        if decision_value is not None:
            decision_value = _finite(decision_value, f"{row_id}.decision_value")
        multiplicity = _row_multiplicity(row, candidate_id, family)
        evaluated_row = _evaluate_row(
            executor, row_id=row_id, candidate_id=candidate_id,
            scenario_id=scenario_id, tool=tool, arguments=arguments,
            decision_value=decision_value, rules=rules,
            quantity_multiplier=multiplicity,
        )
        if family == "d4_4a":
            aggregates = {}
            for metric in sorted(_SCALABLE_METRICS):
                try:
                    aggregates[metric] = _finite(_field(evaluated_row["result"], metric), metric) * multiplicity
                except ProgramToolError:
                    pass
            evaluated_row["aggregate_metrics"] = aggregates
            if "costing.total_capital_cost_usd" in aggregates:
                evaluated_row["decision_value"] = aggregates["costing.total_capital_cost_usd"]
        evaluated.append(evaluated_row)
        candidate_alive[candidate_id] = candidate_alive.get(candidate_id, True) and evaluated_row["feasible"]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in evaluated:
        by_candidate.setdefault(row["candidate_id"], []).append(row)
    candidates = []
    for candidate_id, candidate_rows in by_candidate.items():
        candidates.append({
            "candidate_id": candidate_id,
            "feasible": all(row["feasible"] for row in candidate_rows) and (
                family != "d1_1b" or set(scenario_arguments).issubset({r["scenario_id"] for r in candidate_rows})),
            "missing_scenarios": sorted(set(scenario_arguments) - {r["scenario_id"] for r in candidate_rows}) if family == "d1_1b" else [],
            "row_ids": [row["row_id"] for row in candidate_rows],
            "failed_checks": [
                {"row_id": row["row_id"], **check}
                for row in candidate_rows for check in row["checks"] if not check["pass"]
            ],
        })
    objective = spec.get("objective") or {}
    direction = objective.get("direction")
    selected = []
    if direction in {"min", "max"}:
        feasible = [c for c in candidates if c["feasible"]]
        values = {}
        for candidate in feasible:
            row_values = [r["decision_value"] for r in by_candidate[candidate["candidate_id"]]
                          if r["decision_value"] is not None]
            if row_values:
                values[candidate["candidate_id"]] = row_values[0]
        if values:
            best = (min if direction == "min" else max)(values.values())
            selected = [key for key, value in values.items() if value == best]
    return {
        "mode": "finite_candidates", "rows": evaluated, "skipped_rows": skipped,
        "candidates": candidates, "selected": selected,
    }


def _grid_index(value: float, lower: float, resolution: float, count: int) -> int:
    return max(0, min(count - 1, int(round((value - lower) / resolution))))


def _search_mode(spec: dict[str, Any], executor: _Executor) -> dict[str, Any]:
    tool = str(spec.get("solver_tool") or "")
    if tool not in SOLVER_TOOLS:
        raise ProgramToolError("one_dimensional_search requires a supported solver_tool")
    base = spec.get("base_arguments") or {}
    scenarios = spec.get("scenarios") or []
    if not isinstance(base, dict) or not isinstance(scenarios, list):
        raise ProgramToolError("invalid base_arguments/scenarios")
    variable = _normalize_variable(tool, spec.get("variable"))
    if not scenarios:
        row_arguments = [
            row.get("arguments", {}) for row in (spec.get("rows") or [])
            if isinstance(row, dict) and isinstance(row.get("arguments", {}), dict)
        ]
        consensus = {}
        if row_arguments:
            for key, value in row_arguments[0].items():
                if all(key in item and item[key] == value for item in row_arguments[1:]):
                    if _fold_name(key) != _fold_name(str(spec.get("variable", {}).get("path", ""))):
                        consensus[key] = copy.deepcopy(value)
        scenarios = [{"id": "design", "arguments": consensus}]
    lower = _finite(variable.get("lower"), "variable.lower")
    upper = _finite(variable.get("upper"), "variable.upper")
    resolution = _finite(variable.get("resolution"), "variable.resolution")
    if resolution <= 0 or upper < lower:
        raise ProgramToolError("invalid variable bounds/resolution")
    boundary_mode = variable.get("boundary", "both")
    if boundary_mode not in {"lower", "upper", "both"}:
        raise ProgramToolError("variable.boundary must be lower, upper, or both")
    if variable.get("search_each_scenario") and len(scenarios) > 1:
        intervals_by_scenario = {}
        rows = []
        for index, scenario in enumerate(scenarios):
            child_variable = dict(variable)
            child_variable["search_each_scenario"] = False
            scenario_id = str(scenario.get("id") or f"S{index + 1}")
            try:
                child = _search_mode(
                    {**spec, "scenarios": [scenario], "variable": child_variable}, executor)
            except ProgramToolError as exc:
                return {"mode":"one_dimensional_search", "variable":variable,
                        "intervals_by_scenario":intervals_by_scenario, "rows":list(executor.rows),
                        "error":str(exc), "incomplete_scenarios":[str(s.get("id")) for s in scenarios[index:]]}
            intervals_by_scenario[scenario_id] = child.get("intervals", {})
            rows.extend(child.get("rows", []))
        return {
            "mode": "one_dimensional_search",
            "variable": variable,
            "intervals_by_scenario": intervals_by_scenario,
            "rows": rows,
        }
    count = int(math.floor((upper - lower) / resolution + 1e-7)) + 1
    if count < 1 or count > 100000:
        raise ProgramToolError("variable lattice is invalid or too large")
    rules = _normalize_constraints(spec.get("constraints"), field="constraints")
    recommended_raw = spec.get("recommended_constraints")
    recommended = _normalize_constraints(recommended_raw, field="recommended_constraints") if recommended_raw else []
    normalized_scenarios = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict) or not isinstance(scenario.get("arguments", {}), dict):
            raise ProgramToolError(f"scenarios[{index}] is invalid")
        normalized_scenarios.append({"id": str(scenario.get("id") or f"S{index+1}"), "arguments": scenario.get("arguments", {})})

    cache: dict[tuple[int, str], dict[str, Any]] = {}
    point_summary: dict[int, dict[str, Any]] = {}

    def value_at(index: int) -> float:
        return float(Decimal(str(lower)) + index * Decimal(str(resolution)))

    controlling_scenario = None

    def evaluate(index: int) -> dict[str, Any]:
        nonlocal controlling_scenario
        index = max(0, min(count - 1, index))
        if index in point_summary:
            return point_summary[index]
        value = value_at(index)
        rows = []
        scenario_order = sorted(normalized_scenarios, key=lambda s: s["id"] != controlling_scenario)
        for scenario in scenario_order:
            arguments = _merge(base, scenario["arguments"])
            _set_path(arguments, variable["path"], value)
            arguments = _augment_required_arguments(tool, arguments, rules + recommended)
            row_id = f"X{index}-{scenario['id']}"
            result = executor.run(tool, arguments, row_id=row_id)
            normal_checks = _check(result, rules, scenario["id"])
            margin_checks = _check(result, recommended, scenario["id"]) if recommended else []
            row = {
                "row_id": row_id, "candidate_id": f"X{index}", "scenario_id": scenario["id"],
                "tool": tool, "arguments": arguments, "decision_value": value,
                "checks": normal_checks, "recommended_checks": margin_checks,
                "feasible": bool(normal_checks) and all(c["pass"] for c in normal_checks),
                "recommended_feasible": bool(normal_checks) and all(c["pass"] for c in normal_checks) and bool(margin_checks) and all(c["pass"] for c in margin_checks),
                "result": result,
            }
            executor.rows.append(row)
            cache[(index, scenario["id"])] = row
            rows.append(row)
            if not row["feasible"]:
                # One failed mandatory scenario rejects this common candidate.
                # A passing boundary still requires every scenario to pass.
                controlling_scenario = scenario["id"]
                break
        summary = {
            "index": index, "value": value, "rows": rows,
            "feasible": all(r["feasible"] for r in rows),
            "recommended_feasible": bool(recommended) and all(r["recommended_feasible"] for r in rows),
        }
        point_summary[index] = summary
        return summary

    probes = variable.get("probe_values") or []
    if not isinstance(probes, list):
        raise ProgramToolError("variable.probe_values must be an array")
    probe_indexes = [_grid_index(_finite(v, "probe value"), lower, resolution, count) for v in probes]
    probe_indexes += [0, count - 1, count // 2, count // 4, (3 * count) // 4]
    remaining = executor.budget - sum(e.get("metadata", {}).get("physical_call") is True for e in executor.events)
    if count * len(normalized_scenarios) <= remaining:
        probe_indexes += list(range(count))
    ordered = []
    for index in probe_indexes:
        if index not in ordered:
            ordered.append(index)
    feasible_indexes = []
    for index in ordered:
        try:
            if evaluate(index)["feasible"]:
                feasible_indexes.append(index)
                break
        except ProgramToolError as exc:
            if "budget exhausted" in str(exc):
                break
            raise

    # Two failing endpoints can enclose a narrow feasible window. Refine gaps
    # where individual constraint pass/fail states change, not only gaps with
    # an already-feasible endpoint. Never claim exhaustive infeasibility.
    def check_states(point):
        return {(r["scenario_id"], c["id"]): c["pass"] for r in point["rows"] for c in r["checks"]}

    while not feasible_indexes:
        indexes = sorted(point_summary)
        gaps = []
        for left, right in zip(indexes, indexes[1:]):
            if right-left > 1 and check_states(point_summary[left]) != check_states(point_summary[right]):
                gaps.append((right-left, left, right))
        remaining = executor.budget - sum(e.get("metadata", {}).get("physical_call") is True for e in executor.events)
        if not gaps or remaining < len(normalized_scenarios): break
        _, left, right = max(gaps)
        mid = (left + right)//2
        if evaluate(mid)["feasible"]: feasible_indexes.append(mid)

    def boundary(seed: int, *, low: bool, key: str) -> int:
        left, right = (0, seed) if low else (seed, count - 1)
        # Establish the endpoint status before claiming an adjacent boundary.
        endpoint = left if low else right
        if evaluate(endpoint)[key]:
            return endpoint
        while right - left > 1:
            mid = (left + right) // 2
            passes = bool(evaluate(mid)[key])
            if low:
                if passes:
                    right = mid
                else:
                    left = mid
            else:
                if passes:
                    left = mid
                else:
                    right = mid
        return right if low else left

    intervals: dict[str, Any] = {}
    if feasible_indexes:
        seed = feasible_indexes[0]
        lo = (
            boundary(seed, low=True, key="feasible")
            if boundary_mode in {"lower", "both"} and seed > 0
            else seed if boundary_mode == "upper" else 0
        )
        hi = (
            boundary(seed, low=False, key="feasible")
            if boundary_mode in {"upper", "both"} and seed < count - 1
            else seed if boundary_mode == "lower" else count - 1
        )
        if boundary_mode in {"lower", "both"}:
            evaluate(max(0, lo - 1)); evaluate(lo)
        if boundary_mode in {"upper", "both"}:
            evaluate(hi); evaluate(min(count - 1, hi + 1))
        intervals["basic"] = {
            "lower": value_at(lo) if boundary_mode in {"lower", "both"} else None,
            "upper": value_at(hi) if boundary_mode in {"upper", "both"} else None,
        }
        if recommended:
            recommended_seed = next((i for i, p in sorted(point_summary.items()) if p["recommended_feasible"]), None)
            if recommended_seed is None:
                for index in (lo, hi, (lo + hi) // 2):
                    if evaluate(index)["recommended_feasible"]:
                        recommended_seed = index
                        break
            if recommended_seed is not None:
                rlo = (
                    boundary(recommended_seed, low=True, key="recommended_feasible")
                    if boundary_mode in {"lower", "both"} and recommended_seed > 0
                    else recommended_seed if boundary_mode == "upper" else 0
                )
                rhi = (
                    boundary(recommended_seed, low=False, key="recommended_feasible")
                    if boundary_mode in {"upper", "both"} and recommended_seed < count - 1
                    else recommended_seed if boundary_mode == "lower" else count - 1
                )
                if boundary_mode in {"lower", "both"}:
                    evaluate(max(0, rlo - 1)); evaluate(rlo)
                if boundary_mode in {"upper", "both"}:
                    evaluate(rhi); evaluate(min(count - 1, rhi + 1))
                intervals["recommended"] = {
                    "lower": value_at(rlo) if boundary_mode in {"lower", "both"} else None,
                    "upper": value_at(rhi) if boundary_mode in {"upper", "both"} else None,
                }
    rows = [row for _, point in sorted(point_summary.items()) for row in point["rows"]]
    return {"mode": "one_dimensional_search", "variable": variable, "intervals": intervals, "rows": rows,
            "search_scope": "sampled_grid_and_adjacent_boundaries; global interval requires a connected feasible region",
            "complete_grid": len(point_summary) == count}


def _report(result: dict[str, Any], family: str, physical_calls: int, cache_hits: int) -> str:
    lines = [
        "# Program-verified engineering evidence",
        "",
        f"Task family: {family}. Physical solver calls: {physical_calls}. Cache hits: {cache_hits}.",
        "",
    ]
    if result.get("intervals"):
        lines += ["## Observed grid boundaries (within supplied search range)", "", "```json", json.dumps(result["intervals"], ensure_ascii=False, indent=2), "```", ""]
    if result.get("intervals_by_scenario"):
        lines += ["## Observed grid boundaries by scenario", "", "```json", json.dumps(result["intervals_by_scenario"], ensure_ascii=False, indent=2), "```", ""]
    if result.get("selected"):
        lines += [f"Selected feasible candidate(s): {', '.join(result['selected'])}.", ""]
    lines += [
        "| Row | Candidate | Scenario | Decision | Feasible | Verified checks |",
        "|---|---|---|---:|---|---|",
    ]
    for row in result.get("rows", []):
        checks = [
            f"{c['id']}={c['value']:.6g}{c.get('unit', '')} {c['op']} {c['threshold']:.6g}: {'pass' if c['pass'] else 'FAIL'}"
            for c in row.get("checks", [])
        ]
        checks += [
            f"margin:{c['id']}={c['value']:.6g}{c.get('unit', '')} {c['op']} {c['threshold']:.6g}: {'pass' if c['pass'] else 'FAIL'}"
            for c in row.get("recommended_checks", [])
        ]
        lines.append(
            f"| {row['row_id']} | {row['candidate_id']} | {row['scenario_id']} | "
            f"{row.get('decision_value', '')} | {'yes' if row.get('feasible') else 'no'} | {'; '.join(checks) or '-'} |"
        )
    if result.get("skipped_rows"):
        lines += ["", f"Pruned rows: {json.dumps(result['skipped_rows'], ensure_ascii=False)}"]
    lines += ["", "## Recorded numerical outputs", "", "```json",
              json.dumps([{k: r[k] for k in ("row_id", "arguments", "result", "aggregate_metrics") if k in r}
                          for r in result.get("rows", [])], ensure_ascii=False), "```"]
    if result.get("search_scope"):
        lines += ["", result["search_scope"]]

    lines += ["", "Every numeric check above is computed from the recorded solver result. Failed or missing solver results are not treated as engineering evidence."]
    return "\n".join(lines) + "\n"


def execute_program(arguments: dict[str, Any], call: Callable[[str, dict[str, Any]], str]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ProgramToolError("program arguments must be an object")
    executor = None
    family = str(arguments.get("task_family") or "")
    try:
        context = arguments.get("_public_context")
        if context:
            from .program_contract import bind_public_inputs
            arguments = bind_public_inputs(arguments, context)
            family = arguments["task_family"]
        arguments = validate_contract(arguments)
        mode = arguments["mode"]
        executor = _Executor(call, arguments.get("max_physical_calls", 32))
        for i, diagnostic in enumerate(arguments.get("diagnostic_rows") or []):
            _evaluate_row(executor, row_id=f"diagnostic-{i}", candidate_id="diagnostic",
                          scenario_id=diagnostic["scenario_id"], tool=arguments["solver_tool"],
                          arguments=_merge(arguments.get("base_arguments", {}), diagnostic["arguments"]),
                          decision_value=None, rules=_normalize_constraints(arguments["constraints"], field="constraints"))
        result = _finite_mode(arguments, executor) if mode == "finite_candidates" else _search_mode(arguments, executor)
    except (ProgramToolError, TypeError, ValueError, KeyError, AttributeError) as exc:
        events = executor.events if executor else []
        return {"schema_version": "1.0", "workflow_version": "multifamily-executable-skill@0.2.0",
                "task_family": family, "error": f"ProgramToolError: {exc}",
                "contract_valid": executor is not None, "workflow_complete": False,
                "repair_hint": "Return one complete corrected tool call. Use numeric JSON arrays for constraints, canonical fields and matching scenario IDs. Do not guess missing engineering facts.",
                "rows": executor.rows if executor else [],
                "physical_tool_calls": sum(e.get("metadata", {}).get("physical_call") is True for e in events),
                "_observable_events": events}
    physical = sum(e.get("metadata", {}).get("physical_call") is True for e in executor.events)
    cache_hits = sum(e.get("event_type") == "cache_interaction" for e in executor.events)
    result["has_decision_evidence"] = bool(
        result.get("intervals", {}).get("basic")
        or any(v.get("basic") for v in result.get("intervals_by_scenario", {}).values())
        or any(c.get("feasible") for c in result.get("candidates", [])))
    result.update({
        "schema_version": "1.0",
        "workflow_version": "multifamily-executable-skill@0.2.0",
        "task_family": family,
        "contract_valid": True, "workflow_complete": not bool(result.get("error")),
        "input_binding": arguments.get("_input_binding"),
        "diagnostic_rows": [r for r in executor.rows if r["candidate_id"] == "diagnostic"],
        "physical_tool_calls": physical,
        "cache_hits": cache_hits,
        "verified_report": _report(result, family, physical, cache_hits),
        "_observable_events": executor.events,
    })
    return result
