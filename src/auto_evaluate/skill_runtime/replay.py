"""Offline fixture adapter. Deliberately not a simulator or a live WaterTAP client."""
from __future__ import annotations

from .contracts import ToolResult, ToolSpec, digest, exact_keys, json_copy, number, require


def _numeric_inputs(arguments: dict) -> None:
    for key, value in arguments.items():
        if key == "erd_type":
            require(value == "pressure_exchanger", "INVALID_ARGUMENT", key)
        elif key == "composition_mol_s":
            require(isinstance(value, dict) and bool(value), "INVALID_ARGUMENT", key)
            require(all(number(v) and v >= 0 for v in value.values()), "INVALID_ARGUMENT", key)
        elif key == "minerals":
            require(isinstance(value, list) and bool(value) and all(isinstance(v, str) and v for v in value),
                    "INVALID_ARGUMENT", key)
        else:
            require(number(value), "INVALID_ARGUMENT", key)
        if key == "water_recovery":
            require(0 <= value < 1, "INVALID_ARGUMENT", "water_recovery must be a fraction in [0,1)")


def fixture_tool_specs() -> dict[str, ToolSpec]:
    """Normalized fixture projection, NOT a verified live API response mapping."""
    ro_inputs = frozenset({"feed_flow_mass_kg_s", "feed_nacl_mass_frac", "feed_pressure_bar",
                           "feed_temperature_c", "membrane_area_m2", "A_comp", "B_comp",
                           "pressure_drop_bar"})
    plant_inputs = frozenset({"feed_flow_m3_s", "feed_tds_g_l", "p1_pressure_bar", "ro_area_m2",
                              "feed_temperature_c", "A_comp", "B_comp", "erd_type", "pxr_efficiency"})
    chemistry_inputs = frozenset({"composition_mol_s", "temperature_c", "pressure_bar", "ph",
                                  "water_recovery", "minerals"})
    return {
        "simulate_ro": ToolSpec("simulate_ro", ro_inputs,
            {"water_kg_s": "kg/s", "nacl_mg_l": "mg/L", "inlet_cp": "fraction",
             "water_recovery_pct": "%", "water_lmh": "LMH"}, validate_inputs=_numeric_inputs),
        "simulate_swro_system": ToolSpec("simulate_swro_system", plant_inputs,
            {"product_m3_s": "m3/s", "sec_kwh_m3": "kWh/m3", "lcow_usd2018_m3": "USD_2018/m3",
             "capex_usd2018": "USD_2018"}, validate_inputs=_numeric_inputs),
        "equilibrate_feed": ToolSpec("equilibrate_feed", chemistry_inputs,
            {"gypsum_si": "SI", "ph": "pH"}, validate_inputs=_numeric_inputs),
    }


class FixtureBackend:
    def __init__(self, fixture: dict):
        exact_keys(fixture, {"fixture_only", "source", "records"})
        require(fixture["fixture_only"] is True, "INVALID_FIXTURE", "Must explicitly be an offline fixture")
        require(isinstance(fixture["records"], list), "INVALID_FIXTURE", "Records must be a list")
        self._records = {}
        self.calls: list[dict] = []
        for row in fixture["records"]:
            exact_keys(row, {"tool", "arguments", "data"})
            key = (row["tool"], digest(row["arguments"]))
            require(key not in self._records, "DUPLICATE_FIXTURE", "Input fixtures must be unambiguous")
            self._records[key] = json_copy(row)
        self.source = json_copy(fixture["source"])

    def execute(self, tool: str, arguments: dict) -> ToolResult:
        self.calls.append({"tool": tool, "arguments": json_copy(arguments)})
        key = (tool, digest(arguments))
        require(key in self._records, "FIXTURE_NOT_FOUND", "No matching fixture; no numerical extrapolation")
        row = self._records[key]
        return ToolResult(data=json_copy(row["data"]), effective_inputs=json_copy(arguments),
                          version="offline-fixture-not-a-simulator", raw=json_copy(row))
