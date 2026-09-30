"""Public-input-only family guidance and lossless table input extraction."""
import copy
import re

FAMILY_GUIDANCE = {
    "d1_1a": "Search membrane_area_m2, both boundaries, one common area across all scenarios (search_each_scenario=false). Do not assume larger area is safer. Search beyond the suggested starting area if no failing upper endpoint is observed. Recommend the robust interval midpoint on the stated grid, resolving ties downward; do not invent an area constraint.",
    "d1_1b": "Finite candidates. Include every candidate at every mandatory scenario in stage order: nominal, salinity, temperature checks. The program prunes a candidate only after an observed hard failure. Do not predict which membrane survives. No invented numeric objective is needed if the task requests screening.",
    "d1_1c": "Search feed_pressure_bar, lower boundary independently for each scenario (search_each_scenario=true). Keep the public membrane area and A/B unchanged. Explain the public pump-rating margin and equipment rounding after getting the scenario minima.",
    "d2_2a": "Search pressure windows independently at the public reduced flows. Include the unchanged-pressure baseline in probe_values. Preserve every public quality constraint. An empty sampled window is not an infeasibility proof. Additional feed restoration requires separate evidence, not invented feasible points.",
    "d2_2b": "Use the public priority between pressure-only, flow-only, and coordinated operation. Search the selected control variable on the stated grid; include baseline and alternative controls as explicit diagnostic_rows. Hold membrane properties and temperature fixed. Do not claim an untested alternative fails.",
    "d3_3a": "Use the public named decision variable, usually A_comp or membrane area, with simulate_swro_system. Values and grid must be in physical units (e.g. 0.01e-12, not 0.01 for A_comp). Put basic capacity in constraints; stricter capacity in recommended_constraints, NOT a second mandatory scenario. Give adjacent evidence for each threshold.",
    "d3_3b": "Read the actual public question: search its named equipment variable (e.g. pxr_efficiency), which can differ from the title. Match all measured sensor windows with separate lower/upper inequalities. Set explicit scenario IDs. Include the public maintenance reference as a probe. Keep active RO area and hydraulics fixed.",
    "d4_4a": "Finite design alternatives. For parallel identical trains explicitly provide integer multiplicity on each row; never put train count in decision_value as a proxy for CAPEX. Use total product constraints and compare aggregate CAPEX. Other procurement costs require explicit public evidence.",
    "d5_5a": "For a recovery limit use one_dimensional_search on water_recovery, upper boundary, resolution as a fraction (0.1 percentage point = 0.001). Include all mineral inequalities in constraints and safety margin in recommended_constraints. Copy composition including H2O, fixed pH and doses. Two coarse probes alone cannot prove a boundary.",
    "d5_5b": "Use the treatment variable and criterion in the public question. For minimum acid dose search acid_addition_mol_s, lower boundary. Copy fixed water_recovery, feed ph, H2O and all ions. Check all specified mineral SI and minimum pH; explain non-numeric route exclusions separately.",
}

# Labels come from input tables, never answer/rubric fields. Only unambiguous
# numeric cells are copied; repeated contradictory values are left to the model.
LABELS = {
    "feed mass flow": "feed_flow_mass_kg_s", "feed pressure": "feed_pressure_bar",
    "effective membrane area": "membrane_area_m2", "membrane area": "membrane_area_m2",
    "active membrane area": "membrane_area_m2", "a_comp": "A_comp", "b_comp": "B_comp",
    "permeate pressure": "permeate_pressure_bar", "permeate-side pressure": "permeate_pressure_bar",
    "pressure drop": "pressure_drop_bar", "module pressure drop": "pressure_drop_bar",
    "channel height": "channel_height_m", "spacer porosity": "spacer_porosity",
    "module length": "module_length_m", "equivalent channel length": "module_length_m",
    "nacl mass fraction": "feed_nacl_mass_frac", "temperature": "feed_temperature_c",
    "feed flow": "feed_flow_m3_s", "tds": "feed_tds_g_L", "feed tds": "feed_tds_g_L",
    "tss": "feed_tss_g_L", "feed tss": "feed_tss_g_L", "active ro area": "ro_area_m2",
    "p1 outlet pressure": "p1_pressure_bar", "p1 pressure": "p1_pressure_bar",
    "p1 efficiency": "p1_efficiency", "p2 efficiency": "p2_efficiency",
    "px efficiency": "pxr_efficiency", "feed ph": "ph", "water recovery": "water_recovery",
    "pressure": "pressure_bar",
}
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
SPECIES = {"H2O", "Na", "K", "Ca", "Mg", "Ba", "Sr", "Li", "Cl", "SO4", "HCO3", "CO3", "F", "Br", "B", "SiO2"}

def public_defaults(question, tool):
    from .program_tool import _ALLOWED_ARGUMENTS
    found = {}; conflicting = set(); species = {}
    for line in question.splitlines():
        cells = [c.strip() for c in line.split("|")]
        for i in (0, 1):
            if len(cells) <= i + 1 or not NUMBER.fullmatch(cells[i+1]):
                continue
            label = cells[i]; value = float(cells[i+1])
            if label in SPECIES and tool in {"equilibrate_feed", "analyze_ro_scaling"}:
                species[label] = value
                continue
            key = LABELS.get(label.casefold())
            if tool == "simulate_swro_system" and key == "membrane_area_m2": key = "ro_area_m2"
            if tool == "equilibrate_feed" and key == "feed_temperature_c": key = "temperature_c"
            if key not in _ALLOWED_ARGUMENTS.get(tool, set()): continue
            if key in found and found[key] != value: conflicting.add(key)
            found[key] = value
    for key in conflicting: found.pop(key, None)
    if species: found["composition_mol_s"] = species
    return found

def family_prompt(family):
    if family not in FAMILY_GUIDANCE: raise ValueError("Unknown program family")
    return ("\n[PUBLIC FAMILY CONTRACT]\nUse task_family=" + family + ". " + FAMILY_GUIDANCE[family]
            + "\nUse canonical parameter names from the tool schema. constraints must be a JSON array of objects; numerical thresholds must be evaluated numbers, never strings or arithmetic expressions. Missing information is an error, not permission to invent it. Explicitly provide all scenario IDs and fixed inputs.\n")

def bind_public_inputs(spec, context):
    from .program_tool import _normalize_arguments, _merge, ProgramToolError
    result = copy.deepcopy(spec)
    family = context["task_family"]
    if family not in FAMILY_GUIDANCE: raise ProgramToolError("Unknown public task family")
    result["task_family"] = family
    source = context["question_prompt"]
    tool = result.get("solver_tool") or next((r.get("tool") for r in result.get("rows",[]) if isinstance(r,dict)), None)
    if tool:
        defaults = public_defaults(source, tool)
        base, _ = _normalize_arguments(tool, result.get("base_arguments") or {})
        result["base_arguments"] = _merge(defaults, base)
    else: defaults = {}
    result["_input_binding"] = {"source": "public_question_tables", "task_family": family, "table_defaults": defaults}
    return result
