"""Validate public single-variable contracts; never reads case IDs or gold.

Scope: structured D3 and D5-5a declarations. This does not certify arbitrary
prose, physical models, or global monotonicity.
"""
from copy import deepcopy
import math
import re

VERSION = "public-search-contract@0.1.0"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"


class SearchContractError(ValueError):
    pass


def supported(family):
    return family in {"d3_3a", "d3_3b", "d5_5a"}


def table(question):
    text = (question.replace("≥", ">=").replace("≤", "<=").replace("−", "-")
            .replace("m³", "m3").replace("m²", "m2")
            .replace("×10⁻¹²", "e-12").replace("×10^-12", "e-12"))
    return [[s.strip() for s in line.strip().strip("|").split("|")] for line in text.splitlines() if "|" in line]


def declarations(rows, labels, minimum_column=0):
    return [row[i+1] for row in rows for i, value in enumerate(row[:-1])
            if value.casefold() in labels and (i == 0 or i >= minimum_column)
            and (i == 0 or row[i-1].casefold() not in labels)
            and not row[i+1].casefold().startswith("scope:")]


def unique(values, label):
    if not values or len(set(values)) != 1:
        raise SearchContractError(f"Missing or conflicting public {label}: {values}")
    return values[0]


PLANT_VARIABLES = {
    "A_comp": r"a_comp|membrane (?:water )?permeability(?: a)?|membrane a(?: \(e-12\))?",
    "ro_area_m2": r"ro_area_m2|(?:active )?(?:ro )?membrane area|active ro area",
    "feed_flow_m3_s": r"feed_flow_m3_s|(?:train )?feed flow",
    "feed_tds_g_L": r"feed_tds_g_l|(?:feed )?tds|feed salinity",
    "p1_pressure_bar": r"p1_pressure_bar|p1 (?:outlet )?pressure|high.pressure.pump outlet pressure",
    "p1_efficiency": r"p1_efficiency|hpp efficiency|p1 (?:pump )?efficiency",
    "p2_efficiency": r"p2_efficiency|p2 (?:pump )?efficiency",
    "pxr_efficiency": r"pxr_efficiency|(?:actual )?px efficiency",
}
CHEMISTRY_VARIABLES = {
    "water_recovery": r"water_recovery|water recovery",
    "Ba": r"ba(?: (?:mol/s|input))?|composition_mol_s.ba",
    "blend_fraction": r"blend_fraction|source-b fraction|(?:source-b )?blend fraction",
    "temperature_c": r"temperature_c|temperature(?: °c)?",
    "base_addition_mol_s": r"base_addition_mol_s|naoh(?: mol/s)?|base addition(?: mol/s)?",
    "feed_pressure_bar": r"feed_pressure_bar|feed pressure(?: bar)?",
}


def resolve_variable(text, family):
    name = re.sub(r"\s+only$", "", text.strip(), flags=re.I)
    aliases = PLANT_VARIABLES if family.startswith("d3_") else CHEMISTRY_VARIABLES
    matches = [key for key, pattern in aliases.items() if re.fullmatch(pattern, name, re.I)]
    if len(matches) != 1:
        raise SearchContractError(f"Unknown or ambiguous public decision variable: {text!r}")
    return matches[0]


def parse_resolution(text, variable):
    normalized = re.sub(r"^adjacent\s+", "", text.strip().lower())
    match = re.match(NUMBER, normalized, re.I)
    if not match:
        raise SearchContractError(f"Unrecognized resolution: {text!r}")
    value = float(match[0])
    unit = normalized[match.end():].strip().lstrip("-").strip()
    unit = re.sub(r"(?:^|\s+)(?:fail/pass|resolution|grid)$", "", unit).strip()
    accepted = {"A_comp": {"", "m/s/pa"}, "ro_area_m2": {"", "m2"},
                "feed_flow_m3_s": {"", "m3/s"}, "feed_tds_g_L": {"", "g/l"},
                "p1_pressure_bar": {"", "bar"}, "feed_pressure_bar": {"", "bar"},
                "temperature_c": {"", "°c"}, "Ba": {"", "mol/s"}, "base_addition_mol_s": {"", "mol/s"}}
    if variable in {"water_recovery", "blend_fraction", "p1_efficiency", "p2_efficiency", "pxr_efficiency"}:
        if unit in {"%", "percentage point", "percentage points", "percentage-point"}:
            value /= 100
        elif unit not in {"", "fraction"}:
            raise SearchContractError(f"Unsupported fractional resolution unit: {unit!r}")
    elif unit not in accepted[variable]:
        raise SearchContractError(f"Resolution unit {unit!r} does not match {variable}")
    if not math.isfinite(value) or value <= 0:
        raise SearchContractError("Resolution must be finite and positive")
    return value


def plant_limits(text):
    source = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    result = []
    for label, metric in [("Product", "product_m3_day"), ("SEC", "SEC"), ("P1", "P1_kW"), ("P2", "P2_kW"), ("brine", "brine_m3_s")]:
        for m in re.finditer(r"\b"+label+r"\s*(<=|>=|<|>)\s*("+NUMBER+r")", source, re.I):
            result.append((metric, m[1], float(m[2])))
        for m in re.finditer(r"\b"+label+r"\s*∈\[\s*("+NUMBER+r")\s*,\s*("+NUMBER+r")\s*\]", text, re.I):
            result.extend([(metric, ">=", float(m[1])), (metric, "<=", float(m[2]))])
    if not result or len(re.findall(r"<=|>=|(?<![<>])[<>](?!=)", text)) + 2*text.count("∈") != len(result):
        raise SearchContractError(f"Unsupported public plant criterion: {text!r}")
    return sorted(set(result))


def chemistry_limits(text, minerals):
    result = []
    common = re.search(r"all (?:listed |concentrate )?mineral(?:s)? SI\s*(<=|>=|<|>)\s*("+NUMBER+r")", text, re.I)
    for mineral in minerals:
        match = re.search(re.escape(mineral)+r"\s*(?:SI\s*)?(<=|>=|<|>)\s*("+NUMBER+r")", text, re.I)
        if match or common:
            m = match or common
            result.append(("SI:"+mineral, m[1], float(m[2])))
    remaining = text
    if common: remaining = remaining.replace(common[0], "")
    for mineral in minerals:
        remaining = re.sub(re.escape(mineral)+r"\s*(?:SI\s*)?(<=|>=|<|>)\s*("+NUMBER+r")", "", remaining, flags=re.I)
    if not result or re.search(r"[<>]", remaining):
        raise SearchContractError(f"Unsupported public mineral criterion: {text!r}")
    return sorted(set(result))


def search_requirements(question, family):
    if not supported(family):
        return None
    rows = table(question); plant = family.startswith("d3_")
    labels = {"decision variable", "px decision variable", "membrane a (e-12)"} if plant else {"variable"}
    names = declarations(rows, labels, 4 if plant else 2)
    variable = unique([resolve_variable(s, family) for s in names], "decision variable")
    resolutions = declarations(rows, {"resolution"})
    if plant:
        for row in rows:
            for i, value in enumerate(row):
                if value.casefold() in {"basic", "criterion 1", "recommended", "recommended margin", "criterion 2"} and i+2 < len(row):
                    if re.match(r"(?:Adjacent\s+)?"+NUMBER, row[i+2], re.I):
                        resolutions.append(row[i+2])
    step = unique([parse_resolution(s, variable) for s in resolutions], "search resolution")
    basic = unique(declarations(rows, {"basic", "criterion 1"} if plant else {"basic safety"}), "basic criterion")
    recommended = unique(declarations(rows, {"recommended", "recommended margin", "criterion 2"}), "recommended criterion")
    if plant:
        tool = "simulate_swro_system"
        explicit_tools = declarations(rows, {"primary tool"})
        if explicit_tools and unique(explicit_tools, "primary tool") != tool:
            raise SearchContractError("Whole-plant search requires simulate_swro_system")
        basic_rules, recommended_rules = plant_limits(basic), plant_limits(recommended)
        if "∈" in basic or "∈" in recommended:
            basic_rules, recommended_rules = sorted(set(basic_rules+recommended_rules)), []
    else:
        tool = unique(declarations(rows, {"primary tool"}), "primary tool")
        expected = "analyze_ro_scaling" if variable == "feed_pressure_bar" else "equilibrate_feed"
        if tool != expected:
            raise SearchContractError(f"Variable {variable} requires {expected}, not {tool}")
        minerals = unique(declarations(rows, {"mineral", "minerals", "control minerals"}), "mineral list").split(",")
        minerals = [s.strip() for s in minerals]
        basic_rules, recommended_rules = chemistry_limits(basic, minerals), chemistry_limits(recommended, minerals)
    return {"version": VERSION, "variable": variable, "tool": tool, "resolution": step,
            "basic_rules": basic_rules, "recommended_rules": recommended_rules,
            "public_declarations": {"variable": names, "resolution": resolutions, "basic": basic, "recommended": recommended}}


def check_search_job(job, requirements):
    expected_transform = {"Ba": "barium_chloride", "blend_fraction": "blend"}.get(requirements["variable"])
    if job.get("mode") != "search" or job.get("transform") != expected_transform:
        raise SearchContractError("Compiled search mode or coupling differs from the public variable")
    for key, expected in [("variable", requirements["variable"]), ("tool", requirements["tool"]), ("step", requirements["resolution"])]:
        if job.get(key) != expected:
            raise SearchContractError(f"Compiled {key} differs from public contract: {job.get(key)!r} != {expected!r}")
    for key, req in [("rules", "basic_rules"), ("recommended", "recommended_rules")]:
        actual = sorted(set((r["metric"], r["op"], r["threshold"]) for r in job.get(key, [])))
        if actual != sorted(tuple(r) for r in requirements[req]):
            raise SearchContractError(f"Compiled {key} differ from public criteria")
    if "fixed_inputs" in requirements and job["base"] != requirements["fixed_inputs"]:
        raise SearchContractError("Compiled fixed inputs changed after validation")
    if not job.get("base"):
        raise SearchContractError("Missing fixed-input record")
    return {**requirements, "fixed_inputs": deepcopy(job["base"]), "status": "checked",
            "checks": ["variable", "tool", "resolution", "basic_rules", "recommended_rules"],
            "limits": "Structured declarations only; omitted prose and physical behavior are not certified"}


def check_search_call(job, arguments, value):
    contract = job.get("input_contract")
    if not contract:
        return
    check_search_job(job, contract)
    if value is None or not math.isfinite(value):
        raise SearchContractError("Missing/nonfinite search point")
    index = (value-job["lower"])/job["step"]
    if value < job["lower"] or value > job["upper"] or not math.isclose(index, round(index), abs_tol=1e-6, rel_tol=0):
        raise SearchContractError("Search point is outside the compiled range or lattice")
    expected = deepcopy(job["base"]); source_b = expected.pop("_source_b", None)
    transform = job.get("transform")
    if transform == "blend":
        expected["composition_mol_s"] = {k: (1-value)*v+value*source_b.get(k, v) for k, v in expected["composition_mol_s"].items()}
    elif transform == "barium_chloride":
        expected["composition_mol_s"]["Ba"] = value
        expected["composition_mol_s"]["Cl"] += 2*value
    elif transform:
        raise SearchContractError(f"Unvalidated search transform: {transform}")
    else:
        expected[job["variable"]] = value
    if arguments != expected:
        raise SearchContractError("Physical inputs drifted from the fixed state or declared coupling")
