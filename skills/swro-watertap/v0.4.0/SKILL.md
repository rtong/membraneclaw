---
name: swro-watertap
description: Solve SWRO WaterTAP pressure-selection, membrane-degradation, and scaling/acid-optimization tasks with complete fixed arguments, mandatory derived limits, bounded tool calls, unrounded checks, and concise engineering reports.
---

# SWRO WaterTAP execution card

Solve the engineering problem; do not narrate an open-ended search. Never use memorized
case answers. Use only the question, retrieved domain material, and actual tool results.

## Hard stop rules

- Before the first call, create a private preflight record containing the decision variable,
  all fixed inputs, every constraint, and every requested output.
- Pass every question-specified input explicitly. Never accept a tool default for a stated
  parameter.
- After the first call, compare echoed inputs with the preflight record. Repair any mismatch.
- Change only the declared decision variable between candidates.
- Use no more than five model calls unless the question explicitly requires more candidate
  points or a call fails. Do not use a calculator tool for one-step arithmetic.
- Check feasibility using unrounded values. A candidate failing any constraint is infeasible.
- After the last tool call, produce one compact final answer and stop. Never call another
  tool from inside the final answer and never repeat the conclusion.

## Select the correct tool

- Use `simulate_ro` for operating pressure and membrane transport/degradation tasks.
- Use `analyze_ro_scaling` when RO recovery must feed a concentrate scaling analysis.
- Use `describe_ro_parameters` or `describe_reaktoro_options` only when an argument name,
  supported species, mineral, or unit is genuinely uncertain.

## `simulate_ro` argument lock

Map explicit fields to `feed_flow_mass_kg_s`, `feed_nacl_mass_frac`,
`feed_pressure_bar`, `feed_temperature_c`, `membrane_area_m2`, `A_comp`, `B_comp`,
`permeate_pressure_bar`, `pressure_drop_bar`, `channel_height_m`, `spacer_porosity`,
`module_length_m`, `concentration_polarization`, and `mass_transfer_coefficient`.

Use bar for tool pressures: MPa x 10 = bar. A stated module pressure drop must appear as
`pressure_drop_bar`. Preserve an explicitly supplied mass flow. Prefer tool-reported
permeate volumetric flow. If only permeate mass flow exists, declare one permeate-density
assumption and keep it fixed; never reuse saline feed density as permeate density.

## Mandatory pre-calculations

Complete the applicable calculations before searching and include them in the final answer:

1. Lower-bound margin = observed - limit.
2. Upper-bound margin = limit - observed.
3. `minimum_flux_LMH = target_m3_h x 1000 / area_m2`.
4. `daily_m3 = permeate_m3_h x 24`.
5. `C_rejection_limit = C_feed x (1 - minimum_rejection/100)` and
   `C_active_limit = min(C_direct_limit, C_rejection_limit)`.

Do not begin a boundary search until the controlling target or active limit is explicit.

## Pressure-selection recipe

1. Select the strictest production target, including a stated operating margin, and
   calculate minimum flux.
2. Simulate a reasonable lower pressure `P0`; report `Q0`, flux, and
   `deficit = Q_target - Q0`.
3. Simulate one higher pressure `P1` with identical fixed inputs; calculate
   `slope = (Q1 - Q0) / (P1 - P0)` with units.
4. Estimate `P_est = P0 + deficit/slope`; label it as an estimate.
5. Directly simulate `P_est` and, if needed, one neighboring operating point.
6. Choose the lowest directly verified point satisfying every constraint.

Do not replace the deficit-and-slope calculation with repeated trial points. Separate the
numerical boundary, directly verified feasible point, and practical fixed setpoint.

## Membrane-degradation recipe

1. Compute `C_active_limit` from all quality constraints before the first simulation.
2. Simulate normal B and report `C_active_limit - C_normal`.
3. Simulate every B value explicitly requested by the question.
4. Use one passing and one failing result as a bracket; use no more than two refinements.
5. At the selected B, check permeate NaCl, rejection, recovery, and every other limit using
   unrounded values. Reject it if any limit fails even when rounded display appears equal.
6. Report `(B_max/B_normal - 1) x 100%`.

Never search against direct NaCl alone when the rejection-derived limit is stricter.

## Scaling and acid-optimization recipe

For `analyze_ro_scaling`, explicitly pass:

- the complete `composition_mol_s` dictionary, including H2O and every stated ion;
- `feed_pressure_bar`, `membrane_area_m2`, `feed_flow_mass_kg_s`,
  `feed_temperature_c`, `ph`, `minerals`, and `acid_addition_mol_s`.

Use this sequence:

1. Call the current-pressure, zero-added-acid baseline. Record recovery, water flux,
   concentration factor, concentrate pH, and every requested mineral SI.
2. Call the proposed-pressure, zero-added-acid case with all other inputs identical.
3. Compare pressure cases and identify the primary risk from SI: SI < 0 is undersaturated,
   SI = 0 is saturated, and SI > 0 is thermodynamically able to precipitate.
4. At the proposed pressure, call every acid dose explicitly listed by the question while
   keeping composition and operating inputs fixed.
5. Select the lowest tested candidate satisfying all basic SI limits. Separately select the
   lowest tested candidate satisfying any stated safety-margin limit.
6. State that these are minima among tested candidates, not a continuous optimum, unless a
   bracketed continuous search was explicitly requested and performed.

Acidification strongly affects carbonate-scale risk such as Calcite but is not a primary
control for sulfate minerals such as Gypsum. Do not describe SI as a deposition rate, and do
not present a benchmark-defined SI margin as a universal industry standard.

## Final answer contract

Use six short sections:

1. `Objective and fixed inputs` — decision variable, full composition when applicable,
   fixed pressure drop or pH, and modelling modes.
2. `Derived controlling target` — flux target, active-quality limit, or SI criteria.
3. `Candidate results` — one compact table; distinguish direct calls from estimates.
4. `Final constraint check` — value, limit, signed margin, and pass/fail for every limit.
5. `Boundary and recommendation` — basic minimum, safety-margin recommendation, and why
   lower/higher alternatives are rejected.
6. `Monitoring and limitations` — use measurable indicators for alarms; treat membrane A/B
   and model SI as inferred/model indicators unless directly measured.

Before answering, confirm every requested output is present. If conductivity is requested,
state one conversion method and uncertainty; otherwise do not introduce it. Mark unavailable
outputs explicitly.
