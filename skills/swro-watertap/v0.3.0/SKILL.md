---
name: swro-watertap
description: Solve SWRO WaterTAP pressure-selection and membrane-degradation boundary tasks with complete fixed arguments, mandatory derived limits, bounded simulations, unrounded constraint checks, and concise engineering reports.
---

# SWRO WaterTAP execution card

Solve the engineering problem; do not narrate an open-ended search. Never use memorized
case answers. Use only the question, retrieved domain material, and actual tool results.

## Hard stop rules

- Before the first simulation, write a private preflight record containing the decision
  variable, all fixed inputs, every constraint, and every requested output.
- Pass every question-specified input explicitly. A stated module pressure drop must appear
  as `pressure_drop_bar`; never accept its tool default.
- After the first call, compare the tool's echoed inputs with the preflight record. Repair
  any mismatch before continuing.
- Change only the declared decision variable between candidates.
- Use at most five `simulate_ro` calls unless a call fails. Do not use a calculator tool for
  one-step arithmetic.
- Check feasibility using unrounded values. A candidate failing any constraint is infeasible.
- After the last tool call, produce one compact final answer and stop. Never call another
  tool from inside the final answer and never repeat the conclusion.

## Tool argument lock

Map explicit question fields to these arguments:

`feed_flow_mass_kg_s`, `feed_nacl_mass_frac`, `feed_pressure_bar`,
`feed_temperature_c`, `membrane_area_m2`, `A_comp`, `B_comp`,
`permeate_pressure_bar`, `pressure_drop_bar`, `channel_height_m`,
`spacer_porosity`, `module_length_m`, `concentration_polarization`, and
`mass_transfer_coefficient`.

Use bar for tool pressures: MPa x 10 = bar. Preserve an explicitly supplied mass flow.
Prefer a tool-reported permeate volumetric flow. If only permeate mass flow is available,
declare one permeate-density assumption before calculation and keep it fixed; do not use
saline feed density as permeate density.

## Mandatory pre-calculations

Complete the applicable calculations before searching and include them in the final answer.

1. For each lower bound, margin = observed - limit.
2. For each upper bound, margin = limit - observed.
3. If an hourly permeate target and membrane area are supplied:
   `minimum_flux_LMH = target_m3_h x 1000 / area_m2`.
4. If daily production is constrained:
   `daily_m3 = permeate_m3_h x 24`.
5. If permeate salt and rejection are both constrained:
   `C_rejection_limit = C_feed x (1 - minimum_rejection/100)` and
   `C_active_limit = min(C_direct_limit, C_rejection_limit)`.

Do not begin a boundary search until the controlling target or active limit is explicit.

## Pressure-selection recipe

Use this exact sequence:

1. Select the strictest production target, including any stated operating margin, and
   calculate its minimum flux.
2. Simulate a reasonable lower pressure `P0`; report `Q0`, flux, and
   `deficit = Q_target - Q0`.
3. Simulate one higher pressure `P1` with identical fixed inputs; calculate
   `slope = (Q1 - Q0) / (P1 - P0)` with units.
4. Estimate `P_est = P0 + deficit/slope`. This is an estimate, not a simulation.
5. Directly simulate `P_est` and, if needed, one rounded neighboring operating point.
6. Choose the lowest directly verified point satisfying every constraint.

Do not replace the required deficit-and-slope calculation with repeated trial points. Report
the numerical boundary separately from a practical fixed setpoint. Explain why the lower
neighbor fails and why unnecessary extra pressure is not preferred.

## Membrane-degradation recipe

Use this exact sequence:

1. Compute `C_active_limit` from all quality constraints before the first simulation.
2. Simulate normal B and report its margin `C_active_limit - C_normal`.
3. Simulate every B value explicitly requested by the question.
4. Use one passing and one failing result as a bracket. Refine with no more than two
   additional B calls.
5. At the selected maximum feasible B, check permeate NaCl, rejection, recovery, and every
   other stated limit using unrounded values. If rejection is below its minimum, reject the
   point even when displayed rounding appears equal.
6. Report relative change `(B_max/B_normal - 1) x 100%`.

Identify the first active constraint from the unified constraint register. Do not search
against the direct NaCl limit alone when the rejection-derived NaCl limit is stricter.

## Final answer contract

Keep the final answer within these six short sections:

1. `Objective and fixed inputs` — include the decision variable and confirm the fixed
   pressure drop and modelling modes.
2. `Derived controlling target` — show the flux target or strict active-quality limit.
3. `Candidate results` — one compact table; distinguish direct simulations from estimates.
4. `Final constraint check` — value, limit, signed margin, and pass/fail for every constraint.
5. `Boundary and recommendation` — numerical boundary, directly verified feasible point,
   practical setpoint if requested, and one lower/higher comparison.
6. `Monitoring and limitations` — use directly measurable water-quality/flow/pressure
   indicators for alarms. Treat membrane A or B as latent inferred parameters, not direct
   online measurements.

Before answering, confirm every requested diagnostic output is present. If conductivity is
requested, state one conversion method and its uncertainty; otherwise do not introduce a
conductivity conversion. If any requested output is unavailable, say so explicitly.
