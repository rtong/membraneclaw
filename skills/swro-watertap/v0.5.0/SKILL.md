---
name: swro-watertap
description: Solve SWRO WaterTAP and scaling tasks by first compiling the question into a fixed execution contract, then using bounded tool calls, explicit argument locking, unrounded constraint checks, and a compact auditable final report.
---

# SWRO WaterTAP execution protocol

Solve the engineering task, not just the final number. The goal is a reproducible, auditable
answer that another engineer can verify from the stated inputs, tool calls, and constraint checks.
Do not use memorized benchmark answers. Use only the question, retrieved domain material, and
actual tool results.

## Core principle

Before reasoning about the answer, privately compile the question into one execution contract:

- task family;
- decision variable(s);
- fixed inputs that must never drift;
- constraints with direction and unit;
- mandatory output fields;
- explicit candidate values listed by the question, if any.

Do not start searching until this contract is complete.

## Hard stop rules

- Before the first tool call, make a private preflight record from the execution contract.
- Treat every question-stated input as mandatory. If an argument is stated in the question, pass it
  explicitly. Never accept a tool default for a stated parameter.
- Immediately after each tool call, compare echoed inputs against the preflight record. If any fixed
  input drifted or any stated argument is missing, repair it before continuing.
- Change only the declared decision variable between candidate evaluations.
- If the question provides explicit candidate values, evaluate those candidates before inventing
  extra ones.
- Use the minimum number of calls needed to prove the recommendation. Prefer one baseline, one
  bracket, one direct verification, and one recommendation check over long trial-and-error scans.
- Keep room for the final answer. Do not spend the entire budget on search.
- Check feasibility using unrounded values. A candidate failing any one constraint is infeasible.
- After the last tool call, produce one compact final answer and stop.

## Tool routing

- Use `simulate_ro` for operating-pressure, fixed-condition comparison, and membrane transport tasks.
- Use `analyze_ro_scaling` when RO recovery must feed concentrate chemistry or mineral SI analysis.
- Use `describe_ro_parameters` or `describe_reaktoro_options` only when an argument name, supported
  species, or required unit is genuinely uncertain.

## Argument contract

For every tool call, construct the full argument set from the question before calling.

### `simulate_ro`

Lock and pass:

- `feed_flow_mass_kg_s`
- `feed_nacl_mass_frac`
- `feed_pressure_bar`
- `feed_temperature_c`
- `membrane_area_m2`
- `A_comp`
- `B_comp`
- `permeate_pressure_bar`
- `pressure_drop_bar`
- `channel_height_m`
- `spacer_porosity`
- `module_length_m`
- `concentration_polarization`
- `mass_transfer_coefficient`

Use bar for tool pressures. Convert MPa to bar only once and keep the converted value fixed.
If the question gives both mass flow and volumetric flow, preserve the stated mass flow for the
tool call and use volumetric flow only for reporting or consistency checks.

### `analyze_ro_scaling`

Always pass:

- the full `composition_mol_s` dictionary, including `H2O` and every stated ion;
- `feed_pressure_bar`
- `feed_flow_mass_kg_s`
- `feed_temperature_c`
- `membrane_area_m2`
- `ph`
- `minerals`
- `acid_addition_mol_s`

If a quantity is fixed by the question, keep it fixed across all pressure or acid candidates.
Do not let omitted arguments fall back to defaults.

## Derived-limit protocol

Before searching, derive the controlling target explicitly.

- Lower-bound margin = observed - limit.
- Upper-bound margin = limit - observed.
- `minimum_flux_LMH = target_m3_h x 1000 / area_m2`.
- `daily_m3 = permeate_m3_h x 24`.
- `C_rejection_limit = C_feed x (1 - minimum_rejection / 100)`.
- `C_active_limit = min(C_direct_limit, C_rejection_limit)` whenever both direct concentration and
  rejection constraints exist.

Do not skip this step. Many wrong answers come from searching before the active limit is known.

## Search policy by task family

### Pressure selection

Use a bounded pressure-response protocol:

1. Identify the controlling production target, including any stated operating margin.
2. Run one reasonable low-pressure baseline and report deficit against the controlling target.
3. Run one higher pressure with all other inputs unchanged to obtain a local response slope.
4. Estimate the boundary pressure from deficit and local slope.
5. Directly verify the estimated boundary pressure.
6. If the question asks for an engineering recommendation beyond the theoretical minimum, compare one
   slightly higher practical setpoint and explain the tradeoff.

Do not replace the slope step with a long series of tiny pressure increments. Precision without a
complete conclusion is a failure.

### Membrane degradation or transport drift

Use a limit-consumption protocol:

1. Convert every quality constraint into the tightest active limit before running sensitivity tests.
2. Run the normal membrane state and compute the current safety margin to the active limit.
3. Evaluate every explicit B candidate named in the question with all other inputs fixed.
4. If a passing and failing bracket exists, use at most one interpolation estimate and one direct
   verification.
5. Report both the maximum allowable B and the relative increase from normal.
6. Give an online warning indicator based on measurable output quality, not on directly measuring
   latent A or B.

### Scaling and acid optimization

Use a compare-then-treat protocol:

1. Run the current-pressure, zero-acid baseline.
2. Run the proposed-pressure, zero-acid case with identical composition and operating inputs.
3. Identify the dominant scaling risk from SI sign and closeness to zero.
4. Evaluate every acid dose explicitly listed by the question before inventing new doses.
5. Distinguish the minimum basic-safe candidate from the minimum candidate that satisfies any stated
   extra margin, such as `Calcite SI <= -0.05`.
6. If the question only provides discrete candidates, recommend only among tested candidates and say
   so explicitly.

Acidification is mainly a carbonate-risk control. Do not describe it as the primary remedy for
sulfate scaling. Do not present a benchmark-defined SI margin as a universal industry standard.

## Post-call audit

After each call, privately record:

- which candidate was tested;
- whether echoed inputs match the fixed-input contract;
- key outputs needed by the question;
- signed margins to each relevant limit;
- whether the candidate is feasible, infeasible, or only partially acceptable;
- what one next call is justified, if any.

If the call does not supply a requested output directly, mark it as derived or unavailable. Do not
invent missing tool outputs.

## Final answer contract

The final answer must be short, complete, and auditable. Use these six sections:

1. `Objective and fixed inputs`
2. `Controlling targets and derived limits`
3. `Evaluated candidates`
4. `Final constraint check`
5. `Recommendation and why alternatives are rejected`
6. `Monitoring and limitations`

Within the final constraint check, list every required limit with:

- observed value;
- limit;
- signed margin;
- pass or fail.

If the question asks for both a mathematical minimum and an engineering recommendation, report both.
If a value is estimated rather than directly simulated, label it as estimated and keep the final
recommendation tied to a directly verified candidate whenever possible.

## Anti-patterns to avoid

- starting tool calls before compiling the question;
- omitting stated parameters because the tool has defaults;
- changing more than one variable at once without saying so;
- treating a benchmark-specific reporting convention as optional;
- stopping after a plausible conclusion without a full constraint table;
- using continuous micro-search where one bracket and one verification would suffice;
- converting NaCl concentration to conductivity when the question explicitly says not to;
- claiming an untested continuous optimum when only discrete candidates were evaluated.
