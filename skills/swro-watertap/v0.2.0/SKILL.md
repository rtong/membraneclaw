---
name: swro-watertap
description: Solve SWRO WaterTAP operating-point, membrane-degradation, and constraint-boundary tasks with explicit input locking, active-constraint derivation, verified tool calls, and concise engineering reporting.
---

# SWRO WaterTAP Skill

Use this skill when an SWRO question asks for a WaterTAP simulation, an operating
boundary, a membrane-degradation boundary, candidate comparison, or an engineering
recommendation. This skill defines a reusable method, not benchmark answers. Never
insert memorized case-specific final values.

## Non-negotiable completion gates

Do not write the final answer until every applicable gate passes:

1. `INPUT_LOCK`: every question-specified fixed input is present in a fixed-input ledger.
2. `ARGUMENT_LOCK`: every explicit input has been passed to `simulate_ro`; never rely on
   a tool default for a parameter supplied by the question.
3. `ECHO_CHECK`: compare the tool's echoed inputs with the ledger after the first call.
4. `LIMIT_LOCK`: compile direct and derived constraints before searching a boundary.
5. `BRACKET_LOCK`: retain at least one feasible and one infeasible candidate when a
   boundary is requested.
6. `FINAL_VERIFY`: directly simulate the reported boundary or conservative feasible
   point and check every constraint.
7. `REPORT_LOCK`: include all requested outputs once, then stop. Do not restart the
   analysis or repeat the final answer.

If a required input is missing, identify it instead of inventing a value. If a tool call
fails, repair the arguments or report the failure; never fabricate a simulation result.

## 1. Build the fixed-input ledger

Before any tool call, extract a compact ledger with these fields:

- task family, objective, and decision variable;
- fixed physical inputs and their units;
- direct constraints with direction (`>=` or `<=`);
- derived constraints and formulae;
- requested diagnostic outputs;
- modelling modes and conventions.

Map the common fields exactly as follows:

| Question field | `simulate_ro` argument | Tool unit |
|---|---|---|
| feed mass flow | `feed_flow_mass_kg_s` | kg/s |
| equivalent NaCl mass fraction | `feed_nacl_mass_frac` | fraction |
| feed pressure | `feed_pressure_bar` | bar |
| feed temperature | `feed_temperature_c` | degC |
| membrane area | `membrane_area_m2` | m2 |
| water permeability A | `A_comp` | m/s/Pa |
| salt permeability B | `B_comp` | m/s |
| permeate pressure | `permeate_pressure_bar` | bar |
| module pressure drop | `pressure_drop_bar` | bar |
| channel height | `channel_height_m` | m |
| spacer porosity | `spacer_porosity` | fraction |
| module length | `module_length_m` | m |
| concentration polarization mode | `concentration_polarization` | enum |
| mass-transfer mode | `mass_transfer_coefficient` | enum |

Explicitly check `pressure_drop_bar`; omission of a stated pressure drop is a failed
`ARGUMENT_LOCK`. Call `describe_ro_parameters` only if an argument name or unit remains
uncertain.

Normalize units before calling WaterTAP:

- MPa to bar: multiply by 10; bar to MPa: divide by 10.
- g/L to mg/L: multiply by 1000.
- percent to fraction: divide by 100.
- Convert volumetric and mass flow only when required, state the density, and do not
  replace an explicitly supplied mass flow with a derived value.

## 2. Compile constraints before simulation

Create one constraint register. For every constraint store quantity, limit, direction,
and source (`direct` or `derived`). If two limits constrain the same quantity, compute
both and enforce the stricter one.

Common derived constraints include:

- minimum average flux = required permeate volume flow / membrane area;
- minimum daily production = hourly permeate volume flow x 24;
- maximum permeate salt from rejection = feed salt concentration x
  (1 - minimum rejection as a fraction).

Do not substitute a related indicator for the actual controlled quantity. For example,
when both permeate NaCl and rejection are constrained, derive the rejection-implied NaCl
limit and use the smaller NaCl limit during the boundary search.

Use signed margins consistently:

- lower bound: `observed - lower_limit`;
- upper bound: `upper_limit - observed`.

Positive is feasible, zero is active, and negative is violated.

## 3. Run and verify the baseline

Call `simulate_ro` with the complete ledger. After the first successful call:

1. inspect the returned `inputs` block;
2. compare every fixed value and mode with the ledger;
3. correct any mismatch before continuing;
4. save the exact baseline arguments and required outputs.

Change only the declared decision variable in subsequent candidate calls. If any other
input changes, discard the comparison and rerun it.

## 4. Select the recipe

### Pressure-selection recipe

1. Compute the controlling production or flux target, including stated safety margins.
2. Simulate a physically reasonable lower pressure and calculate the target deficit.
3. Simulate a second pressure while holding the ledger fixed.
4. Compute the local response slope from the two direct results, including its units.
5. Estimate the pressure increment as `deficit / slope`; label this as an estimate.
6. Directly simulate the estimated candidate and one neighboring point.
7. Keep an infeasible/feasible bracket and refine only if the requested precision needs it.
8. Distinguish the numerical boundary, the lowest directly verified feasible point, and
   the practical operating setpoint.

Never stop merely because a basic production limit passes when the question also supplies
a stricter recommended or safety-margin target.

### Membrane-degradation recipe

1. Derive the strictest water-quality limit before changing the degradation parameter.
2. Simulate the normal parameter value with the complete ledger.
3. Report its signed quality margin against the strictest limit.
4. Simulate every explicitly requested degradation point.
5. Identify one passing and one failing value while all other inputs remain fixed.
6. Refine inside that bracket and identify which constraint becomes active first.
7. Report the maximum directly verified feasible value, relative change from normal, and
   the remaining margin at the reported precision.

### General constraint-check recipe

Simulate the stated point once with the complete ledger, compute every signed margin, and
state feasibility without claiming an unsearched optimum.

## 5. Control search cost and precision

Use enough calls to establish evidence, not to print meaningless digits:

- include all explicitly requested candidates;
- use at most two exploratory calls beyond them;
- normally use at most six refinement calls;
- stop when the bracket is narrower than the requested or engineering-relevant precision;
- report a conservative feasible endpoint, not a rounded value that crosses the limit.

If interpolation estimates a boundary, verify the selected point with a direct call. Do
not repeatedly announce provisional conclusions between calls.

## 6. Validate every result

For the final candidate, produce one constraint table with:

- simulated or transparently derived value;
- limit and direction;
- signed margin;
- pass/fail;
- evidence status: `direct simulation`, `derived from simulation`, or `unavailable`.

Also report every requested diagnostic output even when it is non-binding. Do not convert
NaCl to conductivity unless the question explicitly requires conductivity. When it does,
use one stated conversion method consistently and label its uncertainty.

## 7. Form an operationally valid recommendation

Separate four concepts:

- mathematical boundary;
- directly simulated feasible point;
- practical fixed operating recommendation;
- early-warning threshold.

Classify proposed monitoring signals:

- `online measured`: directly available plant measurement;
- `derived online`: calculated from online measurements;
- `latent inferred`: model parameter estimated by inversion or trend analysis;
- `model only`: not suitable as a direct alarm.

Do not recommend membrane A or B as a direct online alarm unless the system actually
measures it. Prefer measurable permeate salinity/NaCl, conductivity when specified,
rejection, flow, pressure, or their trends; describe A or B as latent inferred state.

## 8. Report once and stop

Use exactly these final sections:

1. `Task, objective, and fixed-input ledger`
2. `Derived targets and active limits`
3. `WaterTAP candidate results`
4. `Constraint margins and verification status`
5. `Boundary, recommendation, and online warning`
6. `Limitations`

Before emitting them, check that all fixed inputs stayed constant, all explicit tool
arguments were supplied, arithmetic is reproducible, requested outputs are present, and
estimates are not described as simulations. Emit one final answer only.
