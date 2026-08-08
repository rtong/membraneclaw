---
name: swro-watertap
description: Solve SWRO operating-point, membrane-degradation, and constraint-boundary tasks with WaterTAP using a verified, stepwise workflow.
---

# SWRO WaterTAP Skill

Use this skill for seawater reverse-osmosis questions that require a WaterTAP
`ReverseOsmosis0D` simulation, comparison of candidate operating points, search for
a feasible boundary, or an engineering recommendation.

This skill defines a method, not benchmark answers. Never recall or insert memorized
case-specific final values. Derive every result from the question and actual tool output.

## Completion contract

Do not present a final recommendation until all states below are complete:

1. `classify_task`
2. `extract_and_normalize`
3. `map_tool_arguments`
4. `run_baseline`
5. `analyze_margin`
6. `search_boundary`
7. `validate_constraints`
8. `form_engineering_judgment`
9. `self_check_and_report`

If a required input is missing, state which one is missing instead of silently using a
default. If the tool reports an error, correct the arguments or explain the failure; do
not replace a failed simulation with an invented number.

## 1. Classify the task

Choose one primary recipe:

- `pressure_selection`: minimize feed pressure while satisfying every constraint.
- `membrane_degradation`: maximize an ageing/degradation parameter such as salt
  permeability while satisfying water-quality and operating constraints.
- `general_constraint_check`: simulate a stated operating point and judge feasibility.

Name the selected recipe in the answer.

## 2. Extract and normalize

Create a compact input table containing:

- fixed physical inputs;
- decision variable and direction of optimization;
- direct constraints and whether each is an upper or lower bound;
- derived constraints, with their formula;
- requested outputs;
- units and any explicit modelling convention.

Normalize units before calling WaterTAP. Check especially:

- MPa to bar: multiply by 10;
- bar to MPa: divide by 10;
- m3/h to kg/s: use the stated density assumption, and disclose it;
- kg/s to m3/h: use the stated density assumption, and disclose it;
- g/L and mg/L: factor of 1000;
- percentages versus fractions;
- salt-permeability scientific notation.

Do not replace an explicitly supplied mass flow with a volume-flow conversion unless the
question instructs you to do so.

## 3. Map WaterTAP arguments

Map question variables to `simulate_ro` arguments using `parameter_mapping.json` as the
canonical vocabulary. Before calling, verify that pressure, flow, salinity, temperature,
membrane area, A, B, permeate pressure, pressure drop, concentration-polarization mode,
mass-transfer mode, and any required channel geometry are all accounted for.

Call `describe_ro_parameters` first only when an argument name or unit is genuinely
uncertain. Otherwise call `simulate_ro` directly with every question-specified input.

## 4. Run a baseline

Run one physically meaningful baseline or the explicitly requested normal state. Preserve:

- the exact tool arguments;
- whether the tool returned success or an error;
- the outputs needed for later calculations.

Never say "WaterTAP gives" or "the simulation shows" unless the tool was actually called.

## 5. Analyze margins

For every constraint, calculate a signed safety margin using a direction-aware formula:

- lower-bound constraint: `value - lower_bound`;
- upper-bound constraint: `upper_bound - value`.

Positive means feasible, zero means active, and negative means violated. When two
requirements constrain the same quantity, derive both limits and use the stricter one.

## 6. Search the boundary

### Pressure-selection recipe

1. Establish a lower candidate and evaluate it.
2. Establish a feasible upper candidate without changing fixed inputs.
3. Identify the active or nearly active constraint.
4. Refine only inside the infeasible/feasible bracket using direct simulations.
5. If interpolation is used, label it as an estimate and directly simulate the resulting
   candidate before declaring it feasible whenever the tool is available.
6. Compare the boundary candidate with at least one neighboring operating point.

### Membrane-degradation recipe

1. Simulate the normal parameter value.
2. Derive the strictest water-quality limit before testing degradation points.
3. Find one passing and one failing degradation value while holding every other input fixed.
4. Refine inside that bracket.
5. Identify which constraint becomes active first.
6. Report the maximum feasible parameter and its relative change from the normal state.

Do not claim global optimality without a bracket and a final constraint check.

## 7. Validate all constraints

Use a checklist. For every constraint report:

- simulated or derived value;
- limit;
- direction (`>=` or `<=`);
- signed margin;
- pass/fail.

Also report requested diagnostic quantities even when they are not binding constraints.
If any required quantity is absent from the tool output, derive it transparently or mark it
unavailable; do not omit it silently.

## 8. Form engineering judgment

Distinguish these concepts explicitly:

- numerical or theoretical boundary;
- directly simulated feasible point;
- practical fixed operating recommendation;
- early-warning threshold.

Explain why a lower and a higher neighboring alternative were not selected. Discuss the
trade-off relevant to the question, such as production margin, recovery, energy, water
quality, salt passage, or end-of-module concentration.

## 9. Self-check and report

Before finalizing, verify:

- all fixed inputs stayed constant across candidate simulations;
- every unit conversion is dimensionally correct;
- every requested output is present;
- all constraints were checked in the correct direction;
- arithmetic derived from tool output is reproducible;
- estimates are not described as direct simulations;
- the final recommendation follows from the candidate comparison.

Use these final sections so evaluators can align evidence to workflow stages:

1. `Task and fixed inputs`
2. `Derived targets and limits`
3. `WaterTAP calls and candidate results`
4. `Constraint and margin table`
5. `Boundary and engineering recommendation`
6. `Limitations and verification status`

