---
name: swro-watertap
description: Plan and execute auditable SWRO and WaterTAP decisions with separate input and constraint locks, validated tool echoes, grid-aware stopping, and evidence provenance.
---

# SWRO decision protocol

The question controls all stated inputs, candidates, constraints, units, and requested outputs. Retrieved notes may fill a genuine gap but never replace a question value. Skip retrieval and description tools when the task already supplies the simulator contract.

## Separate inputs from constraints before any call

Create two compact private records and never merge them:

- `INPUT_LOCK`: values that describe the simulated case and must be passed to the tool;
- `OUTPUT_CONSTRAINTS`: limits applied only to returned results.

Scan every sentence and table row marked fixed, specified, given, existing, or baseline. Copy each such value into `INPUT_LOCK`. A limit written with `>=`, `<=`, minimum, maximum, target, or margin belongs in `OUTPUT_CONSTRAINTS` unless the question separately states it as an input.

For chemistry tasks, keep these two pH concepts distinct:

- stated **feed pH** -> `INPUT_LOCK.feed_ph` -> tool argument `ph`;
- minimum or maximum allowed **result pH** -> `OUTPUT_CONSTRAINTS.result_ph` only.

Never substitute the result-pH limit for feed pH and never omit a stated feed pH because a pH constraint also appears.

## Mandatory chemistry call gate

Before each `equilibrate_feed` call, check every applicable line below. If the question states the value and the argument is absent or different, do not call the tool.

- `temperature_c == INPUT_LOCK.temperature_c`
- `pressure_bar == INPUT_LOCK.pressure_bar`
- `ph == INPUT_LOCK.feed_ph`
- `water_recovery == INPUT_LOCK.water_recovery`
- `composition_mol_s == INPUT_LOCK.composition_mol_s`
- `minerals == INPUT_LOCK.minerals`

Build comparable calls as `CALL_ARGS = INPUT_LOCK arguments + {only the current decision-variable value}`. Tool defaults never satisfy a question-stated input.

## Reject mismatched tool echoes

Read echoed `inputs` before reading scientific outputs. First classify the call as `ECHO_VALID` or `ECHO_INVALID(field)`. If any stated input is missing, defaulted, or different, ignore all scientific outputs and make the corrected call next. Do not summarize, search, or decide from an invalid call.

Use `simulate_ro` for membrane-module performance, `simulate_swro_system` for whole-plant production or economics, and scaling/speciation tools for chemistry boundaries. Compare raw values, preserve units, and round only for presentation.

## Plan for information gain

Choose calls by the evidence needed for the decision, not by a universal call cap.

For a one-variable boundary on a stated grid, plan only these evidence roles: a baseline when needed, one discriminating coarse point, and the adjacent failing and passing grid points that prove the boundary. Use observed changes to move toward the boundary; do not scan from the smallest value upward when a larger informative probe can bracket it.

Apply this grid stop gate after every valid result:

- call only values on the requested grid;
- once adjacent grid points give one fail and one pass and every hard constraint has been checked, set `STOP_NOW`;
- after `STOP_NOW`, make no more tool or description calls and write the answer immediately;
- do not add a trend check, safety check, or finer off-grid probe after the boundary is already proved.

For named discrete candidates, evaluate candidates with identical locked inputs, reject constraint violators, and rank only survivors. For a multi-case common window, screen at a common point, refine only the cases controlling the lower and upper bounds, and stop when their intersection is proved. If the evidence cannot support a decision, state the unresolved gap instead of enumerating low-information calls.

## Report the supported decision

When `STOP_NOW` is reached or the available evidence is otherwise decisive, answer before doing optional exploration. Begin with the selected candidate, verified boundary, or infeasibility conclusion, then give only the controlling numerical evidence and hard-constraint checks.

Before emitting the answer, audit that:

- the reported value is on the requested grid or in the stated candidate set;
- every cited result is `ECHO_VALID` under the same `INPUT_LOCK`;
- every `OUTPUT_CONSTRAINTS` item is checked with the correct direction and unit;
- the fixed-input summary reproduces `INPUT_LOCK`, including stated feed pH and recovery;
- the tool-call record lists calls actually made and does not present an invalid call as evidence.

Keep the answer compact enough to finish. If response space is limited, shorten explanation rather than omitting the final decision, constraint checks, or required machine-readable trailer. Emit that trailer immediately after the natural-language answer and do not resume analysis afterward.
