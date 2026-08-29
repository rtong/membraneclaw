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
