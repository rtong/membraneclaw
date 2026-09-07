---
name: swro-watertap
description: Plan and execute auditable SWRO and WaterTAP decisions while locking stated inputs, validating tool echoes, respecting decision grids, and preserving evidence provenance.
---

# SWRO decision protocol

The question controls the requested task, candidates, constraints, units, and every explicitly stated input. Retrieved notes may fill a genuine information gap but never replace a question value. Skip retrieval and description tools when the task already names the simulator and supplies its inputs.

## Lock the call before using a tool

Before the first computational call, create one compact private `LOCKED_ARGS` record containing every question-stated fixed input, the decision variable, allowed candidates or grid, hard constraints, required outputs, and simulator. Build every comparable call as:

`CALL_ARGS = LOCKED_ARGS + {only the current decision-variable value}`

Do not call until every stated fixed value is represented. For `equilibrate_feed`, map a stated feed pH to `ph`, recovery to `water_recovery`, pressure to `pressure_bar`, temperature to `temperature_c`, and the required control minerals to `minerals`. A parameter remains locked even when the tool marks it optional or has a default.

## Validate the echoed inputs before using a result

Read the tool's echoed `inputs` before interpreting scientific outputs. A result is valid only when every question-stated fixed input matches `LOCKED_ARGS`. A missing value that defaulted, including a stated pH, is a mismatch. Mark that result invalid, correct the next call, and do not use the invalid result to choose a boundary or final answer.

Use `simulate_ro` for membrane-module performance, `simulate_swro_system` for whole-plant production or economics, and scaling/speciation tools for chemistry boundaries. Use a description tool only when the required argument contract is genuinely unknown. Compare raw values, preserve units, and round only for presentation.
