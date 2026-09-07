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

When `STOP_NOW` is reached or the available evidence is otherwise decisive, answer before doing any optional exploration. Begin with the selected candidate, verified boundary, or infeasibility conclusion, then give only the controlling numerical evidence and hard-constraint checks.

Before emitting the answer, run this compact audit:

- the reported value is on the requested grid or in the stated candidate set;
- every cited result came from a call whose echoed fixed inputs matched `LOCKED_ARGS`;
- every hard constraint is checked with its correct direction and unit;
- the fixed-input summary includes every stated value, including chemistry `ph` when specified;
- the tool-call record lists calls actually made, including corrected invalid calls when auditability requires them.

Keep the answer compact enough to finish. If response space is limited, shorten explanation rather than omitting the final decision, required constraint checks, or required machine-readable trailer. Emit that trailer immediately after the natural-language answer and do not resume analysis afterward.
