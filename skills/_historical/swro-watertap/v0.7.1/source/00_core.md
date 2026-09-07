---
name: swro-watertap
description: Solve SWRO and WaterTAP engineering tasks with bounded, auditable tool use; apply fixed inputs exactly, isolate controlling constraints, and stop with a complete decision before context is exhausted.
---

# SWRO bounded execution protocol

Use the question as the source of truth. Direct results from the current case outrank retrieval;
retrieval is only for a genuinely unclear tool contract or engineering interpretation. Skip retrieval
and description tools when the question already names the simulator, inputs, units, and procedure.

## Compile the task before calling tools

Create a compact private record containing: decision variable, every fixed input, cases/candidates,
hard constraints with direction and unit, required resolution, required outputs, and the intended
tool. Copy stated thresholds exactly. Do not add a margin or reinterpret a benchmark-specific target.

## Tool and argument invariants

- `simulate_ro`: membrane-module performance; use module area and module-scale mass flow.
- `simulate_swro_system`: whole-plant production, energy, cost, and CAPEX; use plant RO area and
  plant volumetric flow.
- scaling/speciation tools such as `analyze_ro_scaling` or `equilibrate_feed`: mineral SI,
  recovery/dose boundaries, and chemistry decisions.
- description tools: at most once, only when the required argument contract is unknown.

For every computational call, pass every question-stated fixed input explicitly. Between candidate
calls change only the declared decision variable or candidate-specific fields. Check echoed inputs
before using a result. If a fixed value drifted or defaulted, discard that result and make one
corrected call; never repair it by changing another input.

Compare raw values, then round for presentation. Convert units explicitly: `m3/s × 86,400 = m3/d`;
water `kg/s × 3.6 = m3/h`. Never compare quantities with different units.
