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

## Choose the shortest valid search

Use the reference call budget when stated; otherwise use at most 10 results including failed calls.
Dispatch independent candidates together when supported. Do not repeat an identical call. A solver
error permits one corrected retry only when a specific argument or scale defect is identified.

### Explicit candidates

Evaluate each named candidate once, apply every hard constraint, discard infeasible candidates,
then rank only the survivors by the requested objective. Join multi-simulator branches by candidate
ID; never infer plant energy from membrane flux.

### One-variable boundary

Establish direction with existing evidence or one informative probe. Form one fail/pass bracket,
then verify the two adjacent grid points at the requested resolution. Stop after the adjacent pair;
do not refine an already proven boundary.

### Multi-case feasibility window

At one common starting point, screen every case once. Identify the controlling case and the
constraint that creates each side of the possible window. Follow observed tool behavior rather than
an assumed physical trend. Resolve the lower and upper boundaries on the controlling case, each with
adjacent points at the required resolution, then intersect the intervals.

- Empty intersection: declare no robust feasible value and stop; do not force a recommendation.
- Non-empty intersection: verify only the proposed point and any still-unresolved case, then apply
  the question's selection rule.

Do not rescan all cases at every point. A different constraint may control each side of the window.

## Stop and answer

After decisive evidence exists, or when a context/call-budget warning appears, make no exploratory
call. Produce the final answer immediately. It must state the decision or infeasibility, identify the
controlling evidence, show the verified boundary/candidate results, check every requested constraint,
and disclose missing evidence or corrected calls. Never end with analysis alone and never claim an
unobserved result.
