---
name: swro-watertap
description: Solve SWRO and WaterTAP engineering tasks with bounded, auditable tool use; preserve every locked input, target controlling constraints, and finish before a 16k context is exhausted.
---

# SWRO bounded execution protocol

Use the question as the source of truth. Direct results from the current case outrank retrieval;
retrieve only for a genuinely unclear tool contract or engineering interpretation. Skip retrieval and
description tools when the question already names the simulator, inputs, units, and procedure.

## Compile once before calling tools

Make a compact private ledger containing: decision variable, every fixed input, case-specific fields,
hard constraints with direction and unit, required resolution, outputs, and tool. Copy thresholds
exactly. Do not add a margin or substitute a tool default for a stated value.

Before every computational call, build its arguments from that ledger. Pass every question-stated
fixed input explicitly, including coefficients that the tool marks optional. Between calls change
only the declared decision variable or case-specific fields. If the visible call omits a locked input,
do not send it; repair the arguments first. Check echoed inputs before using a result. A drifted or
defaulted result is invalid and permits one corrected call only.

Use `simulate_ro` for membrane-module performance, `simulate_swro_system` for whole-plant production
or economics, and scaling/speciation tools for chemistry boundaries. Description tools are allowed
at most once and only when an argument contract is actually unknown.

Compare raw values and round only for presentation. Convert units explicitly: `m3/s x 86,400 =
m3/d`; water `kg/s x 3.6 = m3/h`. Never compare quantities with different units.

## Search for the decision, not a dense curve

The 16k context is the binding resource. Use at most **8 total tool results**, including description,
calculator, retrieval, and failed calls. At 6 results, exploration ends; reserve the last two only
for decisive adjacent-boundary or final-candidate verification. After result 8, call no tool and
answer. Do arithmetic directly when it does not require a domain simulator.

Evaluate each named candidate once, check every hard constraint, and rank only survivors. For a
one-variable boundary, form one fail/pass bracket, use the observed metric change to estimate the
threshold location, then verify the two adjacent grid points at the requested resolution. Do not
walk through round-number midpoints or repeatedly halve a wide interval when numeric interpolation
can jump near the boundary.

### Multi-case feasibility window

1. Screen all cases once at one common starting point.
2. Identify the case and constraint controlling each side of the possible interval. A case that
   passes one opposing constraint and fails another is high priority.
3. Probe that controlling case once on the other side of the conflict, estimate each threshold from
   observed values, and spend remaining calls on the adjacent grid points.
4. Intersect the resulting bounds. If the lower bound exceeds the upper bound, declare the common
   set empty immediately; do not rescan resolved cases. If non-empty, verify only the proposed point
   and any unresolved controlling case.

Follow observed tool behavior rather than an assumed physical trend. Never repeat an identical call
or spend the remaining budget improving a non-controlling boundary.

## Finish while evidence is still available

Once decisive evidence exists, or a context/call warning appears, stop exploring. Begin with the
decision or infeasibility so truncation cannot hide it. Then identify controlling evidence, report
the verified boundary or candidate values, check every requested constraint, and disclose missing
evidence or corrected calls. Never claim an unobserved result.

Keep the answer compact. Reserve output space for any required machine-readable trailer and emit it
immediately after the natural-language answer. A complete supported decision is more valuable than
one additional probe.
