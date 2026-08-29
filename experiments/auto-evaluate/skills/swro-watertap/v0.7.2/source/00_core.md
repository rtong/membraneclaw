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
