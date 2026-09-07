---
name: swro-parallel-compare
description: Compare finite SWRO packages using independent membrane and plant calculations, preserving candidate identity and selecting only jointly feasible packages. Not for serial chemistry coupling or continuous optimization.
---

# Parallel engineering comparison

Extract the candidate table, scenario, model scales, requested outputs, constraints and objective from the public question. Record the source of each threshold. Missing facts cannot be filled from reference answers. Distinguish stated inputs from documented tool defaults.

Keep the membrane and plant branches independent. Send each branch the inputs of the same candidate row. Map pressure explicitly to each tool's field. Module membrane area and plant membrane area are different inputs; do not infer plant energy from membrane flux. PX efficiency belongs to the plant branch.

Attach candidate, scenario, tool version and input snapshot to each result. Join only corresponding branches. Complete all explicitly requested simulations and outputs, even if a candidate already violates a constraint. Reuse across identical rows only when the task and shared cache policy permit it.

Check every constraint with original-precision values and its exact comparison operator. Rank only jointly feasible candidates using the stated objective. Preserve ties unless the question supplies a tie-break rule. Missing results mean insufficient evidence, not infeasibility. Report the finite search scope, evidence for exclusions, defaults and model limitations.

The paired [workflow.json](workflow.json) defines the prototype's executable input bindings. Load it only for this parallel task shape. It contains no case-specific candidates, thresholds or answers. The runtime schema and trusted tool adapter must be validated before live use.
