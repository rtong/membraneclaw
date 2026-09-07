---
name: swro-watertap
description: Plan and execute auditable SWRO and WaterTAP decisions while preserving task inputs, constraints, units, and evidence provenance.
---

# SWRO decision protocol

Treat the question and retrieved project notes as different evidence sources. The question controls the requested task, candidates, constraints, and explicitly stated inputs. Retrieved notes may supply a missing project fact, but never override a value stated in the question. Do not retrieve general background when the task is already fully specified.

Before calling a tool, compile a compact private ledger with the decision variable, every fixed input, candidate set or resolution, hard constraints with direction and unit, required outputs, and the appropriate simulator. Mark each value as question-stated, retrieved, tool-returned, or derived. Copy thresholds exactly and do not replace a stated value with a default.

Build every call from this ledger. Pass all question-stated fixed inputs explicitly, including optional-looking coefficients. Between comparable calls change only the declared decision variable or case-specific fields. Reject a result if echoed inputs drifted or a required fixed value defaulted; correct the call before using the result.

Use `simulate_ro` for membrane-module performance, `simulate_swro_system` for whole-plant production or economics, and scaling/speciation tools for chemistry boundaries. Use a description tool only when the argument contract is actually unknown. Compare raw values and round only for presentation. Convert units explicitly, including `m3/s x 86,400 = m3/d` and water `kg/s x 3.6 = m3/h`.

## Plan for information gain

Choose a strategy from the task structure rather than following a fixed call count.

- Named discrete candidates: evaluate each candidate once with identical locked inputs, reject every constraint violator, then rank only survivors.
- One-variable boundary: establish a fail/pass bracket, use observed metric change to move near the boundary, and verify adjacent grid points at the requested resolution.
- Multi-case common window: screen cases at a common point, identify the case and constraint controlling each side, refine only those controlling bounds, then intersect the bounds.
- Multiple simulators: call only the simulator responsible for each required quantity and join results through explicitly matched inputs and units.

After each result, update which decision or boundary it can change. Do not repeat an identical call, densely scan an already bracketed range, or refine a non-controlling constraint. Stop when the decision and every required constraint are supported. If evidence is insufficient, state exactly what remains unknown instead of inventing a result.

## Report the supported decision

Begin with the selected candidate, verified boundary, or infeasibility conclusion. Then give the controlling numerical evidence, check every requested hard constraint, and identify the provenance of any retrieved fact that affected the decision. Report corrected or invalid calls when they matter to auditability.

Keep the answer compact but complete. Never claim an unobserved result, hide a violated constraint, or present a retrieved assumption as if it came from the question. Emit any required machine-readable trailer immediately after the natural-language answer.
