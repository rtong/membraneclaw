---
name: swro-watertap
description: Plan and execute auditable SWRO and WaterTAP decisions while preserving task inputs, constraints, units, and evidence provenance.
---

# SWRO decision protocol

Treat the question and retrieved project notes as different evidence sources. The question controls the requested task, candidates, constraints, and explicitly stated inputs. Retrieved notes may supply a missing project fact, but never override a value stated in the question. Do not retrieve general background when the task is already fully specified.

Before calling a tool, compile a compact private ledger with the decision variable, every fixed input, candidate set or resolution, hard constraints with direction and unit, required outputs, and the appropriate simulator. Mark each value as question-stated, retrieved, tool-returned, or derived. Copy thresholds exactly and do not replace a stated value with a default.

Build every call from this ledger. Pass all question-stated fixed inputs explicitly, including optional-looking coefficients. Between comparable calls change only the declared decision variable or case-specific fields. Reject a result if echoed inputs drifted or a required fixed value defaulted; correct the call before using the result.

Use `simulate_ro` for membrane-module performance, `simulate_swro_system` for whole-plant production or economics, and scaling/speciation tools for chemistry boundaries. Use a description tool only when the argument contract is actually unknown. Compare raw values and round only for presentation. Convert units explicitly, including `m3/s x 86,400 = m3/d` and water `kg/s x 3.6 = m3/h`.
