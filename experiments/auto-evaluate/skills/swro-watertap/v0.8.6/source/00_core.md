---
name: swro-watertap
description: Execute SWRO and WaterTAP tools with complete fixed inputs, separate input values from output limits, and stop as soon as decisive evidence exists.
---

# Before the first tool call

Create two short internal records:

- `TOOL_BASE`: every fixed, specified, given, existing, or baseline value that describes the simulated case;
- `LIMITS`: every pass/fail threshold applied to returned outputs.

Never move a value between them. A stated feed pH belongs in `TOOL_BASE.ph`; an acceptable result-pH threshold belongs in `LIMITS.pH`. Pass the feed pH to the tool, then compare the returned pH with the limit.

For `equilibrate_feed`, copy every stated field into every call:

```text
composition_mol_s: TOOL_BASE.composition
temperature_c: TOOL_BASE.temperature
pressure_bar: TOOL_BASE.pressure
ph: TOOL_BASE.feed_pH
water_recovery: TOOL_BASE.recovery
minerals: TOOL_BASE.minerals
decision variable: current candidate
```

If a stated field is missing, do not send the call. After a result, compare echoed `inputs` with `TOOL_BASE` before using any output. Correct an invalid call once; count that result against the tool budget.

Use `simulate_ro` for membrane performance, `simulate_swro_system` for whole-plant results, and chemistry/speciation tools for scaling boundaries. Skip description and retrieval calls when the question already supplies the tool contract.
