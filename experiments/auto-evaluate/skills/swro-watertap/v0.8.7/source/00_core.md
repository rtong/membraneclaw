---
name: swro-watertap
description: Execute SWRO and WaterTAP tools with every stated fixed input, keep input values separate from output limits, and stop when the requested decision is proved.
---

# Build the tool contract before calling

Read the entire question and complete this internal checklist:

```text
TOOL_BASE
[ ] temperature
[ ] pressure
[ ] feed pH
[ ] water recovery
[ ] composition and units
[ ] requested minerals or cases

LIMITS
[ ] every SI, pH, pressure, flow, cost, or other pass/fail threshold
```

Copy every value explicitly stated by the question. A feed pH is an input; an acceptable returned-pH threshold is a limit. If the question states `Feed pH = X`, every chemistry call must contain `ph: X`. Never replace it with the result-pH limit or omit it and accept the tool default.

For `equilibrate_feed`, every call must preserve this shape:

```text
composition_mol_s: TOOL_BASE.composition
temperature_c: TOOL_BASE.temperature
pressure_bar: TOOL_BASE.pressure
ph: TOOL_BASE.feed_pH
water_recovery: TOOL_BASE.recovery
minerals: TOOL_BASE.minerals
decision variable: current candidate
```

Immediately compare returned `inputs` with `TOOL_BASE`. A mismatched result is invalid evidence; correct that candidate once and count both results against the call budget.

Use `simulate_ro` for membrane performance, `simulate_swro_system` for whole-plant performance, and chemistry/speciation tools for scaling boundaries. Skip description and retrieval calls when the question already supplies the contract.
