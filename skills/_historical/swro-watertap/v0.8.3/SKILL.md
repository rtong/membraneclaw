---
name: swro-watertap
description: Execute SWRO and WaterTAP tools from a complete fixed-input template, keep input values separate from output limits, and stop once decisive evidence exists.
---

# Do this before the first tool call

Copy the question into two short records:

- `TOOL_BASE`: every fixed, specified, given, existing, or baseline value that describes the simulated case;
- `LIMITS`: every pass/fail threshold applied to tool outputs.

Do not move a value between these records.

## Chemistry pH rule

| Question meaning | Where it goes |
|---|---|
| Feed pH | `TOOL_BASE.ph`, passed as tool argument `ph` |
| Minimum/maximum acceptable result pH | `LIMITS.pH`, checked after the call |

Variable example: if a question says `Feed pH = X` and requires `result pH >= Y`, call the tool with `ph: X`, then compare returned pH with `Y`. Never call with `ph: Y`. Never omit `ph: X`.

## Copy this call shape

For `equilibrate_feed`, populate every field stated by the question before calling:

```text
composition_mol_s: TOOL_BASE.composition
temperature_c: TOOL_BASE.temperature
pressure_bar: TOOL_BASE.pressure
ph: TOOL_BASE.feed_pH
water_recovery: TOOL_BASE.recovery
minerals: TOOL_BASE.minerals
decision variable: current candidate value
```

If the question states any listed value and the call does not contain it, the call is forbidden.

After every result, compare echoed `inputs` with `TOOL_BASE` before reading SI, flow, pressure, cost, or pH outputs. On any mismatch, write internally `INVALID(field)` and immediately repeat the same candidate with the corrected fixed arguments. Do not use or explain outputs from the invalid call.

Use `simulate_ro` for membrane-module performance, `simulate_swro_system` for whole-plant results, and chemistry/speciation tools for scaling boundaries. Skip description and retrieval calls when the question already supplies the required contract.

## Use the smallest sufficient search

For a one-variable grid boundary, seek four evidence roles: requested baseline, one informative coarse grid point, adjacent failing point, and adjacent passing point. A corrected invalid call replaces its invalid predecessor; the invalid result is not evidence.

Call only requested grid values. As soon as adjacent grid points give one fail and one pass and all `LIMITS` are checked, stop all tools immediately. Do not repeat either point, test between them, or add confirmation beyond them.

For discrete candidates, test each under the same `TOOL_BASE`, reject failures, and rank survivors. For multi-case windows, refine only the cases controlling each boundary.

## Finish immediately after the decision

Report the selected value or infeasibility first, followed by the adjacent boundary evidence and every hard-constraint check. Reproduce the actual `TOOL_BASE` in the fixed-input summary, including feed pH and recovery when stated.

List only tool calls actually made. An invalid call may be disclosed for audit but must not support the conclusion. Keep the response short enough to append the required machine-readable trailer immediately; after the trailer, stop.
