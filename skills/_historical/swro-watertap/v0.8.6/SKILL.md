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

## One-decision-variable grid boundary

This mode applies whenever the task has exactly one numeric decision variable and states a grid step `G`. The number of output constraints does not matter: Calcite, Dolomite, Gypsum, pH, pressure, cost, and other simultaneous limits still form one boundary search when only one input is being varied.

Set `N=0` before searching. Immediately before every proposed tool call, apply this gate:

```text
if FINAL_NOW or N >= 6: do not call a tool; write the final response
otherwise: send one candidate and set N = N + 1
```

- A candidate passes only if every required limit passes.
- Every candidate must be an integer grid point `n*G`; never test values between grid points.
- If a zero baseline is requested, call it once. Use at most two informative coarse candidates, then spend remaining calls only on the closest boundary.
- When no better scale is supplied for a nonnegative treatment dose, prefer `10G` as the first nonzero screen; double once if needed. Do not begin with `G` merely to walk the grid.
- Infer the improvement direction from results. If larger values improve the controlling failed limit, search upward; if smaller values improve it, search downward.
- The instant adjacent grid points prove `A=FAIL` and `A+G=PASS` in the upward direction, set `FINAL_NOW` and select `A+G`. For a downward search, use the symmetric rule.
- After adjacent proof, all confirmation calls are forbidden: do not repeat either point, test the other neighbor, test a wider point, or say “double-check.” Multiple constraints do not cancel this stop.
- After result six, stop even without proof and report the best supported bracket.

Do not call a calculator for simple comparisons. For genuinely discrete alternatives or multi-case windows, keep fixed inputs constant and prune failed branches; the grid-boundary procedure above applies only when its one-variable condition is met.

## Final response

At `FINAL_NOW`, do not plan, summarize the search history, or call another tool. The first content in the next assistant message must be this complete trailer:

```text
[SCORE_POINTS_BEGIN]
{"task_type":"short_task_type","decision_variables":{},"fixed_inputs":{},"tool_calls":["tool@candidate"],"constraint_checks":{},"final_answer":"supported_answer"}
[SCORE_POINTS_END]
```

Replace values but preserve all six keys, tags, commas, brackets, and braces. Use one short string per actual call in `tool_calls`; do not use `call_count`. Include stated feed pH and recovery in `fixed_inputs`. Close the JSON and end tag before writing prose.

After the trailer, use at most four short lines: decision, adjacent evidence, remaining constraint checks, and one model limitation. Do not reproduce tool outputs, the question, a table, or the complete call history.
