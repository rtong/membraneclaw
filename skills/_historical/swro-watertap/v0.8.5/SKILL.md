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

## Single-boundary hard stop

Apply this section only to one case with one numeric decision variable and a stated grid step `G`.

- Use at most **six total tool results**, including description, calculator, invalid, and computational results.
- Every tested candidate must equal an integer grid point `n * G`.
- Seek a baseline if requested, one informative coarse point, then adjacent boundary points.
- Infer the passing direction from observed results. If larger values move the controlling metric toward passing, the next candidate after the closest failing grid point `A` must be `A + G`, never `A - G`. Reverse this rule when smaller values move toward passing.
- If `A` fails and adjacent `A + G` passes, the answer is `A + G`: set `FINAL_NOW`.
- Values strictly between adjacent grid points are forbidden and cannot improve the requested answer.
- At `FINAL_NOW`, or immediately after result six, call no tool and perform no further search. Report the proved boundary or the best supported unresolved bracket.

Do not use a calculator for simple comparisons and do not repeat a point for confirmation.

For discrete candidates, test each under the same `TOOL_BASE`, reject failures, and rank survivors. For multi-case windows, refine only controlling cases; the six-result rule does not apply to these other task structures.

## Final response at `FINAL_NOW`

Use at most eight short natural-language lines: decision, adjacent evidence, required constraint checks, and one limitation statement. Then append the trailer immediately.

Copy this exact JSON structure. Replace values but preserve every comma, brace, bracket, key, and tag:

```text
[SCORE_POINTS_BEGIN]
{"task_type":"short_task_type","decision_variables":{},"fixed_inputs":{},"tool_calls":["tool@candidate"],"constraint_checks":{},"final_answer":"supported_answer"}
[SCORE_POINTS_END]
```

In `tool_calls`, use one short string per call actually made; never replace the list with `call_count`. Reproduce stated feed pH and recovery in `fixed_inputs`. Before sending, verify that the trailer parses as one JSON object and that every `[` has `]` and every `{` has `}`. After the closing tag, stop.
