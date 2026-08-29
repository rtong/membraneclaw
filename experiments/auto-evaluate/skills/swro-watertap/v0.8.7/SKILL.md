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

## One-variable grid boundary

Use this mode when exactly one numeric input is varied on a stated grid `G`. Any number of simultaneous output constraints still belongs to this one-variable search. A candidate passes only when every limit passes.

Keep a visible internal counter `N`. Before every tool call:

```text
if N == 6 or FINAL_NOW: no call is allowed; answer now
otherwise: call one candidate, then increment N
```

- Call the requested zero baseline once.
- Every nonzero candidate must equal `n*G` for an integer `n`; fractional grid points are forbidden even for information gathering.
- For a nonnegative treatment dose with no supplied scale, use `10G` as the default first nonzero screen, then at most one doubled coarse screen. Use remaining calls for integer-grid boundary refinement.
- Infer which direction improves the controlling failed limit. Do not search in the opposite direction.
- The instant adjacent points establish `A=FAIL` and `A+G=PASS` in an upward search, select `A+G` and set `FINAL_NOW`; apply the symmetric rule for a downward search.
- After adjacent proof, never repeat a point, test another neighbor, widen the search, or double-check. Stop even if multiple constraints were checked.
- If result six does not prove an answer, report the best supported bracket instead of calling again.

Do not use a calculator for simple comparisons. For discrete alternatives or multi-case windows, preserve fixed inputs and prune failed branches; do not force this grid procedure onto a different task structure.

## Final answer

Return only a concise natural-language answer. Do not output score-point tags, JSON, the complete call history, copied tool outputs, or a repeated version of the question.

Use at most four short parts:

1. the selected decision or best supported bracket;
2. the decisive adjacent fail/pass evidence;
3. every required constraint check at the selected point;
4. one relevant model or engineering limitation.
