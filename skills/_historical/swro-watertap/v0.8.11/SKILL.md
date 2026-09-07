---
name: swro-watertap
description: Execute SWRO and WaterTAP calculations with fixed inputs preserved and a short bounded search for numeric treatment decisions.
---

# Freeze one call template

Before the first tool call, construct this internal object from the question:

```text
BASE_ARGS = {
  composition_mol_s: all stated species and flows,
  temperature_c: stated temperature,
  pressure_bar: stated pressure,
  ph: stated FEED pH,
  water_recovery: stated recovery,
  minerals: every named control mineral
}
```

No call is allowed until every stated field appears in `BASE_ARGS`. Feed pH is an input and must never be omitted or replaced by a returned-pH limit. Include every named mineral even in the zero baseline.

Every tool call is exactly `BASE_ARGS + current decision candidate`; never rebuild or shorten the arguments between calls. Use the relevant RO tool for membrane or plant performance and `equilibrate_feed` for chemistry boundaries. Skip tool-description and retrieval calls when the question already supplies the contract.

After each result, compare echoed `inputs.ph`, `inputs.water_recovery`, composition, and minerals with `BASE_ARGS`. On any mismatch, mark the result `INVALID`, do not reason from it, and correct the same candidate. Invalid results still consume the call budget.

## Shared-budget boundary search

For one numeric decision variable on grid `G`, keep one compact table containing the candidate and every required constraint result. A tool result updates all constraints; never rerun a candidate or solve each constraint with a separate sweep.

Before labeling PASS or FAIL, normalize every signed comparison:

```text
y <= L: residual = y - L; PASS only if residual <= 0
y <  L: residual = y - L; PASS only if residual < 0
y >= L: residual = L - y; PASS only if residual <= 0
```

Never replace this calculation with wording such as "approximately below" or "just passes."

Identify the `decision boundary` that determines the final recommendation and close it before any report-only boundary. Maintain the nearest valid PASS/FAIL bracket for each requested boundary. Choose the next candidate as an on-grid midpoint that reduces the live interval most; use a sequential adjacent point only when the interval is at most `2G`. Never use an off-grid value. Once adjacent grid points prove a boundary, freeze it and make no more calls inside that interval.

Use one shared result budget: at most 6 results for one boundary and 9 results for multiple requested boundaries. Count invalid and failed calls. With only two calls left, use them only to close the decision boundary. At the limit, stop and answer with the proven boundary or the narrowest honest bracket; never start another tool call or overclaim an unproved maximum/minimum.

## Answer

Use concise natural language only: decision, adjacent fail/pass evidence, all required checks at the selected point, and one limitation. Do not output JSON, tags, copied tool results, or the full call history.
