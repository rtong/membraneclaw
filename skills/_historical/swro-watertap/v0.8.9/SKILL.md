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

## Six-result state machine

For one numeric treatment variable on grid `G`, use this sequence; multiple output constraints do not change it:

```text
S0: requested zero baseline
S1: 10G first nonzero screen
S2: 20G only if 10G still needs stronger treatment
S3: integer-grid midpoint of the useful lower/upper bracket
S4: one adjacent grid point in the direction that fixes S3's failed limit
S5: one final adjacent point only if no adjacent boundary is proved
STOP: answer; never make result 7
```

Maintain result count `N` before every call. If `N >= 6`, answer with the best evidence. Never scan `G, 2G, 3G...`, use fractional grid points, repeat a candidate, or call a calculator for comparisons.

A candidate passes only if all limits pass. As soon as adjacent grid points show FAIL then PASS in the improving direction, select the passing point and stop immediately.

## Answer

Use concise natural language only: decision, adjacent fail/pass evidence, all required checks at the selected point, and one limitation. Do not output JSON, tags, copied tool results, or the full call history.
