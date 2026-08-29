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
