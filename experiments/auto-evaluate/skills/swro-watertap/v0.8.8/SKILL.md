---
name: swro-watertap
description: Execute SWRO and WaterTAP calculations with fixed inputs preserved and a short bounded search for numeric treatment decisions.
---

# Fixed inputs

Before any tool call, copy all stated temperature, pressure, feed pH, recovery, composition, units, minerals, and other fixed values. Every call must reuse them unchanged.

Feed pH is a tool input; a returned-pH limit is only an output check. If feed pH or recovery is stated, explicitly pass `ph` and `water_recovery`; never accept defaults.

Use the relevant RO tool for membrane or plant performance and `equilibrate_feed` for chemistry boundaries. Skip tool-description and retrieval calls when the question already supplies the contract.

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
