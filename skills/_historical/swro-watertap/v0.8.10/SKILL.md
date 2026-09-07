---
name: swro-watertap
description: Execute SWRO and WaterTAP calculations with explicit fixed-input gates, checked treatment compositions, and a bounded numeric search.
---

# Pre-call gate

Before the first tool call, print one short line containing the actual values:

```text
PRECALL BASE_ARGS: temperature_c=...; pressure_bar=...; ph=FEED_PH; water_recovery=...; minerals=[every named mineral]; composition_mol_s={every stated species}
```

Do not call until that line contains every fixed value stated anywhere in the question. Feed pH is not the returned-pH limit. If recovery is stated, `water_recovery` must appear even when the tool accepts a default. Include every named mineral in the baseline.

Every call is `BASE_ARGS + current decision candidate`; do not rebuild, shorten, or silently default the fixed arguments. Use the relevant RO tool for membrane or plant performance and `equilibrate_feed` for chemistry boundaries. Skip description and retrieval calls when the question already supplies the contract.

# Post-call gate

Before interpreting a result, compare echoed `inputs.ph`, `inputs.water_recovery`, composition, and minerals with `BASE_ARGS`.

```text
If a stated input is null, absent, or different: INVALID; use no output; retry the same candidate with corrected BASE_ARGS.
```

An invalid result still consumes the call budget. Never continue the search from an invalid result.

# Composition transformations

For removal fraction `r` of species with charge magnitude `z_removed`, replaced by a species with charge magnitude `z_replacement`:

```text
residual = initial * (1 - r)
removed = initial - residual
replacement_added = removed * z_removed / z_replacement
check_r = 1 - residual / initial
```

Convert percentages to fractions before calculating. Require `check_r` to reproduce the candidate label. Put both `residual` and the charge-equivalent replacement amount into the actual `composition_mol_s` payload, not only the explanation.

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
