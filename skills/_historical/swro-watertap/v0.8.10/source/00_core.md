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
