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
