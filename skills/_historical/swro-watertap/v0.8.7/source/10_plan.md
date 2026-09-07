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
