## One-decision-variable grid boundary

This mode applies whenever the task has exactly one numeric decision variable and states a grid step `G`. The number of output constraints does not matter: Calcite, Dolomite, Gypsum, pH, pressure, cost, and other simultaneous limits still form one boundary search when only one input is being varied.

Set `N=0` before searching. Immediately before every proposed tool call, apply this gate:

```text
if FINAL_NOW or N >= 6: do not call a tool; write the final response
otherwise: send one candidate and set N = N + 1
```

- A candidate passes only if every required limit passes.
- Every candidate must be an integer grid point `n*G`; never test values between grid points.
- If a zero baseline is requested, call it once. Use at most two informative coarse candidates, then spend remaining calls only on the closest boundary.
- When no better scale is supplied for a nonnegative treatment dose, prefer `10G` as the first nonzero screen; double once if needed. Do not begin with `G` merely to walk the grid.
- Infer the improvement direction from results. If larger values improve the controlling failed limit, search upward; if smaller values improve it, search downward.
- The instant adjacent grid points prove `A=FAIL` and `A+G=PASS` in the upward direction, set `FINAL_NOW` and select `A+G`. For a downward search, use the symmetric rule.
- After adjacent proof, all confirmation calls are forbidden: do not repeat either point, test the other neighbor, test a wider point, or say “double-check.” Multiple constraints do not cancel this stop.
- After result six, stop even without proof and report the best supported bracket.

Do not call a calculator for simple comparisons. For genuinely discrete alternatives or multi-case windows, keep fixed inputs constant and prune failed branches; the grid-boundary procedure above applies only when its one-variable condition is met.
