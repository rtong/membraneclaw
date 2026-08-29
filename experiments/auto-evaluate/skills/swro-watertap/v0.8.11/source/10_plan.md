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
