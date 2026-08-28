## Plan for information gain

Choose calls by the evidence needed for the decision, not by a universal call cap.

For a one-variable boundary on a stated grid, plan only these evidence roles: a baseline when needed, one discriminating coarse point, and the adjacent failing and passing grid points that prove the boundary. Use observed changes to move toward the boundary; do not scan from the smallest value upward when a larger informative probe can bracket it.

Apply this grid stop gate after every valid result:

- call only values on the requested grid;
- once adjacent grid points give one fail and one pass and every hard constraint has been checked, set `STOP_NOW`;
- after `STOP_NOW`, make no more tool or description calls and write the answer immediately;
- do not add a trend check, safety check, or finer off-grid probe after the boundary is already proved.

For named discrete candidates, evaluate candidates with identical locked inputs, reject constraint violators, and rank only survivors. For a multi-case common window, screen at a common point, refine only the cases controlling the lower and upper bounds, and stop when their intersection is proved. If the evidence cannot support a decision, state the unresolved gap instead of enumerating low-information calls.
