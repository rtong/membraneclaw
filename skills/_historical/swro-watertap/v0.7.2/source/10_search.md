## Search for the decision, not a dense curve

The 16k context is the binding resource. Use at most **8 total tool results**, including description,
calculator, retrieval, and failed calls. At 6 results, exploration ends; reserve the last two only
for decisive adjacent-boundary or final-candidate verification. After result 8, call no tool and
answer. Do arithmetic directly when it does not require a domain simulator.

Evaluate each named candidate once, check every hard constraint, and rank only survivors. For a
one-variable boundary, form one fail/pass bracket, use the observed metric change to estimate the
threshold location, then verify the two adjacent grid points at the requested resolution. Do not
walk through round-number midpoints or repeatedly halve a wide interval when numeric interpolation
can jump near the boundary.

### Multi-case feasibility window

1. Screen all cases once at one common starting point.
2. Identify the case and constraint controlling each side of the possible interval. A case that
   passes one opposing constraint and fails another is high priority.
3. Probe that controlling case once on the other side of the conflict, estimate each threshold from
   observed values, and spend remaining calls on the adjacent grid points.
4. Intersect the resulting bounds. If the lower bound exceeds the upper bound, declare the common
   set empty immediately; do not rescan resolved cases. If non-empty, verify only the proposed point
   and any unresolved controlling case.

Follow observed tool behavior rather than an assumed physical trend. Never repeat an identical call
or spend the remaining budget improving a non-controlling boundary.
