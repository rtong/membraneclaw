## Choose the shortest valid search

Use the reference call budget when stated; otherwise use at most 10 results including failed calls.
Dispatch independent candidates together when supported. Do not repeat an identical call. A solver
error permits one corrected retry only when a specific argument or scale defect is identified.

### Explicit candidates

Evaluate each named candidate once, apply every hard constraint, discard infeasible candidates,
then rank only the survivors by the requested objective. Join multi-simulator branches by candidate
ID; never infer plant energy from membrane flux.

### One-variable boundary

Establish direction with existing evidence or one informative probe. Form one fail/pass bracket,
then verify the two adjacent grid points at the requested resolution. Stop after the adjacent pair;
do not refine an already proven boundary.

### Multi-case feasibility window

At one common starting point, screen every case once. Identify the controlling case and the
constraint that creates each side of the possible window. Follow observed tool behavior rather than
an assumed physical trend. Resolve the lower and upper boundaries on the controlling case, each with
adjacent points at the required resolution, then intersect the intervals.

- Empty intersection: declare no robust feasible value and stop; do not force a recommendation.
- Non-empty intersection: verify only the proposed point and any still-unresolved case, then apply
  the question's selection rule.

Do not rescan all cases at every point. A different constraint may control each side of the window.
