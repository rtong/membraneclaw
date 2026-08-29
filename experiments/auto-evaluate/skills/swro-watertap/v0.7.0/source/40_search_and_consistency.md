## Call budget (mandatory)
Context is ~16k tokens and every tool result echoes back. `max_tokens` is 2048, so the effective
input budget is roughly 14k tokens. Treat the budget as a hard constraint.

- test explicit candidates from the question first; one failing + one passing point per boundary with <=1 interpolation; directly verify the final boundary; leave budget for the final explanation;
- reference budget + <=2 slack; none stated -> **<=10 calls**.
- Multi-simulator: reference budgets assume **parallel** calls; never serialize independent
  simulations one batch could carry.
- Per boundary: **<=4 calls** (1 fail, 1 pass, <=1 verification, <=1 interpolation), then stop.
- Per screened case: **<=1 call**, except the controlling case.
- At budget-3 used: stop searching; write the complete recommendation and constraint check from
  existing evidence. No conclusion = failure; a complete answer on existing evidence beats an
  abandoned search.

## State ledger (mandatory)
Compact table updated after every call, read before the next: candidate, parameters, pass/fail per
constraint, bound or "resolved". Probe only points the ledger marks unknown and information-relevant;
never re-simulate a resolved point or region. Ledger = source of truth for what remains.

## Dominance pruning (mandatory)
Bounded domain -> first evaluate the maximal corner (max of every decision variable).

- Corner fails any hard target -> record failure, declare infeasible, **STOP and write the
  conclusion immediately**; no interior scan.
- Corner passes -> it is the answer; verify only what the question asks.
- Simulate the question's named points in order, report each; never re-simulate a point dominated
  by an already-decided point.
- Once the decisive evidence exists (corner proof, or fail/pass bracket), do not search further:
  draft the final recommendation and constraint check with the remaining budget.

## Decision-grain discipline (mandatory)
Question fixes a grid/resolution (e.g. `0.001` or `0.1 g/L`): report only grid-aligned values; prove
the boundary with exactly two **adjacent grid points at that resolution** (one pass, one fail), then
STOP.

- Before stopping, verify the pair is truly adjacent at the question resolution: at 0.1 g/L, 39.0 and
  40.0 are **not** adjacent (39.9/40.0 are); 0.00 and 0.01 are adjacent at 0.01, not at 0.001.
- Any off-grid probe or quoted off-grid number is a defect.

## Controlling-case focus (mandatory)
Multi-case: after the initial screen, resolve the binding/failing case first; do not refine an
already-passing case beyond one verification; keep a resolved-vs-binding checklist and move budget
to the binding one.

## Joint use of retrieval and tools
Frame from the question -> retrieve only if framing/interface details are missing -> bounded tool
plan -> tool evidence -> final answer. Never "retrieved answer first, tools as decoration".

## Consistency and self-correction
After each key calculation or call, audit: do two representations of the same quantity agree in
unit/scale; does the conclusion match tool evidence; was a candidate eliminated only by a possibly
misconverted derived value; did rounding flip a fail into a pass. Reconcile material disagreements
before carrying them into elimination or the recommendation.

## Evidence over prior trend
Tool results contradict an assumed trend -> re-check and accept the observed direction unless a
call was plainly wrong; confirm with more than one point before planning further search; never
eliminate a candidate on a trend the tool evidence already contradicts.


