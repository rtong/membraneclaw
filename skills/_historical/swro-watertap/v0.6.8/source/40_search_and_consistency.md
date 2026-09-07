## Call budget (mandatory)

Context is finite (~16k tokens); every tool result is echoed back. More calls = higher risk of
running out before the answer. Budget is a hard constraint.

- Read the benchmark's reference budget (e.g. "about 8 calls", "8–12", or "10+10+10 parallel
  calls"); set total budget to that + at most 2 slack. If none stated, default to **≤10 calls**.
- Multi-simulator tasks: large reference budgets assume **parallel** calls. Never serialize
  independent simulations that one batch could carry.
- Per boundary: **≤4 calls** (1 fail, 1 pass, ≤1 verification, ≤1 interpolation), then stop.
- Per screened case: **≤1 call**, except the controlling case.
- When budget is nearly spent (≥ budget−3 used), stop searching; write the complete
  recommendation and constraint check from existing evidence.
- Ending mid-reasoning with no conclusion is a failure. A complete answer on existing evidence
  beats an abandoned search.

## State ledger (mandatory)

Maintain a compact table, updated after **every** tool call and read before the next: case/candidate
id, parameter values, pass/fail per constraint, bound found or "resolved". Choose each next call
only if the ledger says that point is still unknown and information-relevant. Never re-simulate a
resolved point or probe a resolved region. The ledger is the source of truth for what remains.

## Bounded search

- test explicit candidates from the question first;
- establish one failing and one passing point for a boundary; ≤1 interpolation;
- directly verify the final chosen boundary or recommendation;
- leave budget for the final explanation;
- if the question fixes a resolution (e.g. `0.05 bar`), stop once the fail/pass bracket proves the
  answer at that resolution; keep a progress checklist and move on to unfinished stages.

## Dominance pruning (mandatory)

For a bounded domain, first evaluate the maximal corner (max of every decision variable).

- Corner fails any hard target → record failure, declare infeasible, **STOP**. No interior scan.
- Corner passes → it is the answer; verify only what the question asks.
- Named points in the question are the required proof chain: simulate exactly those, in order,
  and report each. Never re-simulate a point dominated by an already-decided point.

## Decision-grain discipline (mandatory)

If the question fixes a grid or resolution (e.g. `0.001` step):

- report only grid-aligned values; prove the boundary with exactly two adjacent grid points
  (one pass, one fail), then STOP; any off-grid probe or quoted off-grid number is a defect.

## Controlling-case focus (mandatory)

Multi-case tasks: after the initial screen, resolve the case that binds or fails most first; do not
refine an already-passing case beyond one verification; keep a checklist of resolved vs binding
cases and move budget to the binding one.

## Joint use of retrieval and tools

Frame from the question → retrieve only if framing/interface details are missing → bounded tool
plan → tool evidence → final answer. Never "retrieved answer first, tools as decoration".

## Consistency and self-correction

After each key calculation or call, audit: do two representations of the same quantity agree in
unit/scale; does the conclusion match tool evidence; was a candidate eliminated only by a possibly
misconverted derived value; did rounding flip a fail into a pass. Reconcile material
disagreements before carrying them into elimination or the recommendation.

## Evidence over prior trend

If tool results contradict an assumed trend: re-check and accept the observed direction unless a
call was plainly wrong; confirm direction with more than one point before planning further search;
never eliminate a candidate on a trend the tool evidence already contradicts.
