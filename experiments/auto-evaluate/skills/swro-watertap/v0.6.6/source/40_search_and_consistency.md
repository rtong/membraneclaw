## Call budget (mandatory)

Your context window is finite (~16k tokens) and every tool result is echoed back into it.
More calls = higher risk of running out of context before the final answer. Budget is a
hard constraint, not a preference.

- **Read the reference budget first.** Each benchmark states its reference trajectory and
  call budget (e.g. "about 8 calls", "8–12 calls", or "10+10+10 parallel calls"). Set your
  total tool-call budget to that reference, plus at most 2 calls of slack.
- If no reference number is stated, default to **at most 10 calls**.
- **Multi-simulator tasks** (e.g. membrane + plant + chemistry models): the reference can be
  large (e.g. 30 calls), but the calls are meant to run **in parallel**, not as a serial
  ladder. Never serialize independent simulations that could be issued in one batch; a serial
  chain of 30 dependent calls still blows the context.
- Per decision boundary: **at most 4 calls** — one failing point, one passing point, at
  most one verification, at most one interpolation. Stop after that.
- Per design case during screening: **at most 1 call**, except for the controlling case.
- When the budget is nearly spent (>= budget - 3 calls used), stop searching entirely and
  write the complete recommendation and constraint check from existing evidence.
- A transcript that ends mid-reasoning with no conclusion is a failure, even if a boundary
  is not perfectly bracketed. A complete answer on existing evidence beats an abandoned
  search with perfect precision.

## State ledger (mandatory)

Keep a compact table that is updated after **every** tool call, and read it before the
next call:

- candidate or case id;
- parameter values used;
- pass/fail for each stated constraint;
- lower/upper bound found so far, or "resolved".

Every next call must be chosen because the ledger says that point is still unknown and
information-relevant. Do not re-simulate a point the ledger already resolved. Do not probe
a resolved region again. The ledger is the source of truth for "what is left to do".

## Bounded search

Do not treat the environment as a random search engine.

- test explicit candidates first if the question provides them;
- establish one failing and one passing point when a boundary is needed;
- use at most one interpolation estimate when helpful;
- directly verify the final chosen boundary or recommendation;
- leave enough budget for the final explanation.

Precision without a complete recommendation is a failure.

If the question specifies a target resolution such as `0.05 bar`, stop the local search once the
best fail/pass bracket proves the answer at that resolution. Do not keep refining below the required
precision while later stages remain unfinished.

For multi-scenario tasks, keep an explicit progress checklist of remaining scenarios. After resolving
the current stage to the required precision, move on. Do not perfect the first stage while harder
stages are untested.

## Dominance pruning (mandatory)

Corner rule for a bounded feasibility domain: first evaluate the maximal corner, i.e. the point
with the maximum of every decision variable. A failed corner means the whole domain cannot pass:
every other point is dominated.

- If the corner fails any hard target: record the failure, declare infeasible, and STOP. Do not
  simulate, scan, or reason about any interior point. One interior simulation after a failed
  corner is a defect.
- If the corner passes: the corner is the answer; verify only what the question explicitly asks.

Named points: if the question names specific operating points (maximum pressure, baseline flow,
minimum pressure with maximum flow, ...), simulate exactly those named points, in the named order,
and report each one. They are the required proof chain. Do not skip a named point because the
corner already failed, and do not replace a named point with an interior scan.

Never re-simulate a point that cannot change the feasibility decision because a dominating point
was already decided.

## Decision-grain discipline (mandatory)

If the question fixes a decision grid or required resolution (for example a `0.001` step or `0.05 bar`):

- report answers aligned to that grid, never an intermediate off-grid value;
- prove the boundary with exactly two grid points: one passing, one failing, adjacent on the grid;
- STOP after those two points. Do not test any intermediate or off-grid point. Do not quote any
  off-grid number. The pair is the answer;
- any off-grid probe is a defect: finer is not better, it contradicts the grid the question asked for.

Example: with `0.1 g/L` grid, once `39.9` passes and `40.0` fails, stop. Do not test `39.95` or
`39.999`. Report the boundary on the grid immediately.

## Controlling-case focus (mandatory)

When a task has multiple cases or scenarios:

- after the initial screen, resolve the boundary of the case that binds or fails most first;
- do not refine an already-passing or non-limiting case beyond one verification;
- keep a checklist of which case is resolved and which still binds, and move budget to the case
  that binds.

## Joint use of retrieval and tools

Default sequence: frame from the question, retrieve only if framing or interface details are missing,
choose the tool plan, execute a bounded set of tool calls, use retrieved material only to interpret
the tool evidence, produce the final answer. Do not invert this into "retrieved answer first, tools
as decoration" or "long tool search without knowing what success means".

## Consistency and self-correction

After each important calculation or tool call, run a short internal audit:

- do two representations of the same quantity agree in unit and scale;
- does the current conclusion match the tool evidence;
- was a candidate eliminated only by a derived quantity that may be converted incorrectly;
- did rounding change a fail into a pass.

If two internally derived values for the same physical quantity disagree materially, stop and
reconcile them first. Do not carry the contradiction into candidate elimination or the final
recommendation.

## Evidence over prior trend

If observed tool direction contradicts an assumed trend:

- re-check the observation and accept the observed direction unless a tool call was plainly wrong;
- confirm trend direction with more than one data point before planning further search;
- never eliminate a candidate based on a suspected trend that the tool results already contradict.
