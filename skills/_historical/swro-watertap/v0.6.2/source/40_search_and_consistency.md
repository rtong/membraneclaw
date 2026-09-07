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

## Dominance pruning

When an operating point is dominated by one that has already been tested, a failed dominant point
implies a failed dominated point, and a passing dominated point implies a passing dominant point.

- If the max-flow / max-pressure corner already fails a target, any smaller flow or lower pressure
  cannot pass that target: do not probe the dominated points, record the failure and move on;
- if a fixed and already-passing case would only improve under the candidate, one verification
  enough; do not re-screen it repeatedly.

Never re-simulate a point that cannot change the feasibility decision because a dominating point was
already decided.

## Decision-grain discipline

If the question fixes a decision grid or required resolution (for example a `0.001` step or `0.05 bar`):

- report answers aligned to that grid, never an intermediate off-grid value;
- a boundary is proven once one grid point passes and the adjacent grid point fails;
- once the adjacent pair is established, stop: do not probe between grid points and do not quote
  an off-grid number as the answer;
- finer is not better: an off-grid value contradicts the grid the question asked for.

## Controlling-case focus

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