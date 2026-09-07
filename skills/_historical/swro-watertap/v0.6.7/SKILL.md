---
name: swro-watertap
description: Control how the model uses the SWRO environment — frame the task, retrieve only for orientation, use tools for case-specific evidence, arbitrate conflicts by evidence strength, and deliver an auditable final answer.
---

# SWRO environment-use protocol

This skill is not a bank of benchmark solutions. It controls how the model uses the two external
capabilities:

- retrieval over SWRO / WaterTAP knowledge;
- executable tools for case-specific computation and scaling analysis.

Retrieval is for understanding the problem and the tool interface. Tools are for deciding the case.
Never replace case-specific calculation with retrieved prose.

## Goal

Produce an answer that is grounded in the question, supported by actual tool evidence for
case-specific quantities, disciplined in retrieval use, and explicit about constraints, uncertainty,
and recommendation logic.

## Evidence hierarchy

When sources disagree, resolve them in this order:

1. explicit values and requirements stated in the question;
2. directly observed tool outputs for the current case;
3. retrieved domain guidance or tool-interface guidance;
4. prior intuition or generic engineering habit.

Never let retrieval override an explicit question value or a direct tool result for the current case.

## Operating protocol

Before acting, build one compact execution record with:

- task family;
- decision variable(s);
- fixed inputs;
- constraints with direction and unit;
- mandatory outputs;
- explicit candidates given by the question;
- which parts need retrieval, which parts need tools, which need neither.

Do not retrieve or call tools blindly.

Copy every numeric threshold and target from the question verbatim (e.g. `>= 10,200 m3/d`),
including its unit, into the execution record. Do not rephrase, redefine, re-derive, or round
these numbers. Every search target and every final recommendation must be checked against these
verbatim values, never against a remembered or inferred version of them.

Each value stated in the question is the final deciding threshold as given. Do not apply any
additional margin, safety factor, or percentage on top of it, and do not re-derive it from another
stated value unless the question explicitly states that transformation. A threshold that a question
calls a "margin", "recommended", or "target" value is still the complete criterion itself, not an
input to yet another multiplication.

If the answer depends on a number specific to this case, do not rely on retrieval alone: produce it
with a tool.

## When to retrieve

Retrieve only when it adds missing structure:

- the task wording is ambiguous and domain framing is needed;
- a tool argument name or supported option is uncertain;
- engineering interpretation, monitoring indicators, or mitigation logic is asked;
- species, units, or scaling interpretation need confirmation.

Skip retrieval when the question already specifies the needed inputs and procedure.

## Retrieval discipline

When retrieval is used:

- extract only actionable facts; do not copy large blocks of retrieved text into the answer;
- distinguish definition-level facts from case-specific claims;
- treat retrieved numeric values as guidance only, unless they are explicitly stated in the question;
- never manufacture hidden assumptions from retrieval;
- if retrieval is noisy or conflicting, fall back to question values plus tools.

Useful retrieval outputs are: which variable to search, which tool to use, what the tool expects for
arguments and units, and what engineering interpretation should accompany the result.

## Tool routing

Use tools by question type:

- `simulate_ro` for membrane sizing, pressure selection, parameter comparison, operating-point feasibility;
- `analyze_ro_scaling` for concentrate chemistry, mineral SI, pressure-impact-on-scaling, acid-dose selection;
- `describe_ro_parameters` or `describe_reaktoro_options` only when the interface contract is truly uncertain.

## Tool argument discipline

For every computational call:

- pass every question-stated input explicitly; never rely on defaults for a question-stated parameter;
- change only the declared decision variable between candidate evaluations;
- check echoed inputs or returned settings when available; repair drift before reasoning from the result;
- do not call a description tool when the needed argument contract is already clear.

For `simulate_ro`, explicitly lock flow, salinity, temperature, pressure, area, A/B, permeate
pressure, pressure drop, and transport options when the question gives them. For `analyze_ro_scaling`,
explicitly lock the full composition, pressure, temperature, area, pH, minerals, and acid dose.

## Unit-conversion discipline

Never compare quantities across different units without an explicit conversion step. When a
conversion is needed, use the exact factors below. Do not improvise a factor from memory;
use the table.

| Conversion | Factor | Example |
|---|---|---|
| `m3/s` -> `m3/d` | **x 86,400** | 0.12 m3/s = 10,368 m3/d |
| `m3/s` -> `m3/h` | x 3,600 | 0.12 m3/s = 432 m3/h |
| `m3/h` -> `m3/d` | x 24 | 432 m3/h = 10,368 m3/d |
| `kg/s` (water) -> `m3/h` | x 3.6 | 30 kg/s = 108 m3/h |
| `m3/d` -> `m3/s` | / 86,400 | 10,000 m3/d = 0.1157 m3/s |
| `MGD` -> `m3/d` | x 3,785.4 | 2.7 MGD = 10,221 m3/d |

Self-check after every conversion:

- day-to-second conversion always uses **86,400** (60 x 60 x 24), never 3,600 alone or
  3,600 x 1000;
- cross-check the result against the tool's own derived quantity when available: for a
  whole-plant run, `product_flow_m3_s x 86,400` should equal the tool's product-flow metric,
  and `product_flow_m3_s / feed_flow_m3_s` should match the tool's recovery fraction;
- if two representations of the same quantity disagree by a large factor (e.g. 24x, 1000x,
  or 41.7x), the conversion factor is wrong — re-read the tool unit, do not proceed;
- do not prune candidates until units are aligned.

Do not carry a memorized approximate factor (e.g. "x 3.6" or "x 1000") across units. Each
conversion is read from the table, applied once, and cross-checked before it is used in a
decision.

## Output-metric basis

Use the metric that the question asks for, taken from the tool output that defines it. In
particular, when the question requests a whole-plant energy figure (for example daily energy
consumed by the plant), use the plant specific energy consumption x produced water volume
(`SEC kWh/m3` x product flow), not a pump-train power x time. Pump power, pumping power, and SEC
are different quantities: do not interchange them when the question asks for plant-level energy.
If the tool exposes more than one energy-related value, state which one you used and why.

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

## Conflict resolution

If sources disagree:

- question vs retrieval: question wins;
- question vs tool echo on input values: repair the tool call to match the question;
- retrieval vs tool result on case behavior: tool result wins;
- retrieved heuristic vs hard constraint: hard constraint wins;
- tool output unavailable for a requested quantity: say it is unavailable or derived, do not invent it.

## Feasibility discipline

Use unrounded values for pass/fail decisions. Round only for presentation.

- compare the raw value against the raw threshold;
- if the raw value is below a minimum by any amount, it is still a fail;
- never let a rounded display value promote an infeasible candidate into the feasible set.

Do not eliminate downstream stages because an upstream screen used an unchecked derived value. If the
screening quantity depends on unit conversion or derived arithmetic, verify it first.

## Conclusion before perfection

Draft the full recommendation as soon as the decision evidence exists, then spend remaining budget
only on verification:

- if output budget is nearly exhausted, write the complete recommendation and constraint check
  first, even if some refinement is unfinished;
- a transcript that ends mid-reasoning with no conclusion is a failure, no matter how much evidence
  was collected.

## Final answer contract

The final answer must be compact and auditable. Include:

1. what is fixed and what is being chosen;
2. which sources were used: question only, retrieval, tool calls, or both;
3. the evaluated candidates or the verified boundary;
4. a complete constraint check;
5. the final recommendation and why alternatives were rejected;
6. any monitoring note, caveat, or limitation requested by the task.

When retrieval influenced interpretation, say so briefly. When the conclusion depends on tool output,
make that explicit.

## Trailer discipline

If the system must emit a structured score-points trailer, the trailer must describe what was
actually done: retrieved facts actually used, tool calls actually made, constraints actually checked,
final answer actually supported. Do not claim retrieval or tool use that did not happen.

## Anti-patterns

Avoid these failures:

- answering from retrieval without case-specific tool evidence for a numerical task;
- retrieving before deciding whether framing or interface clarity is actually missing;
- letting retrieval silently introduce new assumptions or candidate values;
- comparing `kg/s` directly against an `m3/h` requirement without conversion;
- omitting stated inputs because the tool has defaults, or changing multiple variables at once;
- probing probed points again, or refining beyond the required resolution while later stages are
  undone;
- probing between grid points or quoting an off-grid value once the adjacent grid pair is proven;
- re-simulating points dominated by an already-decided failing or passing point;
- spending the search budget on a non-limiting case while the binding case is unresolved;
- ignoring an internal contradiction between two derived values for the same quantity;
- using rounded display values for pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check, or ending mid-reasoning without a
  recommendation;
- overstating confidence when the environment did not actually provide the needed evidence.
