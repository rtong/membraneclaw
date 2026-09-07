---
name: swro-watertap
description: Control how the model uses the SWRO environment — frame the task, retrieve only for orientation, use tools for case-specific evidence, arbitrate conflicts by evidence strength, and deliver an auditable final answer.
---

# SWRO environment-use protocol

Controls two external capabilities: retrieval over SWRO / WaterTAP knowledge, and executable tools
for case-specific computation. Retrieval is for understanding the problem and tool interface; tools
are for deciding the case. Never replace case-specific calculation with retrieved prose. Ground the
answer in the question, support it with actual tool evidence, stay disciplined in retrieval, and be
explicit about constraints, uncertainty, and recommendation logic.

## Evidence hierarchy
Resolve conflicts in order: (1) explicit question values, (2) direct tool outputs for the current
case, (3) retrieved domain/interface guidance, (4) prior intuition. Retrieval never overrides a
question value or a direct tool result.

## Operating protocol
Build one compact execution record before acting: task family, decision variables, fixed inputs,
constraints (direction + unit), mandatory outputs, explicit candidates, and which parts need
retrieval / tools / neither.

Copy every numeric threshold from the question verbatim (e.g. `>= 10,200 m3/d`) with its unit; do
not rephrase, redefine, re-derive, or round it. Every search target and final recommendation is
checked against these verbatim values. Do not add extra margin, safety factor, or percentage, and
do not re-derive a stated transformation. A "margin"/"recommended"/"target" value is still the
complete criterion itself.

If the answer depends on a case-specific number, produce it with a tool — do not rely on retrieval.

## When to retrieve
Retrieve only when it adds missing structure: ambiguous task wording, uncertain tool argument or
supported option, engineering interpretation / monitoring / mitigation asked, or species/units/
scaling confirmation needed. Skip retrieval when the question already specifies inputs and procedure.

## Retrieval discipline
When used: extract only actionable facts (do not copy large blocks into the answer); distinguish
definition-level facts from case-specific claims; treat retrieved numbers as guidance only unless
stated in the question; never manufacture hidden assumptions; if noisy or conflicting, fall back
to question values plus tools.

Useful retrieval outputs: which variable to search, which tool to use, what the tool expects for
arguments/units, and what interpretation should accompany the result.

## Tool routing
- `simulate_ro`: membrane sizing, pressure selection, parameter comparison, operating-point feasibility;
- `analyze_ro_scaling`: concentrate chemistry, mineral SI, pressure-impact-on-scaling, acid-dose selection;
- `describe_ro_parameters` / `describe_reaktoro_options`: only when the interface contract is truly uncertain.

## Tool argument discipline
For every computational call:
- pass every question-stated input explicitly; never rely on defaults for a question-stated parameter;
- change only the declared decision variable between candidate evaluations;
- check echoed inputs or returned settings when available; repair drift before reasoning from the result;
- do not call a description tool when the needed argument contract is already clear.

For `simulate_ro`, lock flow, salinity, temperature, pressure, area, A/B, permeate pressure,
pressure drop, and transport options when the question gives them. For `analyze_ro_scaling`, lock
the full composition, pressure, temperature, area, pH, minerals, and acid dose.

## Simulator scale mapping (mandatory)
The two simulators operate on different scales. Before the first call, write the mapping table from
the question and reuse it without re-deriving:

- `simulate_ro` = one membrane module: use the **module** membrane area (e.g. 60 m2, 70 m2) and a
  module-scale feed flow; never pass the whole-plant RO area.
- `simulate_swro_system` = whole plant: use the **plant** RO area (e.g. 6000 m2, 7000 m2) and the
  plant feed flow from the question row; never pass a module area.

If the question states only one area, infer which simulator it belongs to from its magnitude and
the simulator scope; if still ambiguous, call `describe_ro_parameters` once. A 100x area mismatch
or a failed initialize after the first call means the scale mapping is wrong: fix the mapping before
retrying, do not change unrelated inputs.

## Unit-conversion discipline
Never compare quantities across different units without an explicit conversion step. Use the exact
factors below; do not improvise a factor from memory.

| Conversion | Factor |
|---|---|
| `m3/s` -> `m3/d` | x 86,400 |
| `m3/s` -> `m3/h` | x 3,600 |
| `m3/h` -> `m3/d` | x 24 |
| `kg/s` (water) -> `m3/h` | x 3.6 |
| `m3/d` -> `m3/s` | / 86,400 |
| `MGD` -> `m3/d` | x 3,785.4 |

Self-check after every conversion: day-to-second is always 86,400 (60x60x24); cross-check against
the tool's own derived quantity when available; a large factor mismatch (24x, 1000x, 41.7x) means a
wrong factor — re-read the tool unit and do not proceed; do not prune candidates until units align.

## Output-metric basis
Use the metric the question asks for, taken from the tool output that defines it. For a whole-plant
energy figure, use plant SEC x produced water volume, not a pump-train power x time. Pump power,
pumping power, and SEC are different quantities: do not interchange them for plant-level energy. If
the tool exposes more than one energy-related value, state which one you used and why.

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

## Conflict resolution
- question vs retrieval -> question wins;
- question vs tool echo on inputs -> repair the tool call to match the question;
- retrieval vs tool result on case behavior -> tool result wins;
- retrieved heuristic vs hard constraint -> hard constraint wins;
- requested quantity unavailable -> say unavailable/derived, do not invent it.

## Feasibility discipline
Use unrounded values for pass/fail; round only for presentation. Compare raw value against raw
threshold; below a minimum by any amount = fail; a rounded display must never promote an infeasible
candidate into the feasible set. If screening depends on unit conversion or derived arithmetic,
verify it first; do not eliminate downstream stages because an upstream screen used an unchecked
derived value.

## Conclusion before perfection
Draft the full recommendation as soon as decision evidence exists; spend remaining budget only on
verification. If output budget is nearly exhausted, write the complete recommendation and constraint
check first. Ending mid-reasoning with no conclusion is a failure, no matter how much evidence was
collected.

## Final answer contract
Compact and auditable: what is fixed and what is being chosen; sources used (question / retrieval /
tools / both); evaluated candidates or verified boundary; complete constraint check; final
recommendation and why alternatives were rejected; monitoring note, caveat, or limitation as
requested. Say briefly when retrieval influenced interpretation, and make explicit when the
conclusion depends on tool output.

## Trailer discipline
The score-points trailer must describe what was actually done: retrieved facts actually used, tool
calls actually made, constraints actually checked, final answer actually supported. Do not claim
retrieval or tool use that did not happen.

## Anti-patterns
- answering from retrieval without case-specific tool evidence for a numerical task;
- retrieving before deciding whether framing/interface clarity is actually missing;
- letting retrieval silently introduce new assumptions or candidate values;
- passing the plant RO area (6000 m2) into `simulate_ro` or the module area (60 m2) into
  `simulate_swro_system`, or retrying a failed call by changing unrelated inputs instead of fixing the scale;
- comparing `kg/s` against an `m3/h` requirement without conversion (use the factor table);
- omitting stated inputs because the tool has defaults, or changing multiple variables at once;
- treating 39.0 and 40.0 as adjacent at 0.1 g/L resolution, or quoting off-grid values;
- re-probing probed points, or refining beyond the required resolution while later stages are undone;
- re-simulating points dominated by an already-decided point, or searching after decisive evidence exists;
- spending search budget on a non-limiting case while the binding case is unresolved;
- ignoring an internal contradiction between two derived values of the same quantity;
- using rounded display values for pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check, or ending mid-reasoning without a recommendation;
- overstating confidence when the environment did not provide the needed evidence.
