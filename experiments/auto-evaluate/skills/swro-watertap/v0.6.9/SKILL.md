---
name: swro-watertap
description: Control how the model uses the SWRO environment — frame the task, retrieve only for orientation, use tools for case-specific evidence, arbitrate conflicts by evidence strength, and deliver an auditable final answer.
---

# SWRO environment-use protocol

Controls use of two external capabilities: retrieval over SWRO / WaterTAP knowledge, and
executable tools for case-specific computation. Retrieval is for understanding the problem and
tool interface; tools are for deciding the case. Never replace case-specific calculation with
retrieved prose. Produce answers grounded in the question, supported by actual tool evidence,
disciplined in retrieval, explicit about constraints, uncertainty, and recommendation logic.

## Evidence hierarchy

Resolve conflicts in order: (1) explicit values in the question, (2) direct tool outputs for the
current case, (3) retrieved domain/interface guidance, (4) prior intuition. Retrieval never
overrides a question value or a direct tool result.

## Operating protocol

Build one compact execution record before acting: task family, decision variables, fixed inputs,
constraints (direction + unit), mandatory outputs, explicit candidates, and which parts need
retrieval / tools / neither.

Copy every numeric threshold from the question verbatim (e.g. `>= 10,200 m3/d`) with its unit; do
not rephrase, redefine, re-derive, or round it. Every search target and final recommendation is
checked against these verbatim values.

Each question-stated value is the final deciding threshold as given. Do not add extra margin,
safety factor, or percentage, and do not re-derive it unless the question states that
transformation. A "margin"/"recommended"/"target" value is still the complete criterion itself.

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

## Unit-conversion discipline
Never compare quantities across different units without an explicit conversion step. Use the exact
factors below; do not improvise a factor from memory.

| Conversion | Factor | Example |
|---|---|---|
| `m3/s` -> `m3/d` | **x 86,400** | 0.12 m3/s = 10,368 m3/d |
| `m3/s` -> `m3/h` | x 3,600 | 0.12 m3/s = 432 m3/h |
| `m3/h` -> `m3/d` | x 24 | 432 m3/h = 10,368 m3/d |
| `kg/s` (water) -> `m3/h` | x 3.6 | 30 kg/s = 108 m3/h |
| `m3/d` -> `m3/s` | / 86,400 | 10,000 m3/d = 0.1157 m3/s |
| `MGD` -> `m3/d` | x 3,785.4 | 2.7 MGD = 10,221 m3/d |

Self-check after every conversion:
- day-to-second always uses **86,400** (60 x 60 x 24), never 3,600 alone or 3,600 x 1000;
- cross-check against the tool's own derived quantity when available: `product_flow_m3_s x 86,400`
  should equal the tool's product-flow metric, and `product_flow_m3_s / feed_flow_m3_s` should
  match the tool's recovery fraction;
- two representations disagreeing by a large factor (24x, 1000x, 41.7x) -> wrong factor; re-read
  the tool unit, do not proceed;
- do not prune candidates until units are aligned.

Do not carry a memorized approximate factor (e.g. "x 3.6" or "x 1000") across units. Each
conversion is read from the table, applied once, and cross-checked before use.

## Output-metric basis
Use the metric the question asks for, taken from the tool output that defines it. For a whole-plant
energy figure (e.g. daily energy consumed), use plant SEC x produced water volume
(`SEC kWh/m3` x product flow), not a pump-train power x time. Pump power, pumping power, and SEC
are different quantities: do not interchange them for plant-level energy. If the tool exposes more
than one energy-related value, state which one you used and why.

## Call budget (mandatory)
~16k-token context; every tool result echoes back. More calls = higher risk of running out before
the answer. Budget is a hard constraint.

- Reference budget + ≤2 slack; none stated -> **≤10 calls**.
- Multi-simulator: reference budgets assume **parallel** calls; never serialize independent
  simulations one batch could carry.
- Per boundary: **≤4 calls** (1 fail, 1 pass, ≤1 verification, ≤1 interpolation), then stop.
- Per screened case: **≤1 call**, except the controlling case.
- At budget−3 used: stop searching; write the complete recommendation and constraint check from
  existing evidence. No conclusion = failure; a complete answer on existing evidence beats an
  abandoned search.

## State ledger (mandatory)
Compact table updated after every call, read before the next: candidate, parameters, pass/fail
per constraint, bound or "resolved". Probe only points the ledger marks unknown and
information-relevant; never re-simulate a resolved point or region. Ledger = source of truth for
what remains.

## Bounded search
- test explicit candidates from the question first;
- one failing + one passing point per boundary; ≤1 interpolation;
- directly verify the final chosen boundary or recommendation;
- leave budget for the final explanation;
- if the question fixes a resolution (e.g. `0.05 bar`), stop once the fail/pass bracket proves the
  answer at that resolution; keep a progress checklist and move on.

## Dominance pruning (mandatory)
Bounded domain -> first evaluate the maximal corner (max of every decision variable).

- Corner fails any hard target -> record failure, declare infeasible, **STOP**. No interior scan.
- Corner passes -> it is the answer; verify only what the question asks.
- Simulate the question's named points in order, report each; never re-simulate a point dominated
  by an already-decided point.

## Decision-grain discipline (mandatory)
Question fixes a grid/resolution (e.g. `0.001`): report only grid-aligned values; prove the
boundary with exactly two adjacent grid points (one pass, one fail), then STOP; any off-grid probe
or quoted off-grid number is a defect.

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
misconverted derived value; did rounding flip a fail into a pass. Reconcile material
disagreements before carrying them into elimination or the recommendation.

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
Use unrounded values for pass/fail; round only for presentation.

- compare raw value against raw threshold;
- below a minimum by any amount = fail; a rounded display must never promote an infeasible
  candidate into the feasible set;
- if screening depends on unit conversion or derived arithmetic, verify it first; do not eliminate
  downstream stages because an upstream screen used an unchecked derived value.

## Conclusion before perfection
Draft the full recommendation as soon as decision evidence exists; spend remaining budget only on
verification. If output budget is nearly exhausted, write the complete recommendation and
constraint check first. Ending mid-reasoning with no conclusion is a failure, no matter how much
evidence was collected.

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
- comparing `kg/s` against an `m3/h` requirement without conversion (use the factor table);
- omitting stated inputs because the tool has defaults, or changing multiple variables at once;
- re-probing probed points, or refining beyond the required resolution while later stages are undone;
- probing off-grid or quoting off-grid values once the adjacent grid pair is proven;
- re-simulating points dominated by an already-decided point;
- spending search budget on a non-limiting case while the binding case is unresolved;
- ignoring an internal contradiction between two derived values of the same quantity;
- using rounded display values for pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check, or ending mid-reasoning without a
  recommendation;
- overstating confidence when the environment did not provide the needed evidence.
