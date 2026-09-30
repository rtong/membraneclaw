---
name: swro-public-workflow
description: Interpret a pre-executed SWRO workflow using its public task contract, observed constraints and numerical evidence. Applies to the local public-program harness; does not request additional simulations.
---

# Public-contract SWRO workflow

The harness compiles the public question and runs the calculation before your
answer. It supplies the contract and recorded evidence. Do not request additional
tools or pretend to execute the procedure yourself.

Read the declared decision variable, fixed inputs, tool, grid and criteria from
the public task and evidence contract. A task-family label is not a substitute
for these declarations. D3 may vary permeability, area, salinity, flow, pressure
or equipment efficiency. D5 single-variable tasks may vary recovery, a named
ion, blend fraction, temperature, dose or pressure. A pressure–recovery–scaling
task requires the coupled tool; equilibrium with an assumed recovery is not the
same calculation.

For D3 and D5-5a, the program checks the structured declarations and preserves
the fixed input record at each search call, including declared chemical coupling.
For other families, `not_checked_for_this_family` means this semantic layer has
not been applied. Even a checked contract does not certify every prose clause,
physical model accuracy, or global search completeness.

Use observed rows to distinguish a passing point, an adjacent fail/pass boundary,
and a complete feasible window. State which of these the task requested and which
the evidence actually establishes. Do not infer minimality from any passing point.
Search ranges inherited from engine defaults are exploration ranges, not public
equipment limits or proof of global infeasibility. A failed solver or incomplete
branch is missing evidence, not a failed engineering constraint.

Keep basic and recommended constraints distinct. Preserve strict versus inclusive
inequalities. For coupled Ba/Cl or source blending, use the recorded composition;
do not substitute recovery or linearly interpolate saturation indices. Keep
permeability in physical units and percentage points distinct from fractions.

Lead with the supported decision. Report the requested numerical outputs, the
binding constraint, the actual search scope and missing checks. Keep calculated
quantities distinct from raw solver outputs. Do not invent omitted values or
declare every requested deliverable complete merely because the program returned.

The fixed-input guard requires every declared invariant retained in a compiled
job before a physical call. Missing fields are rejected instead of silently
accepting tool defaults. This protects the compiled contract; it does not prove
that the text parser extracted every public requirement correctly. D2 now also
uses this call guard, without claiming the D3/D5 semantic parser covers D2.

The v1.2 local-stop routing is withdrawn: its two example tasks already had
correct stopping behavior in the original procedure responses. The standalone
search-control prototype is not an active loss-supported intervention. Existing
constraint-wise boundary refinement is retained; do not count its reuse as a
new skill or claim all recorded search failures have been fixed.

Declared variable arguments must also be present. Permission to vary a value
is not permission to omit it and silently use a tool default. The guard enforces
required field presence as well as invariant values.

For whole-plant results keep system_recovery_pct separate from ro_recovery_pct.
For coupled RO/chemistry results use the recorded recovery, rejection and flux;
they are actual model outputs, not assumed chemical recovery settings.

Structured D3/D5 single-variable tables reject unrecognized numeric inputs
marked fixed. This is not a general natural-language completeness guarantee.
Search calls must preserve nested required components and the declared coupling
at their search point; permission to change a composition is not permission to
drop its species or alter unrelated components.

Finite comparisons require an explicit declared candidate and scenario. Each call
must match their merged input values and candidate multiplicity, including every
nested composition value. Missing or ambiguous selections are rejected before execution.

The metric contract explicitly uses Qp_m3_h = total permeate mass flow in kg/s
multiplied by 3.6, the historical benchmark equivalent-volume convention at
1000 kg/m3. solution_volume_m3_h is the separately named actual solution-volume
estimate from salt mass and concentration. Never interchange these quantities.

Explicitly requested independent boundaries are searched on the full declared
domain. Other hard constraints are checked on the surviving interval and narrow
it when needed; they are never dropped. If independent constraints already
conflict, other checks may be marked not_searched_after_proven_conflict. This is
not a claim that those unchecked constraints pass. When no independent-boundary
selection is declared, the conservative full-domain behavior is retained. A suggested interval midpoint is not a simulated operating
point unless midpoint_verified is true. Do not claim minimum feed restoration
from a proposed seed or a single failed restoration trial. Incomplete evidence
and exhausted budgets must remain explicit in the final recommendation.
Equivalent adaptive candidate orders are allowed when the public task permits
them, provided prerequisites, branch pruning and mandatory checks are respected.

Neighboring operating states may predict trial locations only. All boundary
pass/fail evidence must come from the current operating state. Warm starts
require observed transitions for every independently requested constraint;
otherwise the search expands and falls back to its original declared range.
Report observed_search_range separately from the declared search_range; local
transition verification does not establish global monotonicity.

A certified feasible restoration interval is retained even if budget exhaustion
prevents the adjacent-lower-flow check. Report feasibility and minimum-restoration
certification separately. An incomplete run never becomes complete just because
a feasible interval was saved.

A minimum-feasible search uses a measured feasible anchor and refines the lower
intersection boundary while checking every hard constraint. Its upper sampled
point is not a maximum feasible operating pressure. Missing public equipment
margin or rating increment blocks the equipment-rating deliverable, even when
operating minima have been calculated. Never fill such inputs from gold answers.
