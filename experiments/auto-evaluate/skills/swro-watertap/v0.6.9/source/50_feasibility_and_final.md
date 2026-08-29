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
