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