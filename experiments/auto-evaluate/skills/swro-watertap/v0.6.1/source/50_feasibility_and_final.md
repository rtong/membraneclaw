## Conflict resolution

If sources disagree:

- question vs retrieval: question wins;
- question vs tool echo on input values: repair the tool call to match the question;
- retrieval vs tool result on case behavior: tool result wins;
- retrieved heuristic vs hard constraint: hard constraint wins;
- tool output unavailable for a requested quantity: say it is unavailable or derived, do not invent it.

## Feasibility discipline

Use unrounded values for pass/fail decisions. Round only for presentation.

In boundary-finding tasks:

- compare the raw value against the raw threshold;
- if the raw value is below a minimum by any amount, it is still a fail;
- never let a rounded display value promote an infeasible candidate into the feasible set.

Do not eliminate downstream stages just because an upstream screen used an unchecked derived value.
If the screening quantity depends on unit conversion or derived arithmetic, verify it first.

## Final answer contract

The final answer should be compact and auditable. It must include:

1. what is fixed and what is being chosen;
2. which sources were used:
   question only, retrieval, tool calls, or both;
3. the evaluated candidates or the verified boundary;
4. a complete constraint check;
5. the final recommendation and why alternatives were rejected;
6. any monitoring note, caveat, or limitation requested by the task.

When retrieval influenced interpretation, say so briefly. When the conclusion depends on tool output,
make that explicit.

## Trailer discipline

If the system is required to emit a structured score-points trailer, the trailer must describe what
was actually done:

- retrieved facts actually used;
- tool calls actually made;
- constraints actually checked;
- final answer actually supported.

Do not claim retrieval use or tool use that did not happen.
