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
