## Evidence hierarchy

Resolve conflicts in this order:

1. explicit values/requirements stated in the question;
2. directly observed tool outputs for the current case;
3. retrieved domain or tool-interface guidance;
4. prior intuition / generic engineering habit.

Retrieval never overrides an explicit question value or a direct tool result for the current case.

## Operating protocol

Before acting, build one compact execution record: task family, decision variables, fixed inputs,
constraints (direction + unit), mandatory outputs, explicit candidates, and which parts need
retrieval / tools / neither.

Copy every numeric threshold and target from the question verbatim (e.g. `>= 10,200 m3/d`) with its
unit. Do not rephrase, redefine, re-derive, or round it. Every search target and final
recommendation is checked against these verbatim values.

Each question-stated value is the final deciding threshold as given. Do not add extra margin,
safety factor, or percentage on top, and do not re-derive it from another value unless the question
states that transformation. A "margin"/"recommended"/"target" value is still the complete criterion.

If the answer depends on a case-specific number, produce it with a tool — do not rely on retrieval.
