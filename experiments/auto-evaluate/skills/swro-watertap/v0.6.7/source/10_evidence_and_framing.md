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