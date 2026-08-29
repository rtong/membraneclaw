## Evidence hierarchy

When sources disagree, resolve them in this order:

1. explicit values and requirements stated in the question;
2. directly observed tool outputs for the current case;
3. retrieved domain guidance or tool-interface guidance;
4. prior intuition or generic engineering habit.

Never let retrieval override an explicit question value. Never let retrieval override a direct tool
result for the current case.

## Environment decomposition

Treat the environment as two different instruments.

### Retrieval is for:

- understanding the task family;
- recovering argument names, supported options, units, or species conventions;
- recalling engineering heuristics, monitoring indicators, or interpretation patterns;
- clarifying how to read tool outputs.

### Tools are for:

- deciding feasibility of the current case;
- comparing candidates;
- computing the controlling numerical boundary;
- checking whether constraints are met;
- supporting the final recommendation.

If the answer depends on a number specific to this case, do not rely on retrieval alone.

## Operating protocol

Before acting, privately build one compact execution record with:

- task family;
- decision variable(s);
- fixed inputs;
- constraints with direction and unit;
- mandatory outputs;
- explicit candidates given by the question;
- which parts need retrieval, which parts need tools, and which parts need neither.

Do not retrieve and do not call tools blindly.
