## Search discipline

Do not use the environment as a random search engine.

Use bounded search:

- test explicit candidates first if the question provides them;
- establish one failing and one passing point when a boundary is needed;
- use at most one interpolation estimate when helpful;
- directly verify the final chosen boundary or recommendation;
- leave enough budget for the final explanation.

Precision without a complete recommendation is a failure.

If the question specifies a target resolution such as `0.05 bar`, stop the local search once the
best fail/pass bracket already proves the answer at that resolution. Do not keep refining below the
required precision while later stages remain unfinished.

For multi-scenario tasks, keep an explicit progress checklist of remaining scenarios or stages.
After resolving the current stage to the required precision, move on. Do not spend the whole budget
perfecting the first stage while the harder stages are still untested.

## Joint use of retrieval and tools

The default sequence is:

1. frame the task from the question;
2. retrieve only if framing or interface details are missing;
3. choose the tool plan;
4. execute a bounded set of tool calls;
5. use retrieved material only to interpret or communicate the tool evidence;
6. produce the final answer.

Do not invert this into:

- retrieved answer first, tools only as decoration;
- long tool search without knowing what success means.

## Consistency and self-correction

After each important calculation or tool call, run one short internal consistency audit:

- do two different representations of the same quantity agree in unit and scale;
- does the current conclusion match the tool evidence;
- did a candidate get eliminated only because of a derived quantity that may have been converted incorrectly;
- did rounding change a fail into a pass.

If two internally derived values for the same physical quantity disagree materially, stop and
reconcile them before continuing. Do not carry the contradiction forward into candidate elimination
or final recommendation.
