---
name: swro-watertap
description: Use the SWRO environment correctly: frame the task, decide whether retrieval is needed, use retrieved material only for orientation, use tools for case-specific evidence, arbitrate conflicts by evidence strength, and produce an auditable final answer.
---

# SWRO environment-use protocol

This skill is not a bank of benchmark solutions. Its job is to control how the model uses the
environment.

The environment contains two external capabilities:

- retrieval over SWRO / WaterTAP knowledge;
- executable tools for case-specific computation and scaling analysis.

Use retrieval to understand the problem and tool interface. Use tools to decide the case. Do not
replace case-specific calculation with retrieved prose.

## Goal

Produce an answer that is:

- grounded in the question;
- supported by actual tool evidence for case-specific quantities;
- disciplined in how retrieval is used;
- explicit about constraints, uncertainty, and recommendation logic.

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

## When to retrieve

Run retrieval only when it adds missing structure. Typical reasons:

- the task wording is ambiguous and domain framing is needed;
- a tool argument name or supported option is uncertain;
- the question asks for an engineering interpretation, monitoring indicator, or mitigation logic;
- species, units, or scaling interpretation need confirmation.

Skip retrieval when the question already fully specifies the needed inputs and procedure.

After retrieval, keep only the few facts that change the execution plan. Do not copy large blocks of
retrieved text into the answer.

## Retrieval discipline

When retrieval is used:

- extract only actionable facts;
- distinguish definition-level facts from case-specific claims;
- treat retrieved numeric values as guidance only unless they are explicitly part of the question;
- do not manufacture hidden assumptions from retrieval;
- if retrieval is noisy or conflicting, fall back to question values plus tools.

Useful retrieval outputs are things like:

- which variable is worth searching;
- which tool is appropriate;
- what the tool expects for arguments and units;
- what engineering interpretation should accompany the computed result.

## Tool routing

Use tools by question type:

- use `simulate_ro` for membrane sizing, pressure selection, parameter comparison, and operating-point feasibility;
- use `analyze_ro_scaling` for concentrate chemistry, mineral SI, pressure-impact-on-scaling, and acid-dose selection;
- use `describe_ro_parameters` or `describe_reaktoro_options` only when interface details are truly uncertain.

Do not call an exploratory description tool if the needed argument contract is already clear.

## Tool argument discipline

For every computational tool call:

- pass every question-stated input explicitly;
- never rely on defaults for a question-stated parameter;
- change only the declared decision variable between candidate evaluations;
- immediately check echoed inputs or returned settings when available;
- if a fixed input drifted, repair it before reasoning from the result.

For `simulate_ro`, this usually means explicitly locking flow, salinity, temperature, pressure,
membrane area, A/B, permeate pressure, pressure drop, and transport options when the question gives
them.

For `analyze_ro_scaling`, this usually means explicitly locking the full composition, pressure,
temperature, membrane area, pH, minerals, and acid dose when they matter to the question.

## Unit-conversion discipline

Never compare quantities across different units without an explicit conversion step.

For permeate-water reporting, if the tool gives water mass flow in `kg/s` and the question asks for
`m3/h`, convert explicitly before screening candidates:

- for water, use `1 kg ~= 1 L`;
- therefore `kg/s -> m3/h` is approximately `x 3.6`.

If the tool also exposes another path to the same production quantity, such as flux-times-area or a
direct volumetric value, use it as a consistency check. Do not prune candidates until the units are
aligned.

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

## Anti-patterns

Avoid these failures:

- answering from retrieval without case-specific tool evidence when the task is numerical;
- using tools without first deciding whether retrieval is needed for framing or interface clarity;
- letting retrieval silently introduce new assumptions or candidate values;
- comparing `kg/s` directly against an `m3/h` requirement without conversion;
- omitting explicit inputs because the tool has defaults;
- changing multiple variables at once while pretending to compare one decision variable;
- micro-searching many points when one bracket and one verification would suffice;
- refining beyond the required resolution while later stages or scenarios are still undone;
- ignoring an internal contradiction between two derived values for the same quantity;
- using rounded display values to make pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check;
- overstating confidence when the environment did not actually provide the needed evidence.
