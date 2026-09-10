# Frozen Solver Skill matching-family validation

Protocol version: `teacher-distilled-solver-skill-validation@0.1.0`

## Question

Does the frozen `swro-parallel-compare@0.1.1` Skill transfer from its development case
`D6-6c-01` to matching parallel-comparison cases that were not used to write the Skill?

## Pre-registered cases

- `D6-6c-02`: membrane transport and product-water contract variation.
- `D6-6c-03`: cold-water and mild-ageing variation.
- `D6-6c-06`: ERD type and efficiency variation.

All three retain four candidate rows and the same independent `simulate_ro` and
`simulate_swro_system` branches. Multi-season aggregation case `D6-6c-07` is excluded because it
changes the task shape. The selection is frozen before collecting validation responses.

## Conditions and budget

- C00: the existing 9B Tools preset, no Solver Skill.
- C10: the same preset with the frozen local Skill injected into the prompt.
- Three independent generations per condition and case: 18 solver episodes total.
- RAG off; same generation settings, tool surface, recovery policy and frozen Judge.
- Both target simulator results must be observable for an episode to enter Judge scoring.

## Primary outcome

For each case independently, calculate `max(C10 r1..r3) - max(C00 r1..r3)`. The primary aggregate
is the mean of these three pre-registered case-level differences. Also report case wins/ties/losses,
all paired episode differences, completion, valid paths, tool-argument failures, calls, errors,
latency and tokens.

This is a matching-family transfer check. It is not held-out D7 generalization, hard-executor
evidence, or automatic Skill evolution.

## Decision rule

Continue to C01/C11 only if the frozen Skill has a positive mean case-level best-of-3 effect on at
least two of the three cases and does not introduce a material collapse in execution diagnostics.
Otherwise retain the result and revise the Skill/interface hypothesis before adding execution
control.
