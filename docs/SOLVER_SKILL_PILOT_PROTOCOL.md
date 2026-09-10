# Teacher-distilled compact Solver Skill pilot protocol

Protocol version: `teacher-distilled-solver-skill-pilot@0.2.0`

Status: completed and frozen as run `solver-skill-p3-c00-c10-v3`. The development result was
C00 best-of-3 52 versus C10 best-of-3 100; the paired mean effect was +9.33 with high variability.
The next experiment is the separately pre-registered matching-family validation in
`SOLVER_SKILL_VALIDATION_PROTOCOL.md`.

## Question

Does one frozen, compact procedure distilled from a high-quality GPT-5.6 tool trajectory improve the
fixed 9B Tools solver on the matching parallel-candidate task?

This pilot tests Skill content only. It does not test the hard executor, automatic Skill evolution,
Solver-Skill routing, RAG, or held-out generalization.

## Fixed development case

- `D6-6c-01` (P3): independent RO and plant branches joined by candidate identity.
- The case is a previously inspected development case, not an unseen test.
- P1 is excluded because the current `swro-parallel-compare` Skill explicitly does not cover boundary
  search. It requires a separate matching Skill before it can enter a comparable pilot.
- P2 remains excluded from numerical acceptance while its chemistry assumptions are unresolved.

## Conditions

| Condition | Model and Tools | Solver Skill | RAG | Execution enforcement |
|---|---|---|---|---|
| C00 | Existing `tools` preset | none | off | existing agent loop only |
| C10 | Same existing `tools` preset | `swro-parallel-compare@0.1.1`, injected from the frozen local artifact | off | existing agent loop only |

The system prompt, model ID, WaterTAP binding, generation parameters, output budget, recovery policy,
Judge, question, and rubric are otherwise identical. The local Skill text, version, tree hash, delivery
mode, and full request hash are stored in the run artifacts. No remote Solver Skill may be attached to
the shared `tools` preset.

## Scale and ordering

- Three independent generations per condition: six solver episodes in total.
- Systems are ordered by repeat as C00, C10 and executed with the same concurrency setting.
- Every failed or corrected collection attempt is preserved.
- A response is not accepted as a complete P3 episode unless successful observable results exist for
  both `simulate_ro` and `simulate_swro_system`. OpenWebUI names such as
  `ro-chem-simulate_ro` are accepted by canonical suffix. `calculator`, `http_get`, or another generic
  tool cannot satisfy this gate.
- Collection stops after four attempts for an episode. Exhausted episodes remain failed evidence.

## Scoring and outcomes

All six valid answers are anonymously scored once by the same frozen GPT-5.6 Judge procedure. The
recently tested Evaluation Skill is not used. Primary outcome:

`max(C10 r1..r3) - max(C00 r1..r3)` on the 100-point task-quality score. This is explicitly a
best-of-3 capability pilot under the same fixed three-generation budget for each condition; it is not
an estimate of mean performance or repeatability.

Also report:

- final and native completion;
- valid-path rate;
- `TOOL_ARGUMENT` failure rate;
- observable tool calls and tool errors;
- latency and tokens when available;
- paired per-repeat scores and their mean as diagnostics, while retaining every raw attempt.

Because there is one development case, no confidence interval or generalization claim is made.

## Continue and stop rule

Continue to a matching second task-family Skill and then C01/C11 implementation only if C10 provides a
useful development signal under the frozen best-of-3 criterion without materially worse engineering
execution diagnostics. This one-case development result still cannot establish transfer or
generalization.

If C10 has no score or process benefit, or causes repeated context/tool failures, retain the negative
result and revise the distillation/interface hypothesis before adding more prompt text. This run cannot
be described as Skill self-evolution; evolution requires a later frozen v0-versus-promoted-v* comparison
on validation cases not used for version selection.
