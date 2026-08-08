# Experiment protocol

## Research question

Measure the incremental contribution of retrieval and a structured SWRO-WaterTAP Skill to
the same Qwen-3.5-9B WaterTAP agent.

## Systems

| ID | WaterTAP | RAG | Skill |
| --- | --- | --- | --- |
| `agent` | yes | no | no |
| `agent-rag` | yes | yes | no |
| `agent-rag-skill` | yes | yes | `swro-watertap` |

The benchmark user message, shared system message, model weights, generation parameters,
WaterTAP implementation, and all fixed inputs must be identical across systems. Both RAG
systems must use the same Knowledge snapshot and retrieval settings. The only intended
difference between `agent-rag` and `agent-rag-skill` is the Skill.

## Primary contrasts

- RAG contribution: `Agent-RAG - Agent`.
- Skill contribution: `Agent-RAG-Skill - Agent-RAG`.
- Total augmentation: `Agent-RAG-Skill - Agent`.

## Outcomes

- 100-point benchmark score;
- score and normalized deficit by rubric step;
- failure-code distribution;
- constraint omissions and numerical errors;
- tool-call and argument errors when observable;
- response latency and available token usage;
- answer completeness and epistemic accuracy.

The report calls `max_score - awarded_score` a **step-level score deficit**, not model
training loss.

## Leakage control

- Evaluated systems receive only `题目_Q`.
- GPT-5.6 judge receives `题目_Q`, `分步答案_A`, and `评价标准`.
- RAG and Skill artifacts must not contain case-specific reference answers.
- Skill revisions may use development-set diagnostics only.
- Validation selects a Skill version; held-out test results do not feed back into it.

## Skill revision

Every change creates a new immutable version. Auto Evaluate may propose a patch from
aggregated failure evidence, but a human reviews it before deployment. Promote a new version
only after targeted and broad regression checks show no material degradation.

For the v0.2 development pilot, the executable promotion gate is intentionally strict:
Agent-RAG-Skill must score higher than Agent-RAG on every development case and in the mean,
with no TOOL_ARGUMENT or PARAMETER_EXTRACTION failures. This development gate is a debugging
criterion, not evidence of generalization; a held-out benchmark set is still required.

## GPT-5.6 role

GPT-5.6 supplies an upper-reference answer and rubric judge through the current Codex account,
not a paid API integration. Its self-scored comparison is labelled in the report and is not
the primary experimental result.
