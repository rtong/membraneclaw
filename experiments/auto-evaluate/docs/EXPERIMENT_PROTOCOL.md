# Experiment protocol

## Research question

Measure the incremental contribution of an execution environment and a structured
SWRO-WaterTAP Skill on the same Qwen-3.5-9B base model.

## Systems

| ID | Tools | RAG | Skill |
| --- | --- | --- | --- |
| `baseline` | no | no | no |
| `environment` | yes | yes | no |
| `environment-skill` | yes | yes | `swro-watertap` |

The benchmark user message, shared system message, model weights, generation parameters,
and all fixed inputs must be identical across systems. Environment and Environment-Skill
must use the same WaterTAP tool access, the same Knowledge snapshot, and the same retrieval
settings. The only intended difference between `environment` and `environment-skill` is the Skill.

## Primary contrasts

- Environment contribution: `Environment - Baseline`.
- Skill contribution: `Environment-Skill - Environment`.
- Total augmentation: `Environment-Skill - Baseline`.

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

For the current development pilot, the executable promotion gate is intentionally strict:
Environment-Skill must score higher than Environment on every development case and in the mean,
with no TOOL_ARGUMENT or PARAMETER_EXTRACTION failures. This development gate is a debugging
criterion, not evidence of generalization; a held-out benchmark set is still required.

## GPT-5.6 role and isolation

GPT-5.6 supplies an upper-reference answer and rubric judge through fresh Codex SDK tasks that
reuse the local ChatGPT-account login, not a Platform API key. Each teacher case and each
anonymous candidate rating runs in its own task. The judge sees only one candidate payload and
never receives `judge_mapping.json`. Every machine output is schema-validated before it enters
the report. Its self-scored comparison is labelled in the report and is not the primary
experimental result.
