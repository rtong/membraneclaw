# Experiment protocol

## Research question

Measure the contribution of WaterTAP tools and whether a Qwen-3.5-9B model, guided by a compact
Router Skill, can enable RAG only when external knowledge is required. The solver prompt and model
remain unchanged across the Tools, RAG-available, and Adaptive-RAG conditions.

## Paper-ready D1-D7 systems (current data: D1-D6)

| ID | Tools | RAG | Skill | Role |
| --- | --- | --- | --- | --- |
| `baseline` | no | no | no | evaluated 9B system |
| `tools` | yes | no | no | evaluated 9B system |
| `tools-rag` | yes | yes | no | evaluated 9B system |
| `tools-adaptive-rag` | yes | routed | Router only | evaluated 9B system |
| `gpt-5.6-teacher-general` | no | no | no | reference only |
| `gpt-5.6-teacher-tools` | WaterTAP MCP | no | no | reference only |

Within each benchmark set, the user message, solver system message, 9B model weights, generation
parameters, fixed inputs, and WaterTAP access must be identical across compared solver calls. The
adaptive system first runs the same 9B with `swro-rag-router@0.1.2`, then locally reuses the already
completed `tools` or `tools-rag` physical branch selected by that route. No second adaptive solver request
is sent and no solver Skill is mounted in the main matrix.

## Benchmark views

- D1-D6 are preregistered R0 cases with expected route `skip_rag`.
- D7 source workbooks will first be imported unchanged using the same four logical sheet roles as D1-D6. Their real structure will be audited before any R0/R2 annotation format is fixed.
- v0.8.9 is a frozen preliminary-study artifact, not part of the active evaluation configuration.
- The original RAG corpus is versioned under `rag_knowledge/original/`. D1-D6 do not receive synthetic information ablations because their task-specific thresholds are already disclosed and do not reliably match the corpus.
- After the D7 audit, knowledge-dependent cases will provide the positive RAG evaluation. Suitable R0 controls may come from D7 itself or a preregistered D1-D6 sample.

Views are deterministic derivatives. Original workbooks, gold answers, and rubrics remain unchanged.

## Primary controlled contrasts

- Tools contribution: `tools - baseline`.
- RAG-available contribution: `tools-rag - tools`, reported separately on D1-D6 and D7.
- Adaptive policy reward is evaluated by offline branch replay: `skip_rag`
  inherits the matched `tools` score and `use_rag` inherits the matched
  `tools-rag` score. The selected answer and observable trajectory are reused
  locally and are not submitted to Judge a second time.
- Adaptive routing over no RAG: `tools-adaptive-rag - tools`.
- Adaptive routing over RAG available: `tools-adaptive-rag - tools-rag`.
- Teacher tool contribution: `gpt-5.6-teacher-tools - gpt-5.6-teacher-general`.

## Outcomes and aggregation

- 100-point task-quality score and normalized deficit by rubric step;
- a separate 100-point tool-efficiency score when tools are available;
- failure-code distribution, trajectory loss attribution, propagation, and recovery;
- constraint omissions, numerical errors, tool-call and argument errors when observable;
- response latency, native completion status, context-recovery status, final completion status, and available token usage.
- adaptive-RAG route action, confidence, confusion matrix, Precision/Recall/F1, and counterfactual regret;
- paired mean gains with deterministic 95% bootstrap intervals, completion rate, latency, RAG activation, and execution-error counts.

For each system, the end-to-end score is the arithmetic mean over all rated cases, including the score awarded to actually observed failed or incomplete responses. The successful-response quality mean includes only `status=success` cases and must always be reported with completion rate. Tool-efficiency scores never enter the task-quality mean. The report's `Task loss` is evaluation score deficit (`100 - total_score`), not training loss.

## Isolation, concurrency, and reliability

Every benchmark/system answer and every anonymous candidate rating runs in an independent conversation. Sharing one conversation across cases or candidates is forbidden because it leaks history, anchors later judgments, and grows context. Independent tasks may run concurrently.

The default OpenWebUI system concurrency is 2 and the default Judge concurrency is 4. The three physical arms run before the short Router phase. `tools-adaptive-rag` then creates a local policy-replay record from the selected completed arm: `skip_rag` selects `tools`, while `use_rag` selects `tools-rag`. The replayed duplicate is not sent to Judge; its task-quality score is inherited from the selected physical arm. This rule is route-dependent and therefore supports both R0 and future D7 R2 cases without assuming that every case should skip retrieval.

The evaluated 9B requests disable explicit Thinking and must put the decision first in an answer of at most 700 words. If any evaluated system reaches `context_window_exceeded`, `incomplete_response`, or `output_budget_exhausted` after emitting a non-empty partial response, `context-reset-finalizer@0.2.0` makes exactly one fresh request to the same underlying 9B through the tool-free Baseline preset. The Finalizer receives the original question plus a dynamically bounded excerpt of the partial execution; the excerpt is capped at 12,000 characters and the combined message content at 24,000 characters. It cannot call Tools or RAG, must use only observed evidence, and returns at most 600 words. It is a generic termination/reliability policy, not a Solver Skill and not a source of domain knowledge. It never triggers for other error classes or empty partial responses, and it never recursively retries.

`tools` and `tools-rag` additionally require at least one successful observable WaterTAP/RO-chem
interaction. A final answer with no successful tool call/result is classified as
`required_tool_call_missing`; it is not context-reset recovered and cannot enter a formal score-comparable
run. This separates actual tool evidence from model-generated numerical claims.

Recovered records retain `native_status=error`, `native_error_type`, the partial response and original trajectory, while exposing `completion_mode=context_reset_finalizer` and the recovered final response. Formal reports must show native completion rate, recovery rate and final completion rate separately. A formal run uses `--require-complete-systems`; unrecovered failures stop the pipeline before Teacher/Judge, while successful one-pass recoveries count as final completions without being relabeled as native successes.

## Leakage control

- Evaluated systems receive only the Question sheet.
- GPT-5.6 Judge receives Question, gold/stepwise answer, trajectory rubric, tool-efficiency rubric, the anonymous candidate response, and its observable trajectory.
- RAG and Skill artifacts must not contain case-specific reference answers.
- Skill revisions may use development-set diagnostics only.
- Promotion selects a Skill version; held-out results do not feed back into it.

## Adaptive RAG routing

The router is a short first-stage request to the same 9B base model without tools or RAG. It emits a
validated `use_rag` or `skip_rag` action. A second independent request is sent to `tools-rag` or
`tools` only in an optional independent-replicate diagnostic; the default paper pipeline instead replays
the corresponding completed physical arm locally. D1-D6 use the preregistered R0 default. D7 routing labels will be created as derived metadata
only after inspecting the delivered files; the colleague-authored workbooks remain unchanged.
Benchmark IDs, source names, evidence locations, and labels are never shown to the router. Both fixed arms are evaluated so empirical
optimal action and routing regret can be computed without treating the manual label as score evidence.

The routing claim has a separate short-call ablation: the same 9B receives either a minimal zero-shot
classification contract or `swro-rag-router@0.1.2`. No solver, Tool, or RAG request follows in this
router-only experiment. Report valid-output rate, routing accuracy, confusion matrix, Precision,
Recall, F1, RAG activation rate, paired McNemar evidence, and the Skill minus zero-shot difference.
D1-D6 measure R0 false positives; the combined D1-D7 evaluation is required
for a two-action routing claim.

## Skill revision

`swro-watertap@0.8.9` is frozen and excluded from the paper-ready solver and active profiles. Router
changes create new immutable versions and may use only development-set misroutes expressed as
information-need failures; they must not encode benchmark families or case identifiers.

## GPT-5.6 role and isolation

GPT-5.6 supplies reference answers and rubric judgments through fresh Codex tasks that reuse the local ChatGPT-account login, not a Platform API key. The general Teacher is tool-free; the tools Teacher must make an observable WaterTAP MCP call. Each Judge sees one anonymous candidate and never receives `judge_mapping.json`. Every machine output is schema-validated before it enters the report.
