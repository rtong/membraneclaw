# D1-D6 benchmark audit

Audit date: 2026-08-23

## Research question used for the audit

The intended experiment asks whether Tools, RAG, Skill, and their interaction improve a fixed
16k-context 9B model and close part of the gap to GPT-5.6. The audit therefore distinguishes:

- **R0 — no retrieval opportunity:** the question supplies the tool, inputs, constraints, units,
  candidate space, and interpretation needed for the decision;
- **R1 — retrieval may improve explanation:** general knowledge could enrich a secondary
  explanation, but it is not needed to select the tool, map parameters, or make the decision;
- **R2 — decision-critical retrieval gap:** a fact or mapping absent from the question is required
  for a scored decision;
- **K0/K3 task:** fully specified numerical work or simulation evidence dominates;
- **Skill opportunity:** planning, argument preservation, pruning, cross-tool state lineage,
  stopping, and final constraint coverage.

The proposed adaptive RAG method needs a meaningful number of pre-labelled R2 cases. R0 cases are
useful for measuring whether the router avoids unnecessary retrieval, but cannot demonstrate a
positive knowledge contribution from RAG.

## Inventory and structural findings

The full normalized D1-D6 set contains **117 cases**:

| Domain | Cases | Dominant capability |
|---|---:|---|
| D1 | 22 | membrane candidate and boundary search |
| D2 | 26 | operating correction and constrained control search |
| D3 | 14 | whole-plant energy and equipment boundary search |
| D4 | 7 | whole-plant cost sensitivity and option comparison |
| D5 | 13 | equilibrium chemistry and pretreatment boundary search |
| D6 | 35 | multi-simulator joins, state lineage, and robust ranking |

The retired `skill_dev_20` subset contained **19**, not 20, cases: D1=3, D2=4, D3=3,
D4=2, D5=3, and D6=4. It was removed from the active benchmark registry after this audit.

Automated structural screening of the 117 question prompts found:

- 79 explicitly name at least one known simulator; this undercounts cases whose tool contract is
  expressed in a table or reference trajectory rather than the opening prose;
- 90 specify candidates, grids, resolution, or an allowed numerical domain;
- 91 contain a structured block of fixed numerical inputs;
- 95 directly list required tasks or answer deliverables;
- 23 explicitly state that a margin or rule is benchmark-specific or must not be generalized.

These are predominantly self-contained tool-use benchmarks, not missing-knowledge benchmarks.

## RAG opportunity by domain

| Domain | Current RAG opportunity | Audit finding |
|---|---|---|
| D1 | R0 | Simulator behavior and boundary evidence determine the answer. The question already supplies the search objective and constraints. |
| D2 | R0 | Operating variables, allowed moves, lexicographic objectives, and feasibility criteria are supplied. Retrieval cannot replace the simulations. |
| D3 | Mostly R0; limited R1 | Energy interpretation could be explained with general knowledge, but formulas, audit assumptions, simulator scope, and decision thresholds are normally supplied. |
| D4 | Mostly R0; limited R1 | Procurement and cost interpretation appear knowledge-oriented, but escalation factors, contingencies, currency basis, and option rules are explicitly provided. |
| D5a | R0 | The chemistry tool, species, minerals, SI criteria, variable, and boundary resolution are given. |
| D5b | R1 candidates, no clear R2 | Pretreatment selection could have required chemistry knowledge, but current prompts disclose the mechanism, candidate route, representation, hard constraints, and model limitations. |
| D6 | R0/K3 | The challenge is multi-model state lineage and combinatorial execution. External prose does not replace the required tool evidence. |

No clearly decision-critical R2 gap was identified in the current prompt designs. This means the
current collection can test **RAG avoidance**, but cannot cleanly test whether retrieved knowledge
improves decisions.

The local repository also contains no export of OpenWebUI `Knowledge01`: no document inventory,
content snapshot, chunking settings, embedding version, or corpus hash. Consequently, corpus-task
coverage and answer leakage cannot currently be audited or reproduced.

## Current 19-case development subset

| Case | RAG label | Tool/Skill opportunity | Context risk |
|---|---|---|---|
| D1-1a-feasibility | R0 / skip | multi-case screen, controlling boundaries, infeasibility stop | high |
| D1-1b | R0 / skip | candidate pruning and stress-test staging | high |
| D1-1c-vfd-minimum-stable-pressure-and-operating-window | R0 / skip | per-case pressure brackets and equipment-window join | high |
| D2-2a-feed-salinity-salt-shock | R0 / skip | smallest corrective pressure and full-constraint verification | medium |
| D2-2b-04-red-tide-high-salinity-pump-limit-no-feasible-solution | R0 / skip | informative corners and infeasibility proof | medium |
| D2-2b-constraint-conflict | R0 / skip | branch pruning and lexicographic coordinated control | medium |
| D2-2b-narrow-window | R0 / skip | adjacent two-sided pressure-window proof | medium |
| D3-3a-n01-salinity-intrusion | R0 / skip | adjacent TDS limits, units, energy reporting | medium |
| D3-3a-n02-membrane-aging | R0 / skip | permeability boundary and maintenance trigger | medium |
| D3-3b-n05-pressure-setpoint-energy | R0-R1 / normally skip | coupled capacity/SEC boundary and energy accounting | medium-high |
| D4-4a-capex-design-sensitivity | R0 / skip | one-at-a-time sensitivities and elasticity calculation | medium |
| D4-4a-n03-temperature-model-margin | R0-R1 / normally skip | model-boundary explanation; vendor TCF is already disclosed | medium-high |
| D5-5a-n01-recovery-limit | R0 / skip | controlling-mineral boundary | medium |
| D5-5a-n06-pressure-scaling | R0 / skip | RO-to-chemistry coupling with actual recovery | medium |
| D5-5b-n01-carbonate-acid-boundary | R1 candidate | route selection is scored, but the prompt already supplies route mechanisms and exclusions | low-medium |
| D6-6a-n01 | R0 / skip | three-model identity join and chemistry inheritance | extreme |
| D6-6a-n05 | R0 / skip | seasonal joins and one common acid dose | extreme |
| D6-6b-n07-multi-simulator-swro-pressure-scaling | R0 / skip | pressure-to-recovery-to-dose lineage | high |
| D6-6c-06 | R0 / skip | parallel membrane/plant candidate join | high |

The subset contains no defensible R2 case. A router that selects `skip_rag` for nearly every item is
therefore behaving consistently with the benchmark, but provides no evidence that RAG can add
knowledge when knowledge is genuinely missing.

## Context-window and execution feasibility

Reference call budgets could be extracted for 66 of 117 cases. Among those 66:

- 47 have an upper reference budget greater than 8 calls;
- 20 have an upper reference budget greater than 9 calls;
- 21 have a lower reference budget of at least 8 calls;
- all 22 D1 cases have an upper reference budget greater than 8 calls.

D6 is more severe. Every D6-6a task describes a reference path with 10 `simulate_ro`, 10
`simulate_swro_system`, and 10 `equilibrate_feed` calls. D6-6c-H07 requires 15 membrane plus 15
plant calls. Parallel dispatch reduces wall time but does not remove the returned observations from
the model context.

These tasks are not merely difficult for the 9B model: several reference trajectories are
structurally incompatible with a serial agent transcript and a 16k context. Context overflow should
therefore be reported as an end-to-end system outcome, but it cannot be interpreted solely as a
reasoning failure or solely as evidence that Skill/RAG is ineffective.

## Skill identifiability

The prompts frequently expose the same procedure that the Skill is meant to teach: screen at a
common point, identify the controlling constraint, form a bracket, verify adjacent points, prune
resolved candidates, and stop. This creates a ceiling effect for the Skill comparison.

The clean Skill intervention should add a reusable policy that is absent from the user question.
Current prompts often include both the engineering problem and a near-complete solution procedure,
so `Tools` already receives much of the policy. A high-quality evaluation should keep necessary
deliverables and safety constraints in the question while moving detailed search strategy into the
hidden rubric.

## Development/test dependence

The 19 development cases are representative anchors of highly templated families. A stopword-filtered
token-set screen found that, among the remaining 98 D1-D6 cases:

- 39 have maximum Jaccard similarity at least 0.50 to a development case;
- 20 are at least 0.60;
- 7 are at least 0.70.

This is a lexical diagnostic, not a semantic leakage proof, but it shows that the remaining D1-D6
variants are not a clean unseen test after repeated Skill optimization on the 19 anchors. D1-D6
should be treated as development data. Skill, router, RAG corpus, retrieval settings, and reward
weights must be frozen before D7.

## Consequences for the experiment

1. The current benchmark can support Tools and Skill studies, especially argument correctness,
   planning, pruning, state lineage, and completion.
2. It cannot yet support a strong claim that RAG improves decision quality because it contains no
   clear R2 stratum. The original RAG corpus is now versioned locally, but its operational and safety
   guidance does not match most benchmark-specific decision thresholds in D1-D6.
3. Always-on RAG is expected to be neutral or harmful on most current cases. Adaptive RAG can show
   avoidance benefit, but not positive retrieval benefit.
4. The reference call budgets and 16k deployment constraint must be reconciled before interpreting
   completion failures as model-quality differences.
5. Repeated optimization on D1-D6 followed by reporting on closely related D1-D6 variants would
   overstate generalization.

## Recommended next design actions

Do not change the existing normalized cases until a versioned benchmark policy is agreed. First:

1. Keep the exported OpenWebUI source documents and hashes under `rag_knowledge/original/`; record
   chunk size, overlap, top-k, threshold, and retrieval mode before the formal RAG run.
2. Add an explicit `rag_need` annotation to every evaluation case: R0, R1, or R2, with the missing
   fact and the supporting corpus document recorded before model runs.
3. Define natural D7 R2 cases whose operational or safety decision genuinely depends on a fact in
   the original corpus; do not manufacture D1-D6 facts that are absent from that corpus.
4. Keep natural D7 R0 controls with comparable numerical difficulty so the router is rewarded both
   for retrieving and for abstaining.
5. Create a question-only view that removes procedural search hints already represented in the
   hidden trajectory rubric; preserve deliverables, fixed inputs, constraints, and safety rules.
6. Separate context-feasible tasks from stress tasks. Do not aggregate 4-call D4 tasks and 30-call
   D6 tasks without reporting completion and context-overflow strata.
7. Treat D1-D6 as development, freeze all policies, and use D7 as the first untouched test set.
