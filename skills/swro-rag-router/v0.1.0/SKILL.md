---
name: swro-rag-router
description: Decide whether an SWRO or WaterTAP benchmark needs external retrieved knowledge before tool-based solving.
---

# SWRO RAG routing policy

Classify the question only; do not solve it and do not invent missing facts. Choose `use_rag` only
when retrieved general knowledge could change the tool choice, parameter mapping, constraint
interpretation, or engineering decision. The expected benefit must justify adding retrieval text to a
limited context window.

Choose `skip_rag` when the question already supplies the simulator, inputs, units, constraints, and
required procedure, especially for numerical candidate comparison or boundary search. RAG cannot
replace simulation evidence and must not be used merely for reassurance, definitions already present
in the question, or a generic explanation.

Choose `use_rag` for a decision-changing gap such as an unclear tool contract, an unstated mapping
between engineering terminology and tool parameters, or domain/chemistry knowledge required to
interpret results. State one narrow retrieval need; never request benchmark answers or case-specific
target values.

Return exactly one JSON object and no Markdown or commentary:

{"action":"use_rag|skip_rag","reason_code":"MISSING_TOOL_CONTRACT|MISSING_PARAMETER_MAPPING|MISSING_DOMAIN_KNOWLEDGE|FULLY_SPECIFIED_NUMERIC_TASK|SIMULATION_EVIDENCE_DOMINATES","confidence":0.0,"retrieval_need":null}

For `use_rag`, `retrieval_need` must be a short targeted query. For `skip_rag`, it must be `null`.
