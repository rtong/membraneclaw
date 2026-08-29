---
name: swro-rag-router
description: Decide whether an SWRO or WaterTAP task needs external retrieved knowledge before an otherwise identical tool-based solver runs.
---

# Route external knowledge only

Classify the information need; do not solve the task. Base the decision on the question text, not on
benchmark IDs, dataset names, source folders, or assumed category membership.

Choose `skip_rag` when the question already supplies the simulator contract, fixed inputs, units,
constraints, and decision procedure needed for a tool-grounded answer. Numerical difficulty alone is
not a reason to retrieve. RAG cannot replace simulation evidence.

Choose `use_rag` only when missing external knowledge could change the tool choice, parameter
mapping, constraint interpretation, safety boundary, or engineering decision. State one narrow
retrieval need. Do not request benchmark answers or case-specific target values.

Return exactly one JSON object and no Markdown or commentary:

{"action":"use_rag|skip_rag","reason_code":"MISSING_TOOL_CONTRACT|MISSING_PARAMETER_MAPPING|MISSING_DOMAIN_KNOWLEDGE|FULLY_SPECIFIED_NUMERIC_TASK|SIMULATION_EVIDENCE_DOMINATES","confidence":0.0,"retrieval_need":null}

For `use_rag`, `retrieval_need` must be a short targeted query. For `skip_rag`, it must be `null`.
