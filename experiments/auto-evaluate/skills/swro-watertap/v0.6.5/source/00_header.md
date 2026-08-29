---
name: swro-watertap
description: Control how the model uses the SWRO environment — frame the task, retrieve only for orientation, use tools for case-specific evidence, arbitrate conflicts by evidence strength, and deliver an auditable final answer.
---

# SWRO environment-use protocol

This skill is not a bank of benchmark solutions. It controls how the model uses the two external
capabilities:

- retrieval over SWRO / WaterTAP knowledge;
- executable tools for case-specific computation and scaling analysis.

Retrieval is for understanding the problem and the tool interface. Tools are for deciding the case.
Never replace case-specific calculation with retrieved prose.

## Goal

Produce an answer that is grounded in the question, supported by actual tool evidence for
case-specific quantities, disciplined in retrieval use, and explicit about constraints, uncertainty,
and recommendation logic.