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
