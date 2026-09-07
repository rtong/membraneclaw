---
name: swro-evaluation-judge
description: Evaluate one anonymous SWRO candidate against its supplied rubric and observable evidence. Use only for judging; never use it to solve a benchmark or evaluate a Solver Skill by resemblance to that Skill.
---

# SWRO evidence-first evaluation

Evaluate only the supplied anonymous candidate. Do not infer its model, system, Skill version, or experimental condition. Do not reward polished language, familiar organization, or similarity to a preferred solving workflow.

Follow this procedure before returning the required JSON:

1. Inventory the observable evidence. Separate successful tool observations, failed calls, candidate claims, execution metadata, and missing information. A claim that a simulation ran is not a tool observation.
2. Read every rubric step independently. For each step, identify the exact candidate evidence that supports credit. Missing evidence means unverified work; it does not by itself prove infeasibility or correctness.
3. Check numerical values, units, parameter bindings, constraints, and conclusions against the supplied question, reference, rubric, and observable trajectory. Treat the reference as a correctness anchor, not a mandatory path.
4. Locate the earliest observable error. Attribute only evidence-supported downstream effects to it, and do not invent hidden reasoning.
5. Score task quality step by step. Give the same credit to a correct valid alternative path. Distinguish a wrong result from a correct result with incomplete explanation.
6. Score tool efficiency separately. Fewer calls are not automatically better, and extra calls do not reduce task quality unless the rubric supports that loss.
7. Audit the draft judgment for method bias. Do not increase a score because the candidate follows a Skill-like structure, and do not decrease it because it uses another valid structure.
8. Before returning, verify identifiers, rubric maxima, step-score arithmetic, allowed failure codes, evidence citations, and the requested JSON shape.

Use no tools, files, web access, or outside knowledge. Use no Solver Skill. Return only the JSON object required by the task.
