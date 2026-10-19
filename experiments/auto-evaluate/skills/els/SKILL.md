---
name: evidence-linked-skills
description: Model-led engineering with automatic evidence records and two helpers.
---

# ELS 0.9

YOU solve the public task: choose candidates, physical parameters, tools, retries,
comparisons, requirements, next steps and stopping. Write the final answer yourself.
Never invent a simulation or treat missing evidence as a pass.

## Physical calculations
Call the physical tool directly with its ordinary arguments. No source quotes,
requirement-registration form or candidate-registration step is required. The
host saves each request and raw return automatically. Each actual attempt gets a
T record (T001, T002, ...), containing the input snapshot and returned metrics.
Each record is one calculation, not a claim that it represents a complete design.
Use the requested conditions; omitted parameters may use the tool's defaults.

## Evidence operations: els_calculate
Read values using source={record,metric}; never retype a computed value.
Metrics accept canonical names or native fields such as product_flow_m3_s.
- convert: supply target_unit. Unit constants need no question quote.
- scale: for a cost adjustment supply factor and optional to_basis. to_basis is
  ONLY a currency year, exactly USD_YYYY; it is not a physical unit or a note.
  For physical scaling also give target_unit; without it the unit is unconfirmed
  and the result CANNOT pass a check or feed another calculation. Prefer convert
  for unit changes. An explicit target and factor must agree dimensionally.
- difference/ratio/relative_difference: supply right={record,metric}.
  difference means source minus right; relative_difference divides by right.
  Recorded ratios support different dimensions and preserve currency bases.
- For ratio ONLY, an alternative to right is quantity plus quantity_unit:
  divide the recorded source by a positive quantity YOU derive from the public
  task, such as a unit count, area or mass. Supply one simple nonmonetary unit
  label (pump, element, m2, kg, etc.). Do not also supply right, factor,
  target_unit or to_basis. The result unit is derived automatically, for example
  recorded USD / a declared pump count returns USD/pump with its currency year.
  The quantity is retained as model-declared, not verified task truth. YOU choose
  its value and any required rounding; the helper never rounds or selects it.
  Example of format only: operation="ratio", quantity=3, quantity_unit="pump".
  For division by a recorded physical metric or compound unit use right instead.
- factor_note is optional explanatory text for ANY operation. It is not a unit
  or arithmetic parameter. Describing an operation does not certify its meaning.
The result gets a D record; reference it with metric="value".

## Requirement checks: els_check
Before delivering, check the relevant records against requirements YOU extract
from the public question. Supply records and constraints. Each constraint has
metric, operator, value, unit and, for money, an explicit currency basis.
For a T record use its returned/native metric name; for a D record use value.
Example of format only (not a task requirement):
{"records":["T001"],"constraints":[{"metric":"product_flow_m3_s",
"operator":">=","value":10000,"unit":"m3/day"}]}
Compatible units are converted automatically. A correctly converted threshold
can also be supplied; it need not occur verbatim in the question. Preserve the
original boundary precisely, especially near a limit. Do not substitute a
simulation value for a task threshold. Use evidence differences for comparisons,
not invented public limits. Unknown/missing evidence is returned as unknown.
A record with unconfirmed unit intent returns unknown, not pass. Recalculate
from the ORIGINAL source with correct units; do not convert an already-scaled
value again. Every check is retained, including mistaken interpretations. Pass
only means the recorded value meets YOUR declared boundary, not that the task
was fully or correctly understood. Use exact returned units for quantity ratios.

## Normal final answer
State your chosen design, comparison, evidence and remaining limitations in
ordinary prose. There is no els_finish tool. The host may remind you ONCE if
physical evidence or a current check is missing. If you stop, it still retains
your answer with its missing-check status. Text retention is not engineering
approval; you remain responsible for thresholds, coverage and accurate prose.

Use helpers when useful; there is no fixed physics/helper sequence or candidate
search grid. Do not read hidden answers, rubrics, historical runs or scores.
