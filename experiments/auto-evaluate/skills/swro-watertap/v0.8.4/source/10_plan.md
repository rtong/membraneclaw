## Single-boundary hard stop

Apply this section only to one case with one numeric decision variable and a stated grid step `G`.

- Use at most **six total tool results**, including description, calculator, invalid, and computational results.
- Every tested candidate must equal an integer grid point `n * G`.
- Seek a baseline if requested, one informative coarse point, then adjacent boundary points.
- If grid point `A` fails and the next grid point `A + G` passes, the answer is `A + G`: set `FINAL_NOW`.
- Values strictly between `A` and `A + G` are off-grid and forbidden. They cannot improve the requested answer.
- At `FINAL_NOW`, or immediately after result six, call no tool and perform no further search. Report the proved boundary or the best supported unresolved bracket.

Do not use a calculator for simple threshold comparisons. Do not repeat a point for confirmation.

For discrete candidates, test each under the same `TOOL_BASE`, reject failures, and rank survivors. For multi-case windows, refine only the cases controlling each boundary; the six-result rule does not apply to these other task structures.
