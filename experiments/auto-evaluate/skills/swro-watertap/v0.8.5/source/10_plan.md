## Single-boundary hard stop

Apply this section only to one case with one numeric decision variable and a stated grid step `G`.

- Use at most **six total tool results**, including description, calculator, invalid, and computational results.
- Every tested candidate must equal an integer grid point `n * G`.
- Seek a baseline if requested, one informative coarse point, then adjacent boundary points.
- Infer the passing direction from observed results. If larger values move the controlling metric toward passing, the next candidate after the closest failing grid point `A` must be `A + G`, never `A - G`. Reverse this rule when smaller values move toward passing.
- If `A` fails and adjacent `A + G` passes, the answer is `A + G`: set `FINAL_NOW`.
- Values strictly between adjacent grid points are forbidden and cannot improve the requested answer.
- At `FINAL_NOW`, or immediately after result six, call no tool and perform no further search. Report the proved boundary or the best supported unresolved bracket.

Do not use a calculator for simple comparisons and do not repeat a point for confirmation.

For discrete candidates, test each under the same `TOOL_BASE`, reject failures, and rank survivors. For multi-case windows, refine only controlling cases; the six-result rule does not apply to these other task structures.
