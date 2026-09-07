## Final response at `FINAL_NOW`

Use at most eight short natural-language lines: decision, adjacent evidence, required constraint checks, and one limitation statement. Then append the trailer immediately.

Copy this exact JSON structure. Replace values but preserve every comma, brace, bracket, key, and tag:

```text
[SCORE_POINTS_BEGIN]
{"task_type":"short_task_type","decision_variables":{},"fixed_inputs":{},"tool_calls":["tool@candidate"],"constraint_checks":{},"final_answer":"supported_answer"}
[SCORE_POINTS_END]
```

In `tool_calls`, use one short string per call actually made; never replace the list with `call_count`. Reproduce stated feed pH and recovery in `fixed_inputs`. Before sending, verify that the trailer parses as one JSON object and that every `[` has `]` and every `{` has `}`. After the closing tag, stop.
