## Final response

At `FINAL_NOW`, do not plan, summarize the search history, or call another tool. The first content in the next assistant message must be this complete trailer:

```text
[SCORE_POINTS_BEGIN]
{"task_type":"short_task_type","decision_variables":{},"fixed_inputs":{},"tool_calls":["tool@candidate"],"constraint_checks":{},"final_answer":"supported_answer"}
[SCORE_POINTS_END]
```

Replace values but preserve all six keys, tags, commas, brackets, and braces. Use one short string per actual call in `tool_calls`; do not use `call_count`. Include stated feed pH and recovery in `fixed_inputs`. Close the JSON and end tag before writing prose.

After the trailer, use at most four short lines: decision, adjacent evidence, remaining constraint checks, and one model limitation. Do not reproduce tool outputs, the question, a table, or the complete call history.
