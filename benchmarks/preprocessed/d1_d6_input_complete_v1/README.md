# D1-D6 input-complete benchmark view v1

This directory defines a derived, preprocessed view of the 117 D1-D6 source
workbooks. It does **not** modify or copy the original Excel files under
`benchmarks/Datasets Harness/`.

## Data lineage

1. `source_config.json` discovers the original 117 workbooks.
2. Eight audited question-only overlays make a hidden, non-default simulator
   scale explicit.
3. `python ae.py import-benchmarks --benchmark-set d1_d6_input_complete_v1`
   writes the complete derived corpus to
   `benchmarks/normalized_d1_d6_input_complete_v1/`.

Only `question_prompt` may differ from the original `d1_d6` view. Gold
answers, task rubrics, tool-efficiency rubrics, source workbooks, and all
historical runs remain unchanged.

`change_manifest.json` is the human-readable audit record. The normalized JSON
also records an `overlay_id` and overlay SHA-256 under
`source.derived_overlay` for each amended case.

