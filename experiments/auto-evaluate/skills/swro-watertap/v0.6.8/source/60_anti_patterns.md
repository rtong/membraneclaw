## Anti-patterns

Avoid:

- answering from retrieval without case-specific tool evidence for a numerical task;
- retrieving before deciding whether framing/interface clarity is actually missing;
- letting retrieval silently introduce new assumptions or candidate values;
- comparing `kg/s` against an `m3/h` requirement without conversion (use the factor table);
- omitting stated inputs because the tool has defaults, or changing multiple variables at once;
- re-probing probed points, or refining beyond the required resolution while later stages are undone;
- probing off-grid or quoting off-grid values once the adjacent grid pair is proven;
- re-simulating points dominated by an already-decided point;
- spending search budget on a non-limiting case while the binding case is unresolved;
- ignoring an internal contradiction between two derived values of the same quantity;
- using rounded display values for pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check, or ending mid-reasoning without a
  recommendation;
- overstating confidence when the environment did not provide the needed evidence.
