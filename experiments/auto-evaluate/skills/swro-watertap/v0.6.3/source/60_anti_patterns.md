## Anti-patterns

Avoid these failures:

- answering from retrieval without case-specific tool evidence for a numerical task;
- retrieving before deciding whether framing or interface clarity is actually missing;
- letting retrieval silently introduce new assumptions or candidate values;
- comparing `kg/s` directly against an `m3/h` requirement without conversion;
- omitting stated inputs because the tool has defaults, or changing multiple variables at once;
- probing probed points again, or refining beyond the required resolution while later stages are
  undone;
- probing between grid points or quoting an off-grid value once the adjacent grid pair is proven;
- re-simulating points dominated by an already-decided failing or passing point;
- spending the search budget on a non-limiting case while the binding case is unresolved;
- ignoring an internal contradiction between two derived values for the same quantity;
- using rounded display values for pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check, or ending mid-reasoning without a
  recommendation;
- overstating confidence when the environment did not actually provide the needed evidence.