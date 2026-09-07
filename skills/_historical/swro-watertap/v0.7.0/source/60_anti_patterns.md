## Anti-patterns
- answering from retrieval without case-specific tool evidence for a numerical task;
- retrieving before deciding whether framing/interface clarity is actually missing;
- letting retrieval silently introduce new assumptions or candidate values;
- passing the plant RO area (6000 m2) into `simulate_ro` or the module area (60 m2) into
  `simulate_swro_system`, or retrying a failed call by changing unrelated inputs instead of fixing the scale;
- comparing `kg/s` against an `m3/h` requirement without conversion (use the factor table);
- omitting stated inputs because the tool has defaults, or changing multiple variables at once;
- treating 39.0 and 40.0 as adjacent at 0.1 g/L resolution, or quoting off-grid values;
- re-probing probed points, or refining beyond the required resolution while later stages are undone;
- re-simulating points dominated by an already-decided point, or searching after decisive evidence exists;
- spending search budget on a non-limiting case while the binding case is unresolved;
- ignoring an internal contradiction between two derived values of the same quantity;
- using rounded display values for pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check, or ending mid-reasoning without a recommendation;
- overstating confidence when the environment did not provide the needed evidence.
