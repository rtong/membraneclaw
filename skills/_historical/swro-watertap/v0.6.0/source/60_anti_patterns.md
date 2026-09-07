## Anti-patterns

Avoid these failures:

- answering from retrieval without case-specific tool evidence when the task is numerical;
- using tools without first deciding whether retrieval is needed for framing or interface clarity;
- letting retrieval silently introduce new assumptions or candidate values;
- comparing `kg/s` directly against an `m3/h` requirement without conversion;
- omitting explicit inputs because the tool has defaults;
- changing multiple variables at once while pretending to compare one decision variable;
- micro-searching many points when one bracket and one verification would suffice;
- refining beyond the required resolution while later stages or scenarios are still undone;
- ignoring an internal contradiction between two derived values for the same quantity;
- using rounded display values to make pass/fail decisions near a threshold;
- reporting a conclusion without a full constraint check;
- overstating confidence when the environment did not actually provide the needed evidence.
