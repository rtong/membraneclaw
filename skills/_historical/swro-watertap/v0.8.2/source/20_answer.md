## Report the supported decision

When `STOP_NOW` is reached or the available evidence is otherwise decisive, answer before doing optional exploration. Begin with the selected candidate, verified boundary, or infeasibility conclusion, then give only the controlling numerical evidence and hard-constraint checks.

Before emitting the answer, audit that:

- the reported value is on the requested grid or in the stated candidate set;
- every cited result is `ECHO_VALID` under the same `INPUT_LOCK`;
- every `OUTPUT_CONSTRAINTS` item is checked with the correct direction and unit;
- the fixed-input summary reproduces `INPUT_LOCK`, including stated feed pH and recovery;
- the tool-call record lists calls actually made and does not present an invalid call as evidence.

Keep the answer compact enough to finish. If response space is limited, shorten explanation rather than omitting the final decision, constraint checks, or required machine-readable trailer. Emit that trailer immediately after the natural-language answer and do not resume analysis afterward.
