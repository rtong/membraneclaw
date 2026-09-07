## Report the supported decision

When `STOP_NOW` is reached or the available evidence is otherwise decisive, answer before doing any optional exploration. Begin with the selected candidate, verified boundary, or infeasibility conclusion, then give only the controlling numerical evidence and hard-constraint checks.

Before emitting the answer, run this compact audit:

- the reported value is on the requested grid or in the stated candidate set;
- every cited result came from a call whose echoed fixed inputs matched `LOCKED_ARGS`;
- every hard constraint is checked with its correct direction and unit;
- the fixed-input summary includes every stated value, including chemistry `ph` when specified;
- the tool-call record lists calls actually made, including corrected invalid calls when auditability requires them.

Keep the answer compact enough to finish. If response space is limited, shorten explanation rather than omitting the final decision, required constraint checks, or required machine-readable trailer. Emit that trailer immediately after the natural-language answer and do not resume analysis afterward.
