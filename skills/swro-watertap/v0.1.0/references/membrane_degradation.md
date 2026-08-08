# Membrane-degradation recipe

Use when a membrane parameter changes with ageing and the objective is the largest feasible value.

1. Hold pressure, flow, temperature, membrane area, and unrelated membrane parameters fixed.
2. Derive all water-quality limits and select the strictest effective limit.
3. Simulate the normal membrane state and calculate its safety margin.
4. Find a higher passing point and a failing point for the degradation parameter.
5. Refine between them, checking every coupled quality constraint at each candidate.
6. Identify the first active constraint and quantify change relative to normal operation.
7. Set an observable warning threshold earlier than the mathematical failure boundary.

The recipe must not encode permeability or quality values from evaluation cases.

