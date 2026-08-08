# Pressure-selection recipe

Use when pressure is the decision variable and the objective is the lowest feasible value.

1. Convert production requirements into any useful derived target, such as average flux.
2. Simulate a reasonable low-pressure point with every fixed input explicitly supplied.
3. Compute deficits and all constraint margins.
4. Obtain a feasible upper point without changing other inputs.
5. Refine inside the infeasible/feasible bracket using direct simulations.
6. Validate the lowest feasible candidate and compare it with a neighboring higher point.
7. Report the numerical boundary separately from a practical operating recommendation.

The recipe must not encode pressure values from evaluation cases.

