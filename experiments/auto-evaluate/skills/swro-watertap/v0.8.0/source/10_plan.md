## Plan for information gain

Choose a strategy from the task structure rather than following a fixed call count.

- Named discrete candidates: evaluate each candidate once with identical locked inputs, reject every constraint violator, then rank only survivors.
- One-variable boundary: establish a fail/pass bracket, use observed metric change to move near the boundary, and verify adjacent grid points at the requested resolution.
- Multi-case common window: screen cases at a common point, identify the case and constraint controlling each side, refine only those controlling bounds, then intersect the bounds.
- Multiple simulators: call only the simulator responsible for each required quantity and join results through explicitly matched inputs and units.

After each result, update which decision or boundary it can change. Do not repeat an identical call, densely scan an already bracketed range, or refine a non-controlling constraint. Stop when the decision and every required constraint are supported. If evidence is insufficient, state exactly what remains unknown instead of inventing a result.
