## Tool routing

Use tools by question type:

- `simulate_ro` for membrane sizing, pressure selection, parameter comparison, operating-point feasibility;
- `analyze_ro_scaling` for concentrate chemistry, mineral SI, pressure-impact-on-scaling, acid-dose selection;
- `describe_ro_parameters` or `describe_reaktoro_options` only when the interface contract is truly uncertain.

## Tool argument discipline

For every computational call:

- pass every question-stated input explicitly; never rely on defaults for a question-stated parameter;
- change only the declared decision variable between candidate evaluations;
- check echoed inputs or returned settings when available; repair drift before reasoning from the result;
- do not call a description tool when the needed argument contract is already clear.

For `simulate_ro`, explicitly lock flow, salinity, temperature, pressure, area, A/B, permeate
pressure, pressure drop, and transport options when the question gives them. For `analyze_ro_scaling`,
explicitly lock the full composition, pressure, temperature, area, pH, minerals, and acid dose.

## Unit-conversion discipline

Never compare quantities across different units without an explicit conversion step. If the tool
gives water mass flow in `kg/s` and the question asks for `m3/h`, convert explicitly before
screening: for water `1 kg ~= 1 L`, so `kg/s -> m3/h` is approximately `x 3.6`. If the tool exposes
another path to the same quantity (flux-times-area or a direct volumetric value), use it as a
consistency check. Do not prune candidates until units are aligned.

## Output-metric basis

Use the metric that the question asks for, taken from the tool output that defines it. In
particular, when the question requests a whole-plant energy figure (for example daily energy
consumed by the plant), use the plant specific energy consumption x produced water volume
(`SEC kWh/m3` x product flow), not a pump-train power x time. Pump power, pumping power, and SEC
are different quantities: do not interchange them when the question asks for plant-level energy.
If the tool exposes more than one energy-related value, state which one you used and why.