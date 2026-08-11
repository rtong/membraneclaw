## Tool routing

Use tools by question type:

- use `simulate_ro` for membrane sizing, pressure selection, parameter comparison, and operating-point feasibility;
- use `analyze_ro_scaling` for concentrate chemistry, mineral SI, pressure-impact-on-scaling, and acid-dose selection;
- use `describe_ro_parameters` or `describe_reaktoro_options` only when interface details are truly uncertain.

Do not call an exploratory description tool if the needed argument contract is already clear.

## Tool argument discipline

For every computational tool call:

- pass every question-stated input explicitly;
- never rely on defaults for a question-stated parameter;
- change only the declared decision variable between candidate evaluations;
- immediately check echoed inputs or returned settings when available;
- if a fixed input drifted, repair it before reasoning from the result.

For `simulate_ro`, this usually means explicitly locking flow, salinity, temperature, pressure,
membrane area, A/B, permeate pressure, pressure drop, and transport options when the question gives
them.

For `analyze_ro_scaling`, this usually means explicitly locking the full composition, pressure,
temperature, membrane area, pH, minerals, and acid dose when they matter to the question.

## Unit-conversion discipline

Never compare quantities across different units without an explicit conversion step.

For permeate-water reporting, if the tool gives water mass flow in `kg/s` and the question asks for
`m3/h`, convert explicitly before screening candidates:

- for water, use `1 kg ~= 1 L`;
- therefore `kg/s -> m3/h` is approximately `x 3.6`.

If the tool also exposes another path to the same production quantity, such as flux-times-area or a
direct volumetric value, use it as a consistency check. Do not prune candidates until the units are
aligned.
