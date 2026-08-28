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

Never compare quantities across different units without an explicit conversion step. When a
conversion is needed, use the exact factors below. Do not improvise a factor from memory;
use the table.

| Conversion | Factor | Example |
|---|---|---|
| `m3/s` -> `m3/d` | **x 86,400** | 0.12 m3/s = 10,368 m3/d |
| `m3/s` -> `m3/h` | x 3,600 | 0.12 m3/s = 432 m3/h |
| `m3/h` -> `m3/d` | x 24 | 432 m3/h = 10,368 m3/d |
| `kg/s` (water) -> `m3/h` | x 3.6 | 30 kg/s = 108 m3/h |
| `m3/d` -> `m3/s` | / 86,400 | 10,000 m3/d = 0.1157 m3/s |
| `MGD` -> `m3/d` | x 3,785.4 | 2.7 MGD = 10,221 m3/d |

Self-check after every conversion:

- day-to-second conversion always uses **86,400** (60 x 60 x 24), never 3,600 alone or
  3,600 x 1000;
- cross-check the result against the tool's own derived quantity when available: for a
  whole-plant run, `product_flow_m3_s x 86,400` should equal the tool's product-flow metric,
  and `product_flow_m3_s / feed_flow_m3_s` should match the tool's recovery fraction;
- if two representations of the same quantity disagree by a large factor (e.g. 24x, 1000x,
  or 41.7x), the conversion factor is wrong — re-read the tool unit, do not proceed;
- do not prune candidates until units are aligned.

Do not carry a memorized approximate factor (e.g. "x 3.6" or "x 1000") across units. Each
conversion is read from the table, applied once, and cross-checked before it is used in a
decision.

## Output-metric basis

Use the metric that the question asks for, taken from the tool output that defines it. In
particular, when the question requests a whole-plant energy figure (for example daily energy
consumed by the plant), use the plant specific energy consumption x produced water volume
(`SEC kWh/m3` x product flow), not a pump-train power x time. Pump power, pumping power, and SEC
are different quantities: do not interchange them when the question asks for plant-level energy.
If the tool exposes more than one energy-related value, state which one you used and why.