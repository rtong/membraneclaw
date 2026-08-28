## Tool routing
- `simulate_ro`: membrane sizing, pressure selection, parameter comparison, operating-point feasibility;
- `analyze_ro_scaling`: concentrate chemistry, mineral SI, pressure-impact-on-scaling, acid-dose selection;
- `describe_ro_parameters` / `describe_reaktoro_options`: only when the interface contract is truly uncertain.

## Tool argument discipline
For every computational call:
- pass every question-stated input explicitly; never rely on defaults for a question-stated parameter;
- change only the declared decision variable between candidate evaluations;
- check echoed inputs or returned settings when available; repair drift before reasoning from the result;
- do not call a description tool when the needed argument contract is already clear.

For `simulate_ro`, lock flow, salinity, temperature, pressure, area, A/B, permeate pressure,
pressure drop, and transport options when the question gives them. For `analyze_ro_scaling`, lock
the full composition, pressure, temperature, area, pH, minerals, and acid dose.

## Simulator scale mapping (mandatory)
The two simulators operate on different scales. Before the first call, write the mapping table from
the question and reuse it without re-deriving:

- `simulate_ro` = one membrane module: use the **module** membrane area (e.g. 60 m2, 70 m2) and a
  module-scale feed flow; never pass the whole-plant RO area.
- `simulate_swro_system` = whole plant: use the **plant** RO area (e.g. 6000 m2, 7000 m2) and the
  plant feed flow from the question row; never pass a module area.

If the question states only one area, infer which simulator it belongs to from its magnitude and
the simulator scope; if still ambiguous, call `describe_ro_parameters` once. A 100x area mismatch
or a failed initialize after the first call means the scale mapping is wrong: fix the mapping before
retrying, do not change unrelated inputs.

## Unit-conversion discipline
Never compare quantities across different units without an explicit conversion step. Use the exact
factors below; do not improvise a factor from memory.

| Conversion | Factor |
|---|---|
| `m3/s` -> `m3/d` | x 86,400 |
| `m3/s` -> `m3/h` | x 3,600 |
| `m3/h` -> `m3/d` | x 24 |
| `kg/s` (water) -> `m3/h` | x 3.6 |
| `m3/d` -> `m3/s` | / 86,400 |
| `MGD` -> `m3/d` | x 3,785.4 |

Self-check after every conversion: day-to-second is always 86,400 (60x60x24); cross-check against
the tool's own derived quantity when available; a large factor mismatch (24x, 1000x, 41.7x) means a
wrong factor — re-read the tool unit and do not proceed; do not prune candidates until units align.

## Output-metric basis
Use the metric the question asks for, taken from the tool output that defines it. For a whole-plant
energy figure, use plant SEC x produced water volume, not a pump-train power x time. Pump power,
pumping power, and SEC are different quantities: do not interchange them for plant-level energy. If
the tool exposes more than one energy-related value, state which one you used and why.
