---
name: swro-watertap
description: Execute SWRO and WaterTAP calculations with fixed inputs preserved and a short bounded search for numeric treatment decisions.
---

# Fixed inputs

Before any tool call, copy all stated temperature, pressure, feed pH, recovery, composition, units, minerals, and other fixed values. Every call must reuse them unchanged.

Feed pH is a tool input; a returned-pH limit is only an output check. If feed pH or recovery is stated, explicitly pass `ph` and `water_recovery`; never accept defaults.

Use the relevant RO tool for membrane or plant performance and `equilibrate_feed` for chemistry boundaries. Skip tool-description and retrieval calls when the question already supplies the contract.
