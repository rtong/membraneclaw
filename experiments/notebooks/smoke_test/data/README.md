# Seed data for notebook 11

Not the task's data. `experiments/membrane_grpo/data/` holds `train`/`dev`/`test`/
`holdout_shift`, and nothing here replaces or extends them — that tree is another project's
and is read-only from this lane.

These records exist for one question `10` left open: whether the two dead labels
(`organic_fouling`, `isolate_and_evaluate_replacement`) can be installed from data that is
*outside the task's distribution*, so that "the seed only installed the vocabulary" becomes
testable rather than asserted. Seeding from `train.jsonl` cannot answer it — a seed drawn from
the task's own training split is indistinguishable from supervised fine-tuning on the task.

| file | what it is |
| --- | --- |
| `seed_records_raw.jsonl` | exactly what the generating agent returned, kept verbatim as provenance. Records only, no labels. |
| `seed_cases.jsonl` | the subset that survives validation, with answers derived locally by `task.generate.truth_from_record` — the same function that grades every split. |

**The labels are never supplied by the generator.** A generated label that disagreed with the
decision table would be silently wrong and would train the policy on it; deriving them from the
record with the real grader makes that impossible.

The records are off-distribution by plant scale — small municipal, large industrial and
seawater trains, at flows, conductivities, pressures, temperatures and recoveries outside the
bands the existing splits occupy. The decision table reads percentage *changes*, not absolute
values, so it applies unchanged and the labels stay correct.
