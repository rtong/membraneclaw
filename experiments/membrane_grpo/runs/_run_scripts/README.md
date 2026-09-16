# The exact commands P11 and P12 were run with

Kept because a run is only reproducible if the command is, and these were typed
once on the training box rather than checked in first. `p11.sh` is the fresh and
shuffled-seed runs, `p11b.sh` the correct seed queued behind them, `probes.sh`
the temperature-1.0 label probes, `p12.sh` the 400-step run from the seeded
policy.

`sync.sh` is not an experiment. Artefacts are copied back to a laptop and
committed there, so the same bytes then sit on the training box as untracked
files and block the next `git pull` -- three times, before this existed. It
clears such a file only when it is byte-identical to what is being pulled, and
refuses to pull if anything actually differs.

The run logs themselves are not kept: they are progress-bar noise around numbers
that are already in `eval.jsonl`, `metrics.jsonl` and the `paired-*` evaluations.
