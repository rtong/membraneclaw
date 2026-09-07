# Solver / Verifier Skill paper-evidence audit

Audit date: 2026-09-02

## Conclusion

Historical Solver Skill runs are archived under `archive/experiments/historical-solver-skill/runs/060/` and `archive/experiments/historical-solver-skill/runs/089/`. They contain manifests, paired responses, Judge mappings, valid ratings, and matching immutable Skill hashes. They are usable as quantitative exploratory/development evidence, but not as one pooled formal ablation because the two generations used different prompts, sampling parameters, token budgets, benchmark views, and recovery policies. No complete Verifier Skill run is present.

## Solver Skill artifacts available

- Immutable `swro-watertap` artifacts are present from v0.6.0 through v0.8.11.
- Git history also records v0.1.0 through v0.5.0 and the former `agent-rag` versus `agent-rag-skill` promotion protocol.
- `docs/SKILL_ITERATION.md` records that `swro-watertap@0.8.9` was retired as preliminary evidence of family-specific improvement without demonstrated transfer.
- `archive/experiments/historical-solver-skill/runs/060/formal-all-20260813` provides a 10-case v0.6.0 paired study.
- `archive/experiments/historical-solver-skill/runs/060/pilot-new-benchmarks-20260814` provides a four-case v0.6.0 cross-family replication.
- `archive/experiments/historical-solver-skill/runs/089/skill-v089-cross-family-pilot` provides a four-case v0.8.9 de-scaffolded paired study, with a two-case repeated-extremes diagnostic.

Recovered evidence checks:

- every included rating file passes the current structural validator;
- within each selected run, control and candidate keep the same Tools/RAG availability and generation configuration while only the candidate mounts the recorded Skill;
- historical manifest Skill hashes match the current immutable v0.6.0 and v0.8.9 artifacts;
- repeated single-case regression, rollback, smoke, and sequential development runs are excluded from independent-sample means.

The derived audit and statistics are stored in `archive/experiments/historical-solver-skill/runs/skill-history-analysis-20260902/`. They support a methods narrative and an exploratory score figure. They do not support a direct v0.6.0-versus-v0.8.9 ranking or a claim of statistically proven Skill harm/benefit.

## Verifier Skill artifacts available

No run in the current workspace is identified as a Verifier Skill experiment. The `archive/experiments/retired-prompt-state/runs/verifier/` directory contains historical artifacts from a retired prompt-level method named `Tools + Compact Evidence State` (`tools-state`), not a Solver or Verifier Skill. Its executable system/profile configuration and README run entry were removed after the method was excluded from the paper experiment plan; the artifacts below are retained only for provenance.

Available scored `tools-state` evidence is limited to two independent runs of the same D1-1c case:

| Run | Tools | Tools + Compact State | Difference |
|---|---:|---:|---:|
| `stage1-state-d1c-v2-smoke` | 55 | 44 | -11 |
| `stage1-state-d1c-v3-continuation` | 51 | 51 | 0 |

The six-case run `stage1-state-6-v1` contains 11 successful system responses out of 12, but has no complete Judge ratings and includes one missing required-tool-call result. It is not a completed quantitative ablation.

These records show development instability, not a statistically supported Verifier result. They must not be labeled “Verifier Skill” in the paper.

## Minimum acceptable side experiment

If a stronger confirmatory claim is desired, use the preregistered small frozen side experiment instead of another optimization campaign:

1. Select the existing six stratified Stage-1 cases and freeze them before execution.
2. Compare `tools` against one final retired Solver Skill version with the same 9B checkpoint, WaterTAP preset, RAG state, generation settings, and Judge.
3. Do not revise the Skill after seeing these six test results.
4. Report paired case scores, completion mode, tool calls, context failures, and step-level changes.
5. Present the result as a bounded side experiment, not as the paper's main accuracy contribution.

A Verifier Skill experiment is optional. Given the previous optimization cost and lack of stable improvement, the paper can instead state that verifier-style development was explored but excluded from quantitative claims unless its original paired run is recovered.

## Recovered archival-data rule

Keep each recovered run directory intact with:

- `manifest.json` and `evaluation_profile.json`;
- frozen `benchmarks/`;
- paired `responses/`;
- `judge_mapping.json` and `ratings.jsonl`;
- `teacher_responses.jsonl` if used;
- the exact Solver/Verifier Skill artifact or its SHA-256 hash.

Do not merge recovered files into the frozen D1–D6 run. Cross-run summaries must remain derived artifacts, and the historical v0.6.0 and v0.8.9 means must remain in separate panels or table rows because their experimental regimes differ.
