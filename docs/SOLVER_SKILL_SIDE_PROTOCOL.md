# Solver Skill six-case side experiment

Status: preregistered before execution on 2026-09-02.

## Research question

Does adding a task-solving Skill to an otherwise identical Qwen-3.5-9B + WaterTAP + Knowledge01 system produce a consistent engineering-score gain, and does a compact later Skill transfer better than a longer workflow Skill?

This is an exploratory side experiment. It is not the main Tools/RAG/Router result and it will not be used to tune another Solver Skill version after scores are observed.

## Fixed systems

| System ID | Remote preset | RAG | Skill | Temperature |
|---|---|---:|---|---:|
| `tools-rag-skill-control` | existing `tools-rag` | yes | none | 0.0 |
| `tools-rag-solver-v060` | new preset | yes | local source `swro-watertap@0.6.0`; remote clone ID `swro-watertap-v060` | 0.0 |
| `tools-rag-solver-v089` | new preset | yes | local source `swro-watertap@0.8.9`; remote clone ID `swro-watertap-v089` | 0.0 |

All three conditions must use the same Qwen-3.5-9B checkpoint, WaterTAP tool binding, Knowledge01 collection, shared solver prompt, output limit, finalizer policy, and Judge. The local run manifest records the immutable source Skill version and artifact hash. Separate remote Skill IDs are required because OpenWebUI cannot represent two simultaneously attached contents under one global Skill ID.

Version selection was made before execution:

- v0.6.0 represents the longer environment/workflow protocol;
- v0.8.9 represents the later compact numeric Solver protocol that was already frozen as preliminary development evidence;
- no intermediate or later version will be selected after observing these test scores.

## Fixed cases

1. `D1-1a-feasibility`
2. `D1-1c-vfd-minimum-stable-pressure-and-operating-window`
3. `D2-2b-04-red-tide-high-salinity-pump-limit-no-feasible-solution`
4. `D4-4a-capex-design-sensitivity`
5. `D5-5a-n01-recovery-limit`
6. `D6-6a-n05`

These six cases were selected before this Solver Skill run and were previously used as the stratified Stage-1 development sample. They cover feasibility, operating-window, infeasibility, economic sensitivity, scaling/recovery, and multi-step integrated analysis.

## OpenWebUI preparation

1. Clone the existing `tools-rag` preset twice so the base model, WaterTAP tools, and Knowledge01 binding remain identical.
2. Create two separate OpenWebUI Skills:
   - copy `skills/_historical/swro-watertap/v0.6.0/SKILL.md`, change only the frontmatter name to `swro-watertap-v060`, and attach it only to the v0.6.0 preset;
   - copy `skills/_historical/swro-watertap/v0.8.9/SKILL.md`, change only the frontmatter name to `swro-watertap-v089`, and attach it only to the v0.8.9 preset.
3. Do not attach either Skill to the existing `tools-rag` control.
4. Add the two exact preset IDs to `.env`:

```dotenv
OPENWEBUI_MODEL_TOOLS_RAG_SKILL_V060=tools-rag-solver-v060
OPENWEBUI_MODEL_TOOLS_RAG_SKILL_V089=tools-rag-solver-v089
```

If OpenWebUI returns different Skill IDs in the detailed probe, update `remote_skill_id` in `configs/systems.json` to the returned IDs before execution; do not disable the binding check.

## Probe

```powershell
python ae.py probe --benchmark-set d1_d6 --evaluation-profile solver_skill_side --details
```

Proceed only when all three expected presets are found, all use the same Knowledge ID, the control has no Skill, the two candidate presets expose the expected distinct Skill IDs, and `binding_errors` is empty.

## Systems stage

```powershell
python ae.py auto --benchmark-set d1_d6 --evaluation-profile solver_skill_side --run-id solver-skill-side-6-v1 --stage systems --require-complete-systems --case D1-1a-feasibility --case D1-1c-vfd-minimum-stable-pressure-and-operating-window --case D2-2b-04-red-tide-high-salinity-pump-limit-no-feasible-solution --case D4-4a-capex-design-sensitivity --case D5-5a-n01-recovery-limit --case D6-6a-n05
```

This stage requires 18 physical 9B solver responses. Re-run the same command after transient failures so successful cache entries are reused.

## Teacher, Judge, and report

After the completeness check reports 18/18, run the same command with `--stage all`. Existing successful system responses will be skipped and only missing downstream artifacts will run.

## Reporting rule

Report all three conditions even if neither Skill improves the mean. Include:

- six paired scores and mean differences;
- wins, ties, and losses versus control;
- native/recovered/final completion;
- tool-call count, latency, and Tool efficiency;
- step-level gains and losses;
- the fact that n=6 is an exploratory fixed side experiment.

Do not claim general Solver Skill harm or benefit from this sample alone. Its paper role is to test whether direct behavior/workflow instructions provide a stable enough signal to justify remaining in the main architecture.
