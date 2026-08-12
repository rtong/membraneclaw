# Skill iteration protocol

## Current candidate

`swro-watertap@0.6.0` is the current development candidate. The OpenWebUI runtime artifact is the
self-contained `skills/swro-watertap/v0.6.0/SKILL.md`. The adjacent JSON files are the
machine-readable local source of truth for workflow, mappings, and failure prevention;
OpenWebUI does not read those JSON files.

To keep maintenance sane, the human-edited markdown source is now split into modules under
`skills/swro-watertap/v0.6.0/source/`. Rebuild the runtime artifact with:

```powershell
python skills/swro-watertap/v0.6.0/build_skill.py
```

Edit the source modules first, then regenerate `SKILL.md`, and only then copy the rebuilt runtime
file into OpenWebUI.

The current design keeps the same benchmark-answer isolation rule and uses a reusable execution protocol:

- compile the question into decision variables, fixed inputs, constraints, outputs, and explicit candidates;
- lock all stated arguments and reject silent tool defaults before every call;
- run a post-call audit with signed margins and a single justified next action;
- prioritize explicit question candidates over invented micro-search;
- separate theoretical boundary estimates from directly verified recommendations;
- emit one complete final answer with a full pass-fail table and monitoring guidance.

## Development gate

The three current benchmarks are development data. A candidate is promoted only when the
following command returns success:

```powershell
python ae.py skill-gate --run-id pilot-003
```

The default gate requires the candidate to beat Environment on every development case and in
the mean, with no TOOL_ARGUMENT or PARAMETER_EXTRACTION failures. If it fails, use the
step-level diagnoses to create a new immutable version; do not edit the already evaluated version.

## OpenWebUI deployment

1. Open the existing `swro-watertap` Skill in OpenWebUI.
2. Run `python skills/swro-watertap/v0.6.0/build_skill.py`.
3. Replace the OpenWebUI markdown with `skills/swro-watertap/v0.6.0/SKILL.md` and save.
4. Confirm that only Environment-Skill has this Skill attached.
5. Confirm Environment and Environment-Skill still use the same Knowledge collection.
6. Run `python ae.py probe --details` before starting a new run.

The run manifest records `swro-watertap@0.6.0` and a local artifact hash. Use a new run ID
instead of overwriting an earlier experiment run.
