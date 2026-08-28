# Router Skill iteration protocol

## Active research boundary

`swro-watertap@0.8.9` is frozen as preliminary evidence that a task-specific solver Skill can improve
one benchmark family but may not transfer. It is not mounted in the paper-ready systems and is no longer
optimized. Its final artifact remains under `skills/swro-watertap/v0.8.9/`.

The active development Skill is `swro-rag-router@0.1.2`. It does not solve the engineering problem. It classifies
whether missing external knowledge could change the later tool-based decision, then returns `use_rag`
or `skip_rag`. Version 0.1.2 adds a decision-rule sufficiency gate after the natural short probe showed
that 0.1.1 treated known measurements as a fully specified task even when the governing external rule
was absent.

## Router-only ablation

Run the fixed D1-D6 pilot:

```bat
python ae.py router-eval --benchmark-set d1_d6 --run-id router-r0-pilot --pilot
```

The command sends each question twice to the same 9B model:

- `zero-shot`: minimal task and JSON contract only;
- `router-skill`: the frozen Router Skill instructions.

No WaterTAP tool, Knowledge collection, or solver request is used. The output
`runs/router-r0-pilot/router_summary.json` reports valid-response rate, route accuracy, activation rate,
latency, and the Skill-minus-zero-shot difference.

D1-D6 are R0 cases, so this pilot measures false RAG activation and output stability only. A complete
two-action routing result requires D7 R2 cases mixed with D1-D6 before evaluation.

## Reward-guided revision

This method is reward-guided prompt/Skill optimization, not reinforcement learning of model weights.
Only revise the Router when a development result reveals a transferable information-need error. Never
write case IDs, benchmark family names, source filenames, reference answers, or task-specific target
values into the Skill.

For end-to-end runs, execute:

```bat
python ae.py reward-analysis --run-id <run-id>
```

The command writes `reward_analysis.json` and `router_update_plan.json`. The latter contains misroute
evidence only; the retired Solver Skill promotion and source-file targeting loop has been removed.

Each accepted Router revision must use a new immutable version while keeping the 9B weights, solver
prompt, Tools preset, Knowledge corpus, and generation settings fixed.
