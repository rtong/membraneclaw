# OpenWebUI setup checklist

Create three stable Qwen-3.5-9B presets for the paper-ready D1-D7 evaluation matrix.

| Preset | Environment variable | WaterTAP | Knowledge/RAG | SWRO Skill |
| --- | --- | --- | --- | --- |
| `baseline` | `OPENWEBUI_MODEL_BASELINE` | no | no | no |
| `tools` | `OPENWEBUI_MODEL_TOOLS` | yes | no | no |
| `tools-rag` | `OPENWEBUI_MODEL_TOOLS_RAG` | yes | yes | no |

Only `tools-rag` attaches the immutable Knowledge collection. `tools` and `tools-rag` use the same
Qwen weights and WaterTAP access; neither attaches the solver Skill. The adaptive virtual system routes
between these two physical presets. Its Router Skill is sent programmatically and is not attached to a preset.

Disable the former `swro_context_guard` Filter. It truncated transcripts but could not reliably control model-side tool calls, and it is not part of the revised experiment.

Do not attach a solver Skill to these presets. The Router Skill is read from the repository and sent only
in a separate short routing request.

## Controls

- Use the same Qwen weights, temperature, top-p, output limit, shared prompt, and WaterTAP access wherever the matrix says they are controlled.
- Leave temperature, top-p, max-tokens, and Thinking unset/default in the three OpenWebUI presets and in global Model Defaults. The evaluation client sends `temperature=0.2`, `top_p=0.9`, `max_tokens=2048`, and `enable_thinking=false`; an explicit preset value such as `max_tokens=8000` may override the request and invalidates the intended control.
- Leave the preset System Prompt empty. The versioned shared solver and finalizer prompts are supplied by `configs/systems.json`.
- The evaluation defaults to two concurrent independent OpenWebUI requests. Keep all three presets on the same base weights; lower `--system-concurrency` to 1 only for server-capacity diagnosis.
- Do not paste benchmark questions, answers, rubrics, or case-specific values into presets, Knowledge, or Skills.
- Disable unrelated web, note, calendar, and personal-memory features for the evaluation user.
- Give the evaluation user read access to the required Knowledge, Skill, and WaterTAP tool.
- Record an immutable Knowledge label in `OPENWEBUI_RAG_VERSION` after every corpus update.
- Keep `swro-watertap@0.8.9` off all three paper-ready presets. It is a frozen preliminary artifact.

## Programmatic access

Enable OpenWebUI API Keys, generate a dedicated evaluation key in `Settings -> Account`, and place it only in `.env`. Run `python ae.py probe --benchmark-set d1_d6 --details` before a pilot. The command checks the three physical preset IDs used by the four virtual conditions and verifies their bindings.

Model/preset presence alone is not a tool-health check. The evaluation runner uses the same shared
solver prompt across physical conditions, but rejects any `tools` or `tools-rag` answer with no
successful, observable WaterTAP/RO-chem call as `required_tool_call_missing`. A minimal manual capability probe
should visibly contain both ``🔧 ro-chem-simulate_ro(...)`` and a following ``↳`` result; a fluent
answer without those events is a failed Tools condition, not a successful tool-free fallback.
