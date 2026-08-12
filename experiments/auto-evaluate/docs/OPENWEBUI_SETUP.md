# OpenWebUI setup checklist

Create three stable model presets in the already-deployed OpenWebUI instance.

## Baseline

- Display/model ID recorded in `OPENWEBUI_MODEL_BASELINE`.
- No WaterTAP tools available.
- No Knowledge attached.
- No SWRO Skill attached.

## Environment

- Display/model ID recorded in `OPENWEBUI_MODEL_ENVIRONMENT`.
- WaterTAP available.
- Attach the chosen Knowledge collection.
- No SWRO Skill attached.

## Environment-Skill

- Display/model ID recorded in `OPENWEBUI_MODEL_ENVIRONMENT_SKILL`.
- WaterTAP available.
- Attach exactly the same Knowledge collection and retrieval settings as Environment.
- Rebuild the runtime artifact first with
  `python skills/swro-watertap/v0.6.0/build_skill.py`.
- Replace the existing `swro-watertap` content with
  `skills/swro-watertap/v0.6.0/SKILL.md`, keeping the stable Skill ID
  `swro-watertap`, then attach it.

## Controls

- Do not paste benchmark questions, answers, or rubrics into presets, Knowledge, or Skills.
- Do not give Environment or Environment-Skill a stronger generic system prompt. `configs/systems.json` sends the same
  shared system message to all systems.
- Use the same temperature, top-p, output limit, and underlying Qwen model.
- Disable unrelated web, note, calendar, and personal-memory features for the evaluation user.
- Give the evaluation user read access to the Knowledge and Skill.
- Record an immutable Knowledge label in `OPENWEBUI_RAG_VERSION` after every corpus update.

## Programmatic access

Enable OpenWebUI API Keys, generate a dedicated evaluation key in `Settings -> Account`, and
place it only in `.env`. Run `python ae.py probe` before the pilot. The command checks that all
three configured model IDs are visible to that account, and that Baseline / Environment /
Environment-Skill follow the expected Knowledge and Skill bindings.
