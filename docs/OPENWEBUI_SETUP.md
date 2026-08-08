# OpenWebUI setup checklist

Create three stable model presets in the already-deployed OpenWebUI instance. All three use
the same MembraneClaw/WaterTAP-capable base connection.

## Agent

- Display/model ID recorded in `OPENWEBUI_MODEL_AGENT`.
- WaterTAP available.
- No Knowledge attached.
- No SWRO Skill attached.

## Agent-RAG

- Display/model ID recorded in `OPENWEBUI_MODEL_AGENT_RAG`.
- WaterTAP available.
- Attach the chosen Knowledge collection.
- No SWRO Skill attached.

## Agent-RAG-Skill

- Display/model ID recorded in `OPENWEBUI_MODEL_AGENT_RAG_SKILL`.
- WaterTAP available.
- Attach exactly the same Knowledge collection and retrieval settings as Agent-RAG.
- Replace the existing `swro-watertap` content with
  `skills/swro-watertap/v0.5.0/SKILL.md`, keeping the stable Skill ID
  `swro-watertap`, then attach it.

## Controls

- Do not paste benchmark questions, answers, or rubrics into presets, Knowledge, or Skills.
- Do not give Full a stronger generic system prompt. `configs/systems.json` sends the same
  shared system message to all systems.
- Use the same temperature, top-p, output limit, and underlying Qwen model.
- Disable unrelated web, note, calendar, and personal-memory features for the evaluation user.
- Give the evaluation user read access to the Knowledge and Skill.
- Record an immutable Knowledge label in `OPENWEBUI_RAG_VERSION` after every corpus update.

## Programmatic access

Enable OpenWebUI API Keys, generate a dedicated evaluation key in `Settings -> Account`, and
place it only in `.env`. Run `python ae.py probe` before the pilot. The command checks that all
three configured model IDs are visible to that account.
