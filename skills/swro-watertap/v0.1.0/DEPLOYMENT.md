# OpenWebUI deployment

1. Open `Workspace -> Skills -> New Skill` in the remote OpenWebUI instance.
2. Import or paste `SKILL.md`.
3. Use `swro-watertap` as the stable Skill ID and include `0.1.0` in the display name.
4. Attach it only to the `Agent-RAG-Skill` model preset.
5. Give the evaluation account read access to the Skill.
6. Keep the `Agent` and `Agent-RAG` presets unbound from this Skill.
7. Export the deployed Skill after any edit and record its hash in the experiment notes.

OpenWebUI executes tools separately; this Skill teaches the model how and when to use the
already-deployed WaterTAP tool. Do not paste benchmark answer sheets into the Skill.
