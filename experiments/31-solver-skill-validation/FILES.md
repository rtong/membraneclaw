# E31 file map

- `docs/SOLVER_SKILL_VALIDATION_PROTOCOL.md`：冻结验证协议和继续条件。
- `configs/solver_skill_validation.json`：三题选择、C00/C10、3次采样和Skill哈希门禁。
- `skills/swro-parallel-compare/v0.1.1`：冻结Skill；禁止为验证题修改。
- `src/auto_evaluate/solver_skill_pilot.py`：prepare、systems、judges和逐题best-of-3汇总。
- `src/auto_evaluate/runner.py`：本地Prompt Skill注入、目标工具门禁和失败归档。
- `tests/test_solver_skill_pilot.py`：单题兼容和多题主指标测试。
- `runs/solver-skill-p3-c00-c10-v3`：父开发试点，已冻结。
- `runs/solver-skill-p3-validation-3case-v3`：已完成但存在Prompt输入混杂的验证run，必须冻结。
- 下一正式run：版本化修正公开进料输入后的三题重验；新run ID尚未分配。
