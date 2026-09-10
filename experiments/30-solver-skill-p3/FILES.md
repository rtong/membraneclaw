# E30 file map

## 研究定义

- `docs/SOLVER_SKILL_PILOT_PROTOCOL.md`：冻结实验协议。
- `configs/solver_skill_pilot.json`：P3、C00/C10、3次采样、best-of-3和工具门禁。
- `scheme/可执行工程Skill最小原型规格_v0.1.md`：更大范围的设计来源；并非本轮硬执行器。

## Skill

- `skills/swro-parallel-compare/v0.1.0`：v2使用的历史冻结版本，不修改。
- `skills/swro-parallel-compare/v0.1.1`：v3和后续E31验证使用的冻结包。

## 功能代码

- `src/auto_evaluate/solver_skill_pilot.py`：prepare、systems、judges和best-of-3报告。
- `src/auto_evaluate/runner.py`：本地Prompt Skill注入、目标工具完整性门禁和失败归档。
- `src/auto_evaluate/cli.py`：`solver-skill-pilot` CLI入口和preset probe。
- `src/auto_evaluate/judge.py`：匿名Judge任务准备。

## 测试

- `tests/test_solver_skill_pilot.py`：实验冻结、Skill隔离和best-of-3分析。
- `tests/test_pipeline.py`：目标工具门禁、轨迹、失败归档和执行合同。
- `tests/test_evaluation_profiles.py`：C00/C10共享同一物理preset的probe合同。

## Run

- `archive/experiments/solver-skill-p3-invalid/runs/solver-skill-p3-c00-c10-v1`：无模型调用的错误元数据预检。
- `archive/experiments/solver-skill-p3-invalid/runs/solver-skill-p3-c00-c10-v2`：错误远程工具表面，无有效WaterTAP episode。
- `runs/solver-skill-p3-c00-c10-v3`：已完成并冻结的开发试点。
