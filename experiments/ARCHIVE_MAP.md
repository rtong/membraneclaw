# Archive migration map

归档日期：2026-09-07。所有操作均为工作区内的整目录移动，没有删除内容。

| 原路径 | 新路径 | 分类 |
|---|---|---|
| `runs/060` | `archive/experiments/historical-solver-skill/runs/060` | 历史Solver Skill |
| `runs/089` | `archive/experiments/historical-solver-skill/runs/089` | 历史Solver Skill |
| `runs/skill-history-analysis-20260902` | `archive/experiments/historical-solver-skill/runs/skill-history-analysis-20260902` | 历史Solver Skill汇总 |
| `skills/swro-watertap` | `skills/_historical/swro-watertap` | 历史Skill资产 |
| `runs/verifier` | `archive/experiments/retired-prompt-state/runs/verifier` | 退休Prompt State资料 |
| `runs/smoke-tool-SigleRegression-20260826-v1` | `archive/experiments/smoke-runs/runs/smoke-tool-SigleRegression-20260826-v1` | smoke |
| `runs/smoke-tools-d1d5sample-20260828-v1` | `archive/experiments/smoke-runs/runs/smoke-tools-d1d5sample-20260828-v1` | smoke |
| `runs/solver-skill-p3-c00-c10-v1` | `archive/experiments/solver-skill-p3-invalid/runs/solver-skill-p3-c00-c10-v1` | 无效P3预检 |
| `runs/solver-skill-p3-c00-c10-v2` | `archive/experiments/solver-skill-p3-invalid/runs/solver-skill-p3-c00-c10-v2` | 无效P3工具表面 |

若需要恢复，只能按照本表反向整目录移动，并在恢复后重新执行链接检查和完整测试。
归档完成时的目录哈希记录在 [ARCHIVE_CHECKSUMS.json](ARCHIVE_CHECKSUMS.json)。
