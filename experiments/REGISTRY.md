# Experiment registry

最新开发入口见仓库根目录的 [CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md)。

| ID | 实验 | 状态 | 主要资产 | 结果 |
|---|---|---|---|---|
| E00 | D1-D6论文主结果 | `frozen` | `configs/evaluation_profiles.json` | `runs/d1-d6-full-20260828-v1` |
| E10 | D1-D6分层重复性 | `paused` | `configs/d1_d6_repeatability_26.json` | Rep-2/3完成，Rep-4不完整，Rep-5未运行 |
| E20 | Evaluation Skill Judge校准 | `completed` | `skills/swro-evaluation-judge/v0.1.0` | `runs/judge-calibration-12-v1` |
| E30 | P3 Teacher-distilled Solver Skill | `completed_development_signal` | `skills/swro-parallel-compare/v0.1.1` | v3 best-of-3：52→100；效果不稳定 |
| E31 | Solver Skill三题同族验证 | `completed_with_benchmark_confound` | `configs/solver_skill_validation.json` | v3主指标-4.33；Prompt遗漏Gold使用的进料流量，暂停扩展并准备修正重验 |
| E40 | 可执行Skill合同运行时 | `completed_offline_prototype` | `src/auto_evaluate/skill_runtime` | demo/preflight，不是线上增分证据 |
| E50 | 真实D7 Router/RAG | `planned` | `docs/D7_BENCHMARK_SPEC.md` | 等待真实D7 |
| H10 | 历史Solver Skill v0.6/v0.8 | `historical` | `skills/_historical/swro-watertap` | `archive/experiments/historical-solver-skill/runs` |
| H20 | Prompt State/Verifier相关历史 | `historical` | 已退出活动配置 | `archive/experiments/retired-prompt-state/runs/verifier` |

不要把 E30、E40 合并描述：E30测试短 Solver Skill 文本是否帮助9B；E40只证明确定性合同执行器
能够在离线 fixture 上执行工作流，不证明9B得分提升或 Skill自进化。
