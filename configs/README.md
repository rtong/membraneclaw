# Configuration index

| 配置 | 作用 |
|---|---|
| `benchmark_sets.json` | benchmark集合注册表和当前默认集合 |
| `benchmarks.json` | 当前D1-D6 Excel发现规则 |
| `evaluation_profiles.json` | 系统矩阵、比较和执行profile |
| `systems.json` | OpenWebUI物理preset和生成/恢复参数 |
| `d1_d6_repeatability_26.json` | 固定26题重复性选择与replicate列表 |
| `judge_calibration_12.json` | 固定12份Judge校准候选 |
| `solver_skill_pilot.json` | 当前P3 C00/C10 v0.2.0协议配置 |
| `router_evaluation.json` | Router zero-shot/Skill独立评测 |
| `benchmarks_d7*.json` | 真实D7入口及mock诊断入口 |

运行正式实验前必须由 `prepare` 将配置、benchmark、Skill及哈希冻结到新的run目录。不要通过
修改已经创建的run快照来更新实验。
