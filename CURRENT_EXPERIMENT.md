# Current experiment

当前开发实验是 **P3 Teacher-distilled Solver Skill C00/C10 pilot**。

## 当前状态

- 状态：`blocked_external_tool_binding`
- 原因：远程 OpenWebUI `tools` preset 在最近一次运行中只暴露了 `calculator` 和
  `http_get`，没有产生 `simulate_ro` 或 `simulate_swro_system` 的可观察结果。
- 当前没有可用于判断 Solver Skill 得分效果的有效运行。
- 等远程工具恢复后，下一正式 run ID 为 `solver-skill-p3-c00-c10-v3`。

## 当前冻结定义

- 协议：`teacher-distilled-solver-skill-pilot@0.2.0`
- Solver Skill：`swro-parallel-compare@0.1.1`
- Case：`D6-6c-01`
- C00：相同 Tools preset，不注入 Solver Skill
- C10：相同 Tools preset，注入冻结的本地 Solver Skill
- 每个条件：3个有效 episode
- 主指标：`best-of-3 task score`，即 `max(C10) - max(C00)`
- 有效工具门禁：必须同时观察到 `simulate_ro` 和 `simulate_swro_system` 的成功结果
- RAG：关闭
- Judge：J0冻结 Judge，两个条件共用

## 文件入口

- 实验卡片：[experiments/30-solver-skill-p3/README.md](experiments/30-solver-skill-p3/README.md)
- 单行命令：[experiments/30-solver-skill-p3/COMMANDS.md](experiments/30-solver-skill-p3/COMMANDS.md)
- 文件映射：[experiments/30-solver-skill-p3/FILES.md](experiments/30-solver-skill-p3/FILES.md)
- 运行状态：[experiments/30-solver-skill-p3/RESULTS.md](experiments/30-solver-skill-p3/RESULTS.md)

## 恢复条件

只有在远程 preset 能真实返回两个目标 WaterTAP 工具结果后，才创建 v3。不要继续 Judge
`solver-skill-p3-c00-c10-v2`，也不要覆盖 v1/v2。

## 源码位置约定

`D:\PhD_Work\auto-evaluate` 是版本管理和文件整理的源码目录。
`F:\MembraneClaw\ScrapingPipe\auto-evaluate` 是 Miniforge 执行副本。代码、配置、Skill和文档
应从 D 盘同步到 F 盘，不应在两处独立修改；运行结果再按 run ID 同步回源码目录检查。
