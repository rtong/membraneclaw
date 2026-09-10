# Current experiment

当前开发实验是 **冻结Solver Skill的D6_6c三题输入修正重验**。

## 当前状态

- 状态：`benchmark_amendment_required`
- 开发试点 `solver-skill-p3-c00-c10-v3` 已完成：best-of-3由52提高到100，但稳定性不足。
- 三题验证 `solver-skill-p3-validation-3case-v3` 已完成：主指标-4.33，配对均值+6.22，2/3题获胜。
- v3未通过预注册继续门槛，且公开Prompt遗漏了Gold使用的膜单元1 kg/s和工厂0.10 m3/s进料流量。
- 下一正式 run ID：完成版本化输入修正后分配；当前不进入C01/C11或扩大题量。
- 验证题在运行前固定为 `D6-6c-02`、`D6-6c-03`、`D6-6c-06`。

## 当前冻结定义

- 协议：`teacher-distilled-solver-skill-validation@0.1.0`
- Solver Skill：`swro-parallel-compare@0.1.1`
- Skill哈希与v3完全一致，验证期间禁止修改
- Cases：`D6-6c-02`、`D6-6c-03`、`D6-6c-06`
- C00：相同 Tools preset，不注入 Solver Skill
- C10：相同 Tools preset，注入冻结的本地 Solver Skill
- 每题每个条件：3个有效 episode，共18个
- 主指标：每题独立计算 `max(C10) - max(C00)`，再对3题取平均
- 有效工具门禁：必须同时观察到 `simulate_ro` 和 `simulate_swro_system` 的成功结果
- RAG：关闭
- Judge：J0冻结 Judge，两个条件共用

## 文件入口

- 实验卡片：[experiments/31-solver-skill-validation/README.md](experiments/31-solver-skill-validation/README.md)
- 单行命令：[experiments/31-solver-skill-validation/COMMANDS.md](experiments/31-solver-skill-validation/COMMANDS.md)
- 文件映射：[experiments/31-solver-skill-validation/FILES.md](experiments/31-solver-skill-validation/FILES.md)
- 运行状态：[experiments/31-solver-skill-validation/RESULTS.md](experiments/31-solver-skill-validation/RESULTS.md)

## 停止条件

v3已经冻结。由于主指标为负且执行诊断未改善，当前停止扩展；由于题目输入与Gold不一致，
不能把该结果简单归因为Skill无效。先版本化修正Prompt并重验三题，再重新应用原继续条件。

## 源码位置约定

`D:\PhD_Work\auto-evaluate` 是版本管理和文件整理的源码目录。
`F:\MembraneClaw\ScrapingPipe\auto-evaluate` 是 Miniforge 执行副本。代码、配置、Skill和文档
应从 D 盘同步到 F 盘，不应在两处独立修改；运行结果再按 run ID 同步回源码目录检查。
