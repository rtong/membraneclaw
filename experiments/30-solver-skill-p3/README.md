# E30 - P3 Teacher-distilled Solver Skill

这是当前开发实验。它只回答一个问题：在相同9B、相同WaterTAP工具、关闭RAG的条件下，加入
一份紧凑的教师蒸馏求解步骤，是否提高P3有限候选并联比较题的最佳可达得分。

## 条件

- C00：Tools，不加Solver Skill。
- C10：相同Tools，加入 `swro-parallel-compare@0.1.1`。
- 每个条件固定采集3份满足目标工具门禁的有效回答。
- 主结果是best-of-3；三轮均值和逐轮配对只作为诊断。

## 当前状态

`completed_development_signal`。v3正式结果为C00 best-of-3 52、C10 best-of-3 100，主效应
+48；但C10三次得分为20、50、100，说明效果不稳定。该结果已冻结，下一步进入E31同族验证。

## 不属于本实验的内容

- 不测试RAG或Router。
- 不测试硬执行器C01/C11。
- 不测试自动Skill进化。
- 不支持held-out泛化结论。

详细入口：

- [FILES.md](FILES.md)
- [COMMANDS.md](COMMANDS.md)
- [RESULTS.md](RESULTS.md)
- [正式协议](../../docs/SOLVER_SKILL_PILOT_PROTOCOL.md)
