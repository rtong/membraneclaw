# E00 - Frozen D1-D6 paper mainline

- 状态：`frozen`
- Run：`runs/d1-d6-full-20260828-v1`
- 范围：117个D1-D6案例，Baseline、Tools、Tools+RAG及策略回放记录
- 论文作用：Tools相对Baseline的主证据；D1-D6上的RAG稳定性证据
- 完整冻结规则：`runs/d1-d6-full-20260828-v1/FROZEN_RUN.md`

此 run 不允许覆盖。D1-D6是自包含的R0集合，不能用它证明RAG对R2知识缺口或Router完整泛化
有效；这些结论必须等待真实D7。
