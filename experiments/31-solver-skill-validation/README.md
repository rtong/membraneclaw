# E31 - Frozen Solver Skill matching-family validation

这是当前主线实验。它检验在 `D6-6c-01` 开发题上产生正向信号的冻结短Skill，能否迁移到
三道未参与Skill编写的同族题。

## 固定设计

- Cases：`D6-6c-02`、`D6-6c-03`、`D6-6c-06`。
- C00：Tools，不加Solver Skill。
- C10：相同Tools，注入冻结的 `swro-parallel-compare@0.1.1`。
- 每题每条件3次，共18个有效Solver episode。
- 主指标：逐题best-of-3的C10-C00差值，再对3题取平均。

## 继续条件

平均逐题best-of-3为正、至少2/3题获胜，且执行诊断没有明显恶化，才进入C01/C11。

## 当前结论

`solver-skill-p3-validation-3case-v3`已完成，但主指标为-4.33，未通过继续门槛。审计同时发现
验证Prompt遗漏了Gold实际使用的进料流量，形成benchmark输入混杂。当前暂停扩展，先进行
版本化Prompt修正和三题重验；不得覆盖或删除v3。

- [FILES.md](FILES.md)
- [COMMANDS.md](COMMANDS.md)
- [RESULTS.md](RESULTS.md)
- [正式协议](../../docs/SOLVER_SKILL_VALIDATION_PROTOCOL.md)
