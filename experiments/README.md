# Experiment organization

本目录是实验导航层，不复制 `src/` 中的功能代码，也不复制 `runs/` 中的大型结果。每个实验
卡片把研究问题、配置、代码、Skill、命令和结果目录连接起来，从而可以从一个入口检查完整
实验版本。

## 目录结构

```text
CURRENT_EXPERIMENT.md              当前唯一开发入口
experiments/
  README.md                        本结构说明
  REGISTRY.md                      全部实验及状态
  registry.json                    机器可读注册表
  00-paper-mainline/               已冻结论文主结果
  10-repeatability/                D1-D6重复性审计
  20-judge-calibration/             Evaluation Skill Judge校准
  30-solver-skill-p3/              当前Solver Skill开发实验
  40-executable-skill-runtime/      可执行Skill离线原型
  50-d7-router-rag/                未来真实D7主线
  90-historical/                   历史、退休和无效实验索引
skills/
  README.md                        Skill角色和版本索引
runs/                              当前仍需直接访问的运行及README
archive/                           后续确认后存放历史与退休内容
```

## 状态词

| 状态 | 含义 |
|---|---|
| `frozen` | 正式证据，禁止覆盖 |
| `completed` | 实验已经完成，但不一定是主结果 |
| `paused` | 主动暂停，可按原协议恢复 |
| `blocked` | 需要外部条件恢复后才能继续 |
| `invalid` | 保留审计，但不能用于方法效果结论 |
| `planned` | 尚未运行 |
| `historical` | 早期探索或已经退休的方法 |

## 文件职责

- `configs/`：可执行实验选择与参数。
- `docs/`：正式协议和方法边界。
- `skills/`：版本化 Skill 资产。
- `src/`：共享流水线和实验实现。
- `tests/`：功能与实验合同测试。
- `runs/`：响应、轨迹、评分和报告。
- `experiments/`：把以上内容串成可读的实验版本。

## 整理原则

1. 冻结 run 不移动、不覆盖、不静默修复。
2. 新实验始终使用新 run ID。
3. 无效运行保留，并说明无效原因。
4. 实验卡片只链接真实文件，不复制功能代码。
5. 所有可执行命令保持单行，默认在已激活的 Miniforge `scrapingpipe` 环境运行。
6. 历史文件只有在完成引用扫描、迁移映射和恢复点后才移动到 `archive/`。
