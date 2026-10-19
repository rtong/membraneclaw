# ELS — Evidence-linked Skills（证据关联技能）

**模型决定下一步，WaterTAP计算工程表现，ELS读取已有计算记录，完成换算、差值、比例和边界比较。最终方案仍由模型生成。**

这是当前 ELS 0.9 的独立代码审阅入口。它保留现有运行时行为，附上无需密钥和 WaterTAP 的合成示例；不替换旧评测流水线，不启动模型、Judge或物理实验。

## 从这里开始

在仓库根目录：

```sh
cd experiments/auto-evaluate
python examples/els/offline_demo.py
python -m unittest discover -s tests -p 'test_els*.py' -v
```

Python 3.11+，以上两条命令仅使用标准库。示例把工具返回值写在测试夹具中，目的是检查数据传递，不是模拟器或模型能力评测。

审阅顺序：

1. [方法、接口与接入说明](../../docs/els/DESIGN_CN.md)：谁决定、谁执行，程序如何影响任务。
2. [SKILL.md](SKILL.md)：实际提供给模型的英文说明和调用示例。
3. [runtime.py](runtime.py) → [_arithmetic.py](_arithmetic.py) → [_core.py](_core.py)：当前接口、运算和记录逻辑。
4. [离线示例](../../examples/els/offline_demo.py)和[测试](../../tests/test_els09.py)：运行后检查完整证据链。
5. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)：复制来源及打包时的哈希。

已完成的检查见[离线验证记录](../../docs/els/VALIDATION.md)。

## 三类能力的区别

| 条件 | 模型获得什么 | 谁推进任务 |
|---|---|---|
| Tools | 物理工具及接口说明 | 模型选候选、填参数、读返回、给结论 |
| Tools + IES | 在Tools基础上，提供专业示范提炼的文字步骤 | 模型阅读并执行步骤 |
| Tools + ELS | 在Tools基础上，提供当前技能说明、示例和两个记录处理函数 | 模型决定何时调用、比较什么、检查什么及何时结束 |

IES = Instruction-encoded Engineering Skills。ELS不是预先写好的整题求解脚本，也不代表必须先调用物理工具再调用ELS；已有记录时才能进行对应的数值处理，没有固定的调用次数或搜索网格。

## 程序具体做什么

| 入口 | 模型给出什么 | 程序返回什么 |
|---|---|---|
| 原生物理工具 | 模型选择的工具参数 | 工具结果及T记录；失败调用也保留 |
| `els_calculate` | T/D记录引用、运算类型；按需提供目标单位、倍率、第二条记录或数量 | 新D记录：数值、单位、币值年份和来源 |
| `els_check` | T/D记录、模型从公开题目提取的门槛与比较符号 | 每项通过、不通过或信息不足，附比较差值与来源 |
| `accept_final`（宿主调用） | 模型最终正文 | 原文及流程检查状态；缺证据时最多提示一次，不代写答案 |

`pass`只表示记录满足**模型声明的**门槛。门槛提取错误、比较基准错误、候选覆盖不足、正文矛盾和非最优方案，仍需原rubric及轨迹分析判断。`complete`是流程状态，不是得分，也不保证工程正确。

## 代码范围

- 三个运行时文件与`SKILL.md`逐字节复制自当前本地ELS包；本PR不调整其求解行为。
- `engineering_skills/primitives.py`仅提取当前运行时使用的证据记录方法；方法实现保持一致，去掉了未使用的旧登记/发现接口。
- `adapters/swro.json`提供`simulate_swro_system`的公开参数及返回字段/单位映射；它不包含benchmark答案、候选或门槛。RO、水化学、D6和D7尚未作为本PR的完整适配提供。
- 当前仓库主线没有本地实验用的`model_client.py`及冻结运行包。本PR提供运行时、合成演示和明确的宿主接入契约，**没有把它接入现有`ae.py`/OpenWebUI流水线**。真实API执行仍需宿主实现工具循环、预算控制、快照和结果保存。
- 不提交`.env`、请求日志、评分结果、历史PES或本地冻结实验目录。原始实验的复现条件和失败记录继续保留在原档案中。
