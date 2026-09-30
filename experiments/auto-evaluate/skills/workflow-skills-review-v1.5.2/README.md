# Workflow Skills：代码审查包

这是 public-program@1.5.2 的选定源码快照，用于审查实现。SOURCE_MANIFEST.json 记录原路径和 SHA-256。原始源码未改写；附带的 Skill 文件路径版本为 v1.5.0，与执行代码版本分别管理。

## 实际流程

题目公开文本与任务类型 → 解析成输入和约束契约 → 按预算调用计算工具 → 换算指标并逐项核对约束 → 形成计算证据 → 模型根据证据写结论。

我们把两类易错步骤交给代码：保留已声明的固定工况，以及按明确口径换算结果和判断约束。搜索、调度等代码也包含在快照中，但不能据此声称每一部分都已独立证明提分。

## 从哪里看

| 审查内容 | 文件与函数 |
| --- | --- |
| 题目如何变成输入契约 | [public_program.py](src/auto_evaluate/public_program.py)：compile_public |
| 固定参数是否遗漏或被改动 | [fixed_input_guard.py](src/auto_evaluate/fixed_input_guard.py)：require_fixed_inputs |
| 换算口径和约束判断 | public_program.py：metric_values、checks |
| 执行和报告入口 | public_program.py：execute_public_workflow、render_report |
| 搜索契约的适用范围 | [public_search_contract.py](src/auto_evaluate/public_search_contract.py) |
| 工具执行依赖 | [program_tool.py](src/auto_evaluate/watertap_tools/program_tool.py)、[program_contract.py](src/auto_evaluate/watertap_tools/program_contract.py) |
| 模型如何使用证据 | [SKILL.md](skills/swro-public-workflow/v1.5.0/SKILL.md) |

## 离线核对

在这个文件夹打开终端，运行：

```powershell
python -X utf8 -B review_demo.py
```

只用 Python 标准库和本包源码；不调用 API、WaterTAP 或真实求解器。测试使用人工构造的数据，直接调用真实的检查和换算函数，并验证源码哈希。它验证这些代码分支，不验证模型最终答案或实验分数。

## Double check 的重点

1. 输入保护器只保护已经提取到契约中的字段，不能单独保证题目解析完整。请一起看 compile_public。
2. 解析器仍有任务族格式和适用边界，不是任意自然语言题目的通用解析器。搜索边界也不能自动视为题目的工程约束。
3. Qp_m3_h 的质量流量乘 3.6 是当前协议的等效体积口径，不是任意溶液密度下的精确体积换算；真实溶液体积另存。RO 回收率与系统回收率不混用。
4. 缺少指标、非有限值或违反阈值的检查不会通过；但模型最后写出的文字仍可能与证据不一致。
5. 历史 47 题用于失分分析和开发，不能当作未见题。4 道新合成工况只支持有限的迁移验证，不能直接宣称大量 OOD 验证。源码审查也不能独自证明没有数据泄漏。
6. 当前流程整体的分差不能归因于某一个组件，独立归因需要消融对照。

## 包的范围

包含选定执行源码、Skill 说明、哈希清单和离线检查。未包含原始题库、标准答案、评分数据、API 配置或完整物理工具依赖。因此本包可做实现审查，不是完整实验复现包。execute_public_workflow 的真实工具运行和 D6 分支需要原项目其余模块与环境。

上传建议：放在团队 GitHub 仓库的 workflow-skills-review/ 目录，分享本 README 链接。请勿将原项目整体 git add .；原项目还有大量无关修改和实验文件。本包不替代团队对代码归属、仓库权限及发布范围的确认。
