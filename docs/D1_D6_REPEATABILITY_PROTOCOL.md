# D1–D6分层重复性主实验

## 目的

本实验只回答一个主线问题：冻结D1–D6结果在重复生成时是否稳定。它不优化Prompt、不选择最高分参数，也不用于证明D7的R2或Router效果。

运行前冻结记录见`runs/d1-d6-repeatability-26-plan/PREREGISTRATION.md`。

已有全量运行`d1-d6-full-20260828-v1`作为Rep-1。新增Rep-2至Rep-5，每次在13个任务族各抽取2题，共26题，并运行Baseline、Tools和Tools-RAG三个物理系统。

## 固定设计

- 题目选择：`configs/d1_d6_repeatability_26.json`
- Profile：`d1_d6_repeatability`
- 每个新replicate：26题×3系统=78份系统回答
- 新增重复数：4
- 新增系统回答总数：312
- 生成参数：temperature 0.2、top-p 0.9、max tokens 2048、thinking关闭
- 恢复策略：保持`configs/systems.json`现状
- Teacher：不重新生成
- Adaptive：不重新运行；D1–D6的Adaptive属于本地policy replay，不是新的物理求解分支
- Judge：每个新回答独立评分一次

26题通过固定哈希规则选出，规则只使用版本字符串和case ID，不读取模型分数、错误、调用次数或延迟。题目内容直接读取Rep-1的冻结benchmark快照，而不是读取可能继续变化的当前题库。选择文件会被复制进每个run的benchmark快照并记录SHA-256；相同run ID恢复时不能换题。

## 第一步：远程配置预检

```powershell
python ae.py probe --benchmark-set d1_d6 --evaluation-profile d1_d6_repeatability --details
```

预期只列出：`baseline`、`tools`、`tools-rag`，并且`binding_errors`为空。确认三个别名仍指向与冻结实验相同的9B部署；如果远端模型检查点、WaterTAP、Knowledge01或系统Prompt已改变，不要开始运行。

## 第二步：运行Rep-2 systems

```powershell
python ae.py auto --benchmark-set d1_d6 --case-file configs/d1_d6_repeatability_26.json --evaluation-profile d1_d6_repeatability --run-id d1-d6-repeatability-26-r2 --stage systems --require-complete-systems
```

正常结束应显示：

- `expected: 78`
- `success: 78`
- `incomplete: 0`

如果有失败，原样重跑同一条命令；成功缓存会被复用。不要使用`--force`。

从Rep-3开始，补跑前被替换的失败记录会自动写入`response_attempts/<case>__<system>/attempt-NNN.json`，最终响应中的`collection_attempt`记录该题经过了第几次collection attempt。该功能只保存历史，不改变请求内容或求解行为。

Rep-2完成后、Rep-3开始前固定停止规则：每个replicate最多进行4个systems collection pass（包括首次运行）。若第4次后仍有失败，停止并报告，不能无限补跑到偶然成功。补充记录见`runs/d1-d6-repeatability-26-plan/AMENDMENT_20260905.md`。

## 第三步：评分Rep-2

systems完整后运行：

```powershell
python ae.py auto --benchmark-set d1_d6 --case-file configs/d1_d6_repeatability_26.json --evaluation-profile d1_d6_repeatability --run-id d1-d6-repeatability-26-r2 --stage judges --require-complete-systems --judge-concurrency 4
```

成功后应产生78条rating。该profile没有Teacher任务，因此不会消耗额度重复生成已有Teacher参考。

生成单次报告：

```powershell
python ae.py auto --benchmark-set d1_d6 --case-file configs/d1_d6_repeatability_26.json --evaluation-profile d1_d6_repeatability --run-id d1-d6-repeatability-26-r2 --stage report
```

## 第四步：依次完成Rep-3至Rep-5

重复上述systems、judges和report命令，只替换run ID：

- `d1-d6-repeatability-26-r3`
- `d1-d6-repeatability-26-r4`
- `d1-d6-repeatability-26-r5`

不要并行启动多个replicate。单个replicate内部继续使用默认`system-concurrency=2`和`judge-concurrency=4`。

## 第五步：汇总五次结果

```powershell
python ae.py repeatability-analysis --run-id d1-d6-full-20260828-v1 --run-id d1-d6-repeatability-26-r2 --run-id d1-d6-repeatability-26-r3 --run-id d1-d6-repeatability-26-r4 --run-id d1-d6-repeatability-26-r5
```

输出目录：`runs/d1-d6-repeatability-26-analysis/`

其中包括：

- `RESULTS_CN.md`：可读结论
- `analysis.json`：完整机器可读结果
- `run_summary.csv`：每次运行、每个系统的分数与可靠性
- `case_summary.csv`：每道题跨重复的均值、标准差和范围
- `comparison_summary.csv`：Tools增益和RAG效应的置信区间与方向稳定性

## 扩大到117题的判定规则

只有出现以下情况之一，才追加117题全量重复：

1. Tools相对Baseline的效应在不同replicate之间改变方向。
2. Tools与Tools-RAG排名频繁反转。
3. native/final完成率明显不稳定。
4. 26题上的case-level效应波动大到无法支持论文结论。

若Tools效应方向稳定、系统排序稳定且完成率波动有限，26题×5次已经完成重复性审计，不需要机械地再跑多次117题。

## 后续温度敏感性实验

完成并汇总上述固定参数重复实验后，再决定是否进行一次temperature=0.0单因素实验。该结果必须单独报告，不能与temperature=0.2的五次结果合并。不得同时修改top-p、max tokens、Prompt、工具上限或恢复策略。
