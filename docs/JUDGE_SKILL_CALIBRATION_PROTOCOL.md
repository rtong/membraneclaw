# Evaluation Skill Judge校准协议

状态：在查看J0/J1新评分结果之前预注册；首轮运行目录为`runs/judge-calibration-12-v1`。

## 研究问题

在模型、回答、题目、参考答案、rubric、轨迹、输出schema和并发设置相同的情况下，独立的通用Evaluation Skill是否提高GPT-5.6 Judge的重复稳定性以及与人工专家的一致性？该实验不检验Solver是否变强。

## 固定样本

从不可变运行`d1-d6-full-20260828-v1`复制四道题的Baseline、Tools和Tools-RAG回答，共12份候选，不重新运行9B：

- `D1-1a-feasibility`
- `D1-1b-retrofit`
- `D5-5a-n01-recovery-limit`
- `D6-6c-01`

历史Judge分数只用于在查看本实验结果前覆盖低、中、高分以及原生/Finalizer完成类型，不作为专家真值。每题保留三个系统回答，以便计算题内排序一致性。

## 两个Judge条件

- J0：当前GPT-5.6 Judge任务。
- J1：相同任务加`swro-evaluation-judge@0.1.0`的冻结评价程序。

两组共享只描述可观察事实的程序摘要，包括响应、轨迹、工具调用计数和完成状态；该摘要不判断数值是否正确。J1的Skill不含题号、候选值、参考答案、Solver策略、系统名称或实验条件。Judge任务不读取Solver Skill，也不接收候选的真实系统身份。

每份候选在J0和J1下各运行两次独立Judge，共48个任务。执行顺序固定随机交错，所有任务使用`gpt-5.6-sol`、相同重试上限、输出schema和新会话。

## 指标和结论边界

在没有完整人工评分时，只报告：重复评分平均绝对差、±5/±10分一致率、failure-code重复Jaccard、J1-J0分数偏移、按隐藏系统分组的偏移和时延。此时不得声称J1更准确。

人工专家使用`expert_review_batch.jsonl`盲评全部12份候选，将结果按`expert_ratings.template.jsonl`填写为`expert_ratings.jsonl`。完成后再报告：分数MAE、±5/±10分一致率、题内排序一致率和failure-code Precision/Recall/F1。

J1进入后续新Solver实验的建议门槛在运行前固定为：

1. 专家分数MAE相对J0至少降低3分或相对降低20%；
2. 重复评分平均绝对差不高于J0；
3. failure-code F1不低于J0超过0.05；
4. J1-J0在三个隐藏系统组之间的平均偏移范围不超过5分；
5. 平均时延不超过J0的1.5倍；若token计量可用，平均token不超过1.25倍。

样本只有12份，这些门槛是是否继续使用J1的工程决策规则，不是统计显著性声明。若J1未通过，保留J0和全部负结果。无论结果如何，都不覆盖原D1-D6评分，也不把新旧Judge绝对分数直接合并。

## 执行顺序

```powershell
python ae.py judge-calibration --run-id judge-calibration-12-v1 --stage prepare
```

```powershell
python ae.py judge-calibration --run-id judge-calibration-12-v1 --stage judges --codex-model gpt-5.6-sol --judge-concurrency 4
```

```powershell
python ae.py judge-calibration --run-id judge-calibration-12-v1 --stage report
```

Judge阶段失败时原样重跑同一命令以复用成功缓存，不使用`--force-codex`。`prepare`只允许执行一次且拒绝覆盖已有目录。
