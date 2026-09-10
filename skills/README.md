# Skill index

本目录同时保留活动Skill、当前开发Skill和历史Solver Skill。是否“存在于目录”不等于是否部署
在论文系统中。

| Skill | 角色 | 状态 | 当前用途 |
|---|---|---|---|
| `swro-rag-router@0.1.2` | 判断是否需要RAG | active/frozen | 当前唯一论文系统中的Router Skill |
| `swro-parallel-compare@0.1.1` | P3有限候选并联求解步骤 | frozen validation | v3开发题产生+48 best-of-3信号；E31三题验证期间禁止修改 |
| `swro-evaluation-judge@0.1.0` | Judge评分程序提示 | completed side study | 校准近零差异；P3继续使用J0 |
| `swro-watertap@0.6.x-0.8.11` | 历史通用Solver Skill | historical | 位于 `_historical/swro-watertap/`，仅用于历史开发证据与方法审计 |

`swro-parallel-compare@0.1.0` 已由无效v2运行按哈希引用，因此保持不变；v0.1.1不改变求解
文字和工作流语义，只补全版本化manifest。v0.1.1已经被v3和E31配置按哈希冻结。
