# E50 - Held-out D7 Router and RAG

- 状态：`planned`
- 规范：`docs/D7_BENCHMARK_SPEC.md`
- Router Skill：`skills/swro-rag-router/v0.1.2`
- 当前mock：只验证导入和路由合同，不是泛化证据

真实D7到达后先保留原工作簿，独立标注R0/R2及固定语料覆盖，再冻结选择。之后比较Never-RAG、
Always-RAG、Adaptive-RAG和Oracle，并报告路由准确性、regret、下游得分、可靠性和成本。
