# Benchmark data

| 路径 | 作用 | 是否源数据 |
|---|---|---:|
| `Datasets Harness/D1` 至 `D6` | 当前117题四Sheet Excel工作簿 | 是 |
| `preprocessed/d1_d6_input_complete_v1/` | 8道题的只读覆盖规则与审计清单 | 否，派生定义 |
| `normalized/` | 从Excel生成的D1-D6执行JSON | 否，可重建 |
| `normalized_d1_d6_input_complete_v1/` | 原始Excel加8项输入补全后的117题执行JSON | 否，可重建 |
| `normalized_d7_mock*` | D7导入/Router诊断mock | 否，可重建 |
| `views/` | 派生的模型输入视图 | 否，可重建 |

当前源配置是 `configs/benchmarks.json`，自动发现 `benchmarks/Datasets Harness/D1-D6` 下的
`.xlsx` 文件，并忽略Excel临时锁文件 `~$*.xlsx`。真实D7尚未交付；任何mock都不能作为held-out
泛化证据。

原始Excel是事实来源。标准化JSON、模型视图和run内benchmark快照不得反向覆盖源工作簿。

`d1_d6_input_complete_v1` 是独立的派生 benchmark set。它只在导入阶段向8道题的
`question_prompt` 追加明确的模拟器尺度，不修改Gold、rubric或工具效率rubric，也不替换默认的
`d1_d6` active set。使用前显式运行：

`python ae.py import-benchmarks --benchmark-set d1_d6_input_complete_v1`
