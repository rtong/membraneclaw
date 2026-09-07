# Run directory index

`runs/` 只保留当前仍需直接访问的冻结、完成、暂停或离线原型运行。大型run内容由Git忽略；
正式状态由 `experiments/REGISTRY.md` 和run自身manifest共同记录。

## 当前保留

- `d1-d6-full-20260828-v1`：不可变论文主run。
- `d1-d6-repeatability-26-r2`、`r3`、`r4`：重复性运行。
- `d1-d6-repeatability-26-plan`：重复性预注册和暂停记录。
- `d1-d6-repeatability-26-analysis-interim-r2`：早期跨run分析。
- `judge-calibration-12-v1`：Evaluation Skill Judge校准。
- `skill-runtime-demo-20260906-001`、`skill-runtime-preflight-20260906-001`：离线可执行Skill原型。

P3 v1/v2、历史Solver Skill、退休State和smoke runs已经移至 `archive/experiments/`。完整旧新
路径见 `experiments/ARCHIVE_MAP.md`。

下一正式在线run预留为 `solver-skill-p3-c00-c10-v3`，只有远程WaterTAP工具恢复并通过probe后
才创建。
