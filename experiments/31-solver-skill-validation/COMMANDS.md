# E31 single-line commands

当前远程OpenWebUI服务异常，正式验证命令暂不启用。已删除两次受基础设施故障影响的run；服务恢复后先完成连接与工具调用预检，再分配新的run ID。

所有命令届时仍保持单行，并在F盘执行副本中、激活 Miniforge `scrapingpipe` 后运行。

## Input-complete v1 clean repeat

原始Excel目录：`benchmarks/Datasets Harness`

预处理定义目录：`benchmarks/preprocessed/d1_d6_input_complete_v1`

完整normalized目录：`benchmarks/normalized_d1_d6_input_complete_v1`

配置：`configs/solver_skill_validation_input_complete_v1.json`

```powershell
python ae.py import-benchmarks --benchmark-set d1_d6_input_complete_v1
```

```powershell
python ae.py validate-benchmarks --benchmark-set d1_d6_input_complete_v1
```

```powershell
python ae.py solver-skill-pilot --config configs/solver_skill_validation_input_complete_v1.json --run-id solver-skill-p3-validation-input-complete-3case-v1 --stage prepare
```

后续 `systems`、`judge`、`report` 阶段只使用这个新run ID，不复用旧run。

## Input-complete v1 remaining three cases

配置：`configs/solver_skill_validation_input_complete_extension_v1.json`

Cases：`D6-6c-04`、`D6-6c-05`、`D6-6c-07`

```powershell
python ae.py solver-skill-pilot --config configs/solver_skill_validation_input_complete_extension_v1.json --run-id solver-skill-p3-validation-input-complete-extension-3case-v1 --stage prepare
```

```powershell
python ae.py solver-skill-pilot --config configs/solver_skill_validation_input_complete_extension_v1.json --run-id solver-skill-p3-validation-input-complete-extension-3case-v1 --stage systems
```

```powershell
python ae.py solver-skill-pilot --config configs/solver_skill_validation_input_complete_extension_v1.json --run-id solver-skill-p3-validation-input-complete-extension-3case-v1 --stage judges --judge-concurrency 2
```

```powershell
python ae.py solver-skill-pilot --config configs/solver_skill_validation_input_complete_extension_v1.json --run-id solver-skill-p3-validation-input-complete-extension-3case-v1 --stage report
```
