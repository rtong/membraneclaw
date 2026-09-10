# E30 single-line commands

所有命令均为单行，在F盘执行副本中、激活 Miniforge `scrapingpipe` 后运行。以下命令是已完成
v3的历史执行记录；不要使用相同run ID重建或覆盖结果。当前命令见E31。

## 1. 工具表面验证

```powershell
python ae.py probe-chat --benchmark-set d1_d6 --system tools --case D6-6c-01 --stream --no-thinking --timeout 600 --max-tokens 2048
```

只有输出中出现成功的 `ro-chem-simulate_ro` 和 `ro-chem-simulate_swro_system` 结果才继续。

## 2. 创建新的v3快照

```powershell
python ae.py solver-skill-pilot --run-id solver-skill-p3-c00-c10-v3 --stage prepare
```

## 3. 运行6个Solver episode

```powershell
python ae.py solver-skill-pilot --run-id solver-skill-p3-c00-c10-v3 --stage systems
```

失败补采继续使用完全相同的单行命令和run ID，最多达到配置中的固定上限。

## 4. Judge

```powershell
python ae.py solver-skill-pilot --run-id solver-skill-p3-c00-c10-v3 --stage judges --codex-model gpt-5.6-sol --judge-concurrency 4
```

## 5. 报告

```powershell
python ae.py solver-skill-pilot --run-id solver-skill-p3-c00-c10-v3 --stage report
```
