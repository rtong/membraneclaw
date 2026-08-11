# SWRO Auto Evaluate

本项目用于对 Datasets Harness 中的 SWRO/WaterTAP 工程题进行可复现评测。正式题库只保留一套源文件和一套标准化文件：

```text
benchmarks/
  Datasets Harness/   当前及以后新增的 Excel 源文件
  normalized/         由 Excel 生成的完整标准化 JSON
```

历史 concise、detailed、question_bank、旧 variants 和旧 runs 已移动到本地归档：

```text
archive/legacy-20260810/
```

归档目录被 Git 忽略，不参与正式评测，也不会被自动删除。

## 环境启动

在 Miniforge Prompt 中执行：

```text
conda activate scrapingpipe
cd /d F:\MembraneClaw\ScrapingPipe\auto-evaluate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Codex Teacher/Judge 使用本机 ChatGPT 账户登录，不使用 Platform API Key：

```text
codex login status
```

`.env` 只保存 OpenWebUI 地址、专用访问密钥、三个 preset ID 和 RAG 版本。

## 题库维护

新增题目时：

1. 把 Excel 放入 `benchmarks/Datasets Harness/<分类>/`。
2. 在 `configs/benchmarks.json` 的 `sources` 中增加对应的 `case_id`、相对路径和 `task_family`。
3. 运行以下命令重新生成并校验完整 normalized：

```text
python ae.py validate-benchmarks
```

Excel 中三张表严格分工：`题目_Q` 生成 `question_prompt`；`分步答案_A` 生成 Judge 专用的 `reference_answer`；`评价标准` 生成 Judge 专用的 `rubric`。待测系统和 Teacher 只收到 `question_prompt`。

## 查看和选择题目

查看当前全部 case ID：

```text
python ae.py list-benchmark-sets
python -c "import json; print(*[x['case_id'] for x in json.load(open('benchmarks/normalized/index.json', encoding='utf-8'))['cases']], sep='\n')"
```

不写 `--case` 时运行全部当前题目：

```text
python ae.py auto --run-id all-20260810-r1
```

只运行一道题：

```text
python ae.py auto --run-id smoke-d1-1a-r1 --case D1-1a-multi-condition-membrane-area-window
```

运行多道题时重复 `--case`：

```text
python ae.py auto --run-id smoke-two-r1 ^
  --case D1-1a-multi-condition-membrane-area-window ^
  --case D2-2a-feed-salinity-rise-response
```

`--case` 会先同步完整 `benchmarks/normalized`，再将所选 JSON 和 index 快照写入 `runs/<run-id>/benchmarks/`。三个待测系统、Teacher、Judge、报告和 Skill gate 都使用同一份选题快照。

## 建议的测试顺序

```text
python ae.py probe --details
python ae.py probe-chat --system baseline --case D1-1a-multi-condition-membrane-area-window
python ae.py run --run-id systems-one-r1 --case D1-1a-multi-condition-membrane-area-window
python ae.py auto --run-id e2e-one-r1 --case D1-1a-multi-condition-membrane-area-window
```

中断后使用相同 run ID 和相同参数重跑即可复用成功缓存。`--force` 重跑 OpenWebUI 响应，`--force-codex` 重跑 Teacher/Judge。

最终报告位于：

```text
runs/<run-id>/report.html
```
