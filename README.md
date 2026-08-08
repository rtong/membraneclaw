# SWRO Auto Evaluate

面向 SWRO WaterTAP benchmark 的可复现评测管线。项目与旁边的
`membraneclaw/` 完全独立，不修改或写入该仓库。

评测链路：

```text
Excel benchmark
  -> Agent / Agent-RAG / Agent-RAG-Skill
  -> raw responses and metadata
  -> anonymized GPT-5.6 judge batch
  -> step-level ratings
  -> HTML report and skill diagnostics
```

## 目录

```text
configs/              benchmark 集合与三组系统配置
benchmarks/development 旧测试 benchmark 的独立目录
benchmarks/question_bank 当前题库原始 Excel（按 D1-D5 分类）
benchmarks/normalized  当前默认评测集合的统一 JSON（运行时产生）
skills/               可版本化 SWRO-WaterTAP Skill 能力包
src/auto_evaluate/    管线实现
tests/                 无网络单元测试
runs/                  每次实验的响应、评分与运行清单（运行时产生）
reports/               可选的集中报告目录
```

## 环境准备

1. 复制 `.env.example` 为 `.env`。
2. 在 `.env` 中填写远端 OpenWebUI 地址、个人 API Key 和三个模型 preset ID。
3. 在 OpenWebUI 中保证：
   - `Agent`：WaterTAP，无 RAG，无 Skill；
   - `Agent-RAG`：WaterTAP，同一 RAG，无 Skill；
   - `Agent-RAG-Skill`：WaterTAP，同一 RAG，绑定当前候选版 `swro-watertap@0.5.0`。

`.env` 已被 Git 忽略。不要把真实密钥提交到版本库。

## 命令

以下命令均从本目录运行。推荐在 Miniforge Prompt 或已接好 conda 的 IDE 终端中执行。

```powershell
python ae.py import-benchmarks
python ae.py validate-benchmarks
python ae.py probe --details
python ae.py run --run-id pilot-001
python ae.py cycle --run-id pilot-001
python ae.py prepare-teacher --run-id pilot-001
python ae.py prepare-judge --run-id pilot-001
python ae.py validate-ratings --run-id pilot-001
python ae.py report --run-id pilot-001
python ae.py skill-gate --run-id pilot-001
python -m unittest discover -s tests -v
```

`run` 支持断点续跑：相同题目、系统、prompt 和配置的成功响应不会重复请求。
添加 `--force` 才会覆盖已有成功结果。

正式运行默认使用 SSE 流式响应。该模式与 OpenWebUI 网页端的长任务路径一致，
并避免 WaterTAP 多轮工具调用在非流式连接中被远端提前关闭。

当前默认采用双通道评测输入：被测系统先输出完整回答，再在末尾追加
`[SCORE_POINTS_BEGIN] ... [SCORE_POINTS_END]` 的 JSON trailer。judge 侧会优先使用
该结构化得分点做快速定位，但仍保留完整回答用于核验和防止 overclaim。

## Benchmark 集合

- `configs/benchmarks.json`：当前默认题库，现指向 `benchmarks/question_bank/raw/`
  下的真实题目。
- `configs/benchmarks_development.json`：旧测试 benchmark，仅用于开发回归或调试。
- `benchmarks/development/normalized/`：旧测试题的独立归档。

如果你要更新当前题库的 `normalized` 集合，直接运行：

```powershell
python ae.py import-benchmarks
python ae.py validate-benchmarks
```

如果你要单独回放旧测试集，用：

```powershell
python ae.py import-benchmarks --config configs/benchmarks_development.json --output benchmarks/development/normalized
python ae.py validate-benchmarks --benchmarks-dir benchmarks/development/normalized
```

## 一轮实验怎么跑

下面是一轮最常用的完整流程，假设本轮编号为 `pilot-003`。

### 最短常用路径

优先使用：

```powershell
python ae.py cycle --run-id pilot-003
```

它会自动执行：

- `validate-benchmarks`
- `probe --details`
- `run`
- `prepare-teacher`
- 等待你回填 `teacher_responses.jsonl`
- `prepare-judge`
- 等待你回填 `ratings.jsonl`
- `validate-ratings`
- `report`
- `skill-gate`

你只需要保留两次网页端操作：teacher 和 judge。

下面保留分步流程，方便排查或手动执行。

### 1. 先检查输入和远端配置

```powershell
python ae.py validate-benchmarks
python ae.py probe --details
```

### 2. 跑三个被测系统

```powershell
python ae.py run --run-id pilot-00x
```

运行成功后，三个系统在三道题上的回答会写入 `runs/pilot-003/responses/`。

### 3. 生成 teacher 批任务

```powershell
python ae.py prepare-teacher --run-id pilot-00x
```

这一步会生成：

- `runs/pilot-003/teacher_batch.jsonl`
- `runs/pilot-003/TEACHER_INSTRUCTIONS.md`

### 4. 在网页端运行 teacher

把 `teacher_batch.jsonl` 上传到 GPT-5.6 网页端 teacher 项目，要求它输出一份可直接
保存的 `teacher_responses.jsonl`。输出文件放回：

- `runs/pilot-003/teacher_responses.jsonl`

建议网页端 teacher 项目使用下面这段 system prompt：

```text
You are generating a machine-readable teacher_responses.jsonl file for SWRO Auto Evaluate.

Rules:
1. Process each input JSON object independently.
2. For each input JSON object, output exactly one JSON object on exactly one line.
3. Preserve task_id and case_id exactly as given.
4. Set system_id to "gpt-5.6-teacher".
5. Put the full benchmark answer in response_text.
6. Answer the benchmark blind. Do not use any reference answer or rubric.
7. Use available WaterTAP tooling if configured; otherwise clearly distinguish simulated, calculated, and estimated values.
8. Preserve units, state assumptions, show the calculation path, and check every stated constraint.
9. Return raw JSONL only.
10. Do not wrap the output in Markdown fences.
11. Do not add explanations, headings, numbering, commentary, or blank lines before or after the JSONL.
12. Each output JSON object must occupy exactly one line.
13. The result must be directly saveable as teacher_responses.jsonl.
14. Do not claim that a tool was called unless it was actually called.

If your output contains anything other than raw JSONL lines, it is invalid.
```

网页端输入框只需写：

```text
Please process the uploaded teacher_batch.jsonl and return only valid teacher_responses.jsonl content.
```

### 5. 生成 judge 批任务

确认 `teacher_responses.jsonl` 已放回运行目录后执行：

```powershell
python ae.py prepare-judge --run-id pilot-00x
```

这一步会生成：

- `runs/pilot-003/judge_batch.jsonl`
- `runs/pilot-003/judge_mapping.json`
- `runs/pilot-003/JUDGE_INSTRUCTIONS.md`

### 6. 在网页端运行 judge

只把 `judge_batch.jsonl` 上传到 GPT-5.6 网页端 judge 项目，不要上传
`judge_mapping.json`。网页端输出保存回：

- `runs/pilot-003/ratings.jsonl`

建议网页端 judge 项目使用下面这段 system prompt：

```text
You are generating a machine-readable ratings.jsonl file for SWRO Auto Evaluate.

Rules:
1. Process each input JSON object independently.
2. For each input JSON object, output exactly one JSON object on exactly one line.
3. Preserve task_id, case_id, and candidate_label exactly as given.
4. Score only against the supplied rubric.
5. Do not infer or speculate about the model identity behind the anonymous response.
6. Award partial credit step by step.
7. Cite concise evidence from the candidate response.
8. Distinguish a wrong result from a correct result with incomplete explanation.
9. Use only the supplied failure code vocabulary.
10. The sum of step scores must equal total_score.
11. Copy each step max_score from the rubric.
12. Return raw JSONL only.
13. Do not wrap the output in Markdown fences.
14. Do not add explanations, headings, numbering, commentary, or blank lines before or after the JSONL.
15. Each output JSON object must occupy exactly one line.
16. The result must be directly saveable as ratings.jsonl.
17. If your output contains anything other than raw JSONL lines, it is invalid.

Each output JSON object must contain:
- task_id
- case_id
- candidate_label
- total_score
- steps
- overall_diagnosis
- skill_improvement_suggestions
```

网页端输入框只需写：

```text
Please process the uploaded judge_batch.jsonl and return only valid ratings.jsonl content.
```

### 7. 校验评分并生成报告

```powershell
python ae.py validate-ratings --run-id pilot-00x
python ae.py report --run-id pilot-00x
```

最终报告写入：

- `runs/pilot-003/report.html`

### 8. 可选：检查 Skill 是否晋级

```powershell
python ae.py skill-gate --run-id pilot-00x
```

## GPT-5.6 teacher/judge

本项目不直接调用付费 OpenAI API。teacher 和 judge 都采用“本地生成批任务，网页端
执行后再把 JSONL 放回运行目录”的方式。最少需要人工回填两个文件：

- `teacher_responses.jsonl`
- `ratings.jsonl`

## Skill 版本晋级

`skills/swro-watertap/v0.5.0/SKILL.md` 是可直接导入 OpenWebUI 的自包含运行时
文件；同目录 JSON 保存机器可读的工作流、参数映射和失分预防检查。不要把整个
目录上传到 OpenWebUI，只更新该 `SKILL.md`。

`skill-gate` 将同一轮的 `Agent-RAG-Skill` 与 `Agent-RAG` 比较。当前开发阶段要求：

- 每道 development benchmark 都严格增分；
- 平均分严格增分；
- `Agent-RAG-Skill` 不出现 `TOOL_ARGUMENT` 或 `PARAMETER_EXTRACTION`。

任一条件未满足，命令返回非零状态，该 Skill 版本不晋级。当前三个 benchmark
属于 development set；后续论文结论必须在未用于改 Skill 的 held-out set 上复核。

## 数据隔离

- 被测系统只接收 `题目_Q`。
- `分步答案_A` 与 `评价标准` 只出现在 teacher/judge 阶段。
- Skill 与 RAG 中不得包含 benchmark 的标准答案数值。
- 最终实验应划分 development / validation / held-out test，避免 Skill 对测试集过拟合。
