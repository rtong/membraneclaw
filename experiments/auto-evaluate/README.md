# SWRO Auto Evaluate

本项目对 SWRO/WaterTAP 工程题进行可复现的端到端评测。Excel 是不可变源数据，标准化
JSON 是执行输入；待测模型只收到题目，参考答案和两套 rubric 只供匿名 Judge 使用。流水线
覆盖 benchmark 导入、Router、OpenWebUI 系统执行、双 Teacher、Judge、reward analysis、HTML
报告和论文图导出。

## 当前研究边界

本项目没有更新 9B 或 GPT-5.6 的模型参数，也没有 epoch、training loss、validation loss 或
学习率曲线。报告中的 `Task loss` 仅表示评分缺口（满分减实际得分），不是训练优化损失。
因此当前工作应表述为“工具增强评测”“reward-guided adaptive routing”或“Router 策略评测”，
不能表述为完整的强化学习训练。

当前 EST 论文的主线是：

1. `baseline` 与 `tools` 比较 WaterTAP 工具带来的求解能力变化；
2. `tools` 与 `tools-rag` 在 R0/R2 中成对比较 RAG 的反事实效应；
3. Router 独立判断 `skip_rag`/`use_rag`，不参与工程求解；
4. GPT-5.6 General/Tools 是参考角色，不进入待测系统主排名；
5. 报告同时展示任务质量、Tool 效率、路由、可靠性、错误归因和可观察轨迹。

## 环境

在 Miniforge Prompt 中进入项目目录后执行：

```bat
conda activate scrapingpipe
python -m pip install -e .
python -m unittest discover -s tests -v
```

拉取包含依赖变更的新代码后，也要重新执行一次 `python -m pip install -e .`。项目固定使用
`openai-codex==0.147.0`，因为 Tools Teacher 所需的 `--approve-for-me` 不存在于较旧的
bundled CLI；只运行 `git pull` 不会升级已经安装的旧 SDK。

### Codex 认证：什么时候运行什么命令

Teacher/Judge 会复用当前用户已有的 Codex ChatGPT 登录状态。**正常重复运行
systems、Teacher、Judge 或完整 `auto` 任务时，不需要执行任何登录命令。**

| 情况 | 应执行的操作 |
| --- | --- |
| 新设备首次部署，或流水线明确报认证错误 | 执行下面的“状态检查”命令 |
| 状态显示已登录 | 不要执行 `login`；直接运行评测 |
| 状态显示未登录 | 执行一次“登录”命令，完成后直接运行评测 |
| 明确要切换 ChatGPT 账号 | 执行“登录”命令；注意它会覆盖当前活动登录状态 |

状态检查（只读，不会更改当前账号）：

```bat
python -c "from codex_cli_bin import bundled_codex_path; import subprocess; raise SystemExit(subprocess.call([str(bundled_codex_path()), 'login', 'status']))"
```

登录（仅在未登录或明确切换账号时执行）：

```bat
python -c "from codex_cli_bin import bundled_codex_path; import subprocess; raise SystemExit(subprocess.call([str(bundled_codex_path()), 'login']))"
```

**警告：登录命令会覆盖当前活动登录状态；如果选择另一个账号，当前账号将被替换。**
如需保留当前账号，不要执行登录命令。

Tools Teacher 查找 Codex CLI 的顺序是：`CODEX_CLI_PATH`、PATH 中的 `codex`、SDK
随包安装的 CLI runtime。在新设备执行 `python -m pip install -e .` 后，通常无需
配置 CLI 路径或安装 Codex Desktop。`CODEX_CLI_PATH` 仅用于强制使用指定 CLI 版本。

Tools Teacher 通过无交互 CLI 执行 WaterTAP connected app。先确认 Codex 中已经连接
`watertap`，并确保项目 `.env` 能被流水线加载。Tools Teacher 默认使用
`codex exec --approve-for-me`，避免后台进程因无法点击工具确认而产生
`user cancelled MCP tool call`。代码同时要求至少一个 WaterTAP 调用真正成功且返回非空
observation；仅出现工具事件但调用失败不会进入评分。

如需诊断旧 SDK 路径，可临时设置 `AE_CODEX_TOOL_TEACHER_EXECUTOR=sdk`，但正式 Tools
Teacher 使用默认的自动批准 CLI 路径。`--approve-for-me` 与显式 `--sandbox` 不能同时使用，
当前 worker 已避免这一参数冲突。

## 实验设计

论文主实验将 Solver Skill 与 RAG 路由解耦。`swro-watertap@0.8.9` 已冻结为前期实验产物，
不再进入活动配置。`swro-rag-router@0.1.2` 只用于第一阶段判断 `use_rag` 或 `skip_rag`，
随后由相同的 9B + WaterTAP solver 执行。

D1-D6 预注册为不需要 RAG 的 R0 benchmark。D7 尚未交付，目前只预留与D1-D6相同四Sheet
工作簿的导入入口；交付后先检查真实结构，再从标准化数据生成R0/R2派生标签，不要求同事
修改源工作簿。最终路由评测中的标签和证据不会进入Router或solver prompt。主矩阵包含四个
相同9B权重的受控条件：

| 系统 | Tools | RAG | Skill |
| --- | ---: | ---: | ---: |
| `baseline` | 否 | 否 | 否 |
| `tools` | 是 | 否 | 否 |
| `tools-rag` | 是 | 是 | 否 |
| `tools-adaptive-rag` | 是 | 由 Router 决定 | 仅 Router Skill |

前三个是物理执行分支；`tools-adaptive-rag` 是虚拟策略条件。正式的 Adaptive 策略得分采用
离线分支回放：Router 输出 `skip_rag` 时继承同题 `tools` 得分，输出 `use_rag` 时继承同题
`tools-rag` 得分和回答。默认不再独立运行第二次Adaptive solver，也不把重复回答再次提交Judge。
这样 routing regret 只评价路由选择，不受另一次随机工具轨迹影响；该逻辑同时适用于D7的
`use_rag`和D1-D6的`skip_rag`。

另设 `gpt-5.6-teacher-general` 和 `gpt-5.6-teacher-tools` 两个参考条件。当前 D7 尚未交付，
因此 D1-D6 只能验证 Tools 增益、RAG 副作用和 Router 的 `skip_rag` 假阳性率；完整路由结论
必须等待自然需要外部知识的 D7。现有 RAG 原始资料位于 `rag_knowledge/original/`。D7当前
接收流程和交付后审计步骤见 `docs/D7_BENCHMARK_SPEC.md`。

## OpenWebUI

主实验只需要三个稳定 preset，并在 `.env` 填入实际 ID：

```env
OPENWEBUI_MODEL_BASELINE=baseline
OPENWEBUI_MODEL_TOOLS=tools
OPENWEBUI_MODEL_TOOLS_RAG=tools-rag
OPENWEBUI_RAG_VERSION=your-current-openwebui-knowledge-version
```

`tools` 与 `tools-rag` 必须使用相同 9B 权重和 WaterTAP 权限，区别仅为后者绑定固定版本的
Knowledge。三个 preset 均不挂载 `swro-watertap`。Router Skill 由评测程序作为独立第一阶段
system prompt 发送，不需要绑定到 OpenWebUI preset。

Tools 两个物理分支采用可观察调用完整性门禁：只有 `tools_enabled=true`、但轨迹中没有成功
WaterTAP/RO-chem 调用及 observation 的文本回答会记录为
`required_tool_call_missing`，不得作为 Tools 结果进入正式评分。这用于阻止模型凭空生成
“模拟数值”后被误记为工具增强成功。

### 9B生成与上下文恢复

当前所有 9B 求解和 Router 请求均显式关闭 thinking。主求解配置为
`temperature=0.2`、`top_p=0.9`、`max_tokens=2048`；Router 使用
`temperature=0`、`max_tokens=128`。关闭 thinking 是为了给可见答案和工具结果保留上下文，
不是研究变量，也不代表模型能力增强。OpenWebUI 三个 preset 及全局 Model Defaults 中应将
temperature、top-p 和 max-tokens 留为未设置，避免显式 preset 参数覆盖代码请求；preset 只
负责固定模型、Tools 和 Knowledge 绑定。主回答还要求结论优先、最多700词且不复述试错过程。

Tools 类请求仍可能因为较长的工具 transcript 达到16,384 token上限，Baseline也可能在完整
结论前达到输出上限。对 `context_window_exceeded`、`incomplete_response` 或
`output_budget_exhausted`，流水线只运行一次tool-free `context-reset-finalizer@0.2.0`。它适用于
三个物理系统，只接收原题和已有可观察内容，不重新调用工具，不得虚构未观察到的模拟结果。
partial excerpt最多12000字符，整个finalizer prompt最多24000字符，最终回答最多600词。记录保留：

- `native_status` / `native_error_type`：原生求解是否失败；
- `completion_mode=context_reset_finalizer`：最终回答来自上下文恢复；
- partial response 和原始 trajectory：供 Judge 做失分与首错归因；
- final status：恢复后是否得到可评分的完整回答。

因此报告必须同时展示原生完成率、恢复率和最终完成率。Recovered 回答可以评价完整流水线，
但不能写成 9B 原生求解成功。

089 只作为冻结的前期产物保留，需要查看其最终内容时可构建：

```bat
python skills\swro-watertap\v0.8.9\build_skill.py
```

RAG 继续使用已经部署的 `RO-operational Manual.pdf` 和 `Safety file.xlsx`，不要额外上传人工构造资料。部署检查：

```bat
python ae.py probe --benchmark-set d1_d6 --details
```

## 第一步：独立 Router 评测

```bat
python ae.py router-eval --benchmark-set d1_d6 --run-id router-r0-pilot --pilot
```

该命令在固定的12题pilot上同时运行 `9B zero-shot Router` 和 `9B + Router Skill`，只做短文本
分类，不调用WaterTAP、不加载RAG、也不进入求解阶段。结果写入
`runs/router-r0-pilot/router_summary.json`。D1-D6全部为R0，因此本阶段只检验输出稳定性和
`skip_rag`准确率，不能单独证明Router具有完整泛化性。两组Router统一使用流式响应并关闭
thinking，避免短分类请求被远端约60秒的非流式连接限制干扰。

## 第二步：小型端到端评测

先用 3 道 D1-D6 做链路 smoke；`--case` 可重复：

```bat
python ae.py auto --benchmark-set d1_d6 --run-id adaptive-r0-pilot --stage systems --case D1-1a-feasibility --case D2-2b-constraint-conflict --case D5-5a-n01-recovery-limit
```

只有系统响应完整时才进入Teacher/Judge。3题只用于确认链路，不能作为跨领域分数比较集。

### D1-D6 十二题跨领域均衡集

在链路稳定后，推荐每个领域固定2题，共12题。该集合按题型和复杂度预先选择，不依据任何
系统的已有得分筛题：

| 领域 | Case 1 | Case 2 | 主要覆盖 |
| --- | --- | --- | --- |
| D1 | `D1-1a-feasibility` | `D1-1c-existing-pump-reuse-and-cleaning-trigger` | 膜面积可行性；旧泵复用与清洗触发 |
| D2 | `D2-2a-feed-salinity-control-map` | `D2-2b-constraint-conflict` | 盐度控制图；约束冲突与不可行性 |
| D3 | `D3-3a-n01-salinity-intrusion` | `D3-3b-n01-px-efficiency-inference` | 盐度入侵；PX效率反演 |
| D4 | `D4-4a-capex-design-sensitivity` | `D4-4a-n05-px-efficiency-vendor-guarantee` | CAPEX敏感性；PX质保与能耗门槛 |
| D5 | `D5-5a-n01-recovery-limit` | `D5-5b-scaling-pretreatment-selection` | 回收率化学边界；结垢预处理选择 |
| D6 | `D6-6a-n01` | `D6-6b-n07-multi-simulator-swro-pressure-scaling` | 联合设计优化；多模拟器压力/结垢任务 |

系统阶段（单行）：

```powershell
python ae.py auto --benchmark-set d1_d6 --run-id d1-d6-balanced-12-r2-20260825-v1 --stage systems --require-complete-systems --case D1-1a-feasibility --case D1-1c-existing-pump-reuse-and-cleaning-trigger --case D2-2a-feed-salinity-control-map --case D2-2b-constraint-conflict --case D3-3a-n01-salinity-intrusion --case D3-3b-n01-px-efficiency-inference --case D4-4a-capex-design-sensitivity --case D4-4a-n05-px-efficiency-vendor-guarantee --case D5-5a-n01-recovery-limit --case D5-5b-scaling-pretreatment-selection --case D6-6a-n01 --case D6-6b-n07-multi-simulator-swro-pressure-scaling
```

这会产生36个物理solver请求（12题×3个物理系统）和12个短Router请求，并在本地生成12个
Adaptive策略记录。systems默认并发为2。系统阶段完整后，再使用同一run ID和完全相同的case
集合运行Teacher、Judge和报告；对应任务量分别为24个Teacher任务和60个Judge任务，因为
Adaptive与所选物理分支回答相同，不重复评分。Judge默认并发为4。
该12题集用于比较D1-D6上的Baseline、Tools、Tools + RAG以及Adaptive策略表现。由于D1-D6均为
R0，它还能量化不必要RAG的代价和Router的误调用，但不能证明RAG在R2知识缺口题上的收益。
R2能力必须另用D7或下面的D7 mock开发集检查。当前不建议直接运行全部117题。

## 正式分阶段运行

系统阶段稳定后再运行完整集合：

```bat
python ae.py auto --benchmark-set d1_d6 --run-id adaptive-r0-full --stage systems --require-complete-systems
```

需要正式评分时，使用同一 run ID 继续 Teacher、Judge 和报告：

```bat
python ae.py auto --benchmark-set d1_d6 --run-id adaptive-r0-full --stage teachers
python ae.py auto --benchmark-set d1_d6 --run-id adaptive-r0-full --stage judges
python ae.py auto --benchmark-set d1_d6 --run-id adaptive-r0-full --stage report
```

如远端运行较慢，可用同一 run ID 分阶段续跑：

```bat
python ae.py auto --benchmark-set <set> --run-id <run-id> --stage systems
python ae.py auto --benchmark-set <set> --run-id <run-id> --stage teachers
python ae.py auto --benchmark-set <set> --run-id <run-id> --stage judges
python ae.py auto --benchmark-set <set> --run-id <run-id> --stage report
```

成功结果按 request hash 复用；同一 run ID 续跑时不会重复请求已经成功且输入未变化的任务。
上下文超限等确定性错误不会原样重试，而是按上述单次 context-reset 策略处理。正式运行使用
`--require-complete-systems`，避免把基础设施失败误当成模型低分。所有阶段都必须使用同一个
run ID；使用 `--case` 子集时，每个阶段必须重复相同的 `--case` 参数。

## Reward 闭环

Judge 完成后执行：

```bat
python ae.py reward-analysis --run-id <run-id>
```

新版报告不再把六个条件混成一个排行榜，而是拆成：

1. 主要待测系统：Baseline 与 Tools；
2. Teacher 参考：General 与 Tools 上界；
3. RAG 反事实：按 R0/R2 比较 `skip_rag` 和 `use_rag`；
4. Router 策略：策略回放分、所选物理分支和 routing regret；
5. 多维科研图：任务质量、Tool效率、RAG效应、Router策略和执行可靠性。

报告生成时会把五张论文用 SVG 矢量原图同时写入 `runs/<run-id>/figures/`。也可以单独运行
某一张图：

```powershell
python ae.py plot --run-id <run-id> --figure main-scores
python ae.py plot --run-id <run-id> --figure quality-efficiency
python ae.py plot --run-id <run-id> --figure rag-effect
python ae.py plot --run-id <run-id> --figure router-policy
python ae.py plot --run-id <run-id> --figure reliability
```

使用 `--figure all` 一次重建全部图，使用 `--output figure.svg` 指定单张图输出位置。SVG 可直接
插入 Word/PowerPoint，放大不会失真。这里展示的是评测分数、反事实效应、路由策略和运行
可靠性，不是 training-loss 或 epoch 曲线。

输出包含逐题/逐rubric的配对增益、95% bootstrap区间、完成率、错误类型、延迟、工具调用
和失分归因，并已支持混淆矩阵、Precision、Recall、F1、按RAG需求分层和regret。D7的标签
来源和D1-D7合并配置将在真实文件导入并审计后确定。

Router 优化只允许使用开发集的误路由证据，且不得写入 D1-D7 编号、参考答案、来源名称或
题目专属目标值。D7 交付并启用后，先用少量题确认 RAG 确实提供决策所需知识，再运行同一四组矩阵。

更完整的论文约束和部署检查见 `docs/EXPERIMENT_PROTOCOL.md`、`docs/SKILL_ITERATION.md` 与 `docs/OPENWEBUI_SETUP.md`。

## D7流程模拟

真实D7交付前，可使用六题平衡模拟集调试导入、Router和端到端流程。它包含3个原始R0题和
3个由固定RAG资料支持的派生R2题，单独写入`benchmarks/normalized_d7_mock`；源Excel不变，
且模拟结果不得作为论文D7或RAG增益证据。
完整清单和命令见`docs/D7_MOCK.md`。

若混合模拟集中的R2知识缺口被长题干淹没，可运行独立的六题短配对诊断集
`d7_mock_router_probe`。它只用于`router-eval`，不允许进入系统求解或Judge；命令和判读规则见
`docs/D7_MOCK_ROUTER_PROBE.md`。

显式缺失提示达到满分后，使用`d7_mock_router_natural`检查模型能否自行发现未给出的外部
政策或制造商标准；该集合同样只用于Router诊断。

### D7 mock四题R2/R0 smoke

在真实 D7 交付前，如需检查R2/R0两类路由和RAG链路，可固定以下四题：

| Case | 类别 | 选择原因 |
| --- | --- | --- |
| `D7-mock-01-feasibility` | R2 / `use_rag` | 压力政策知识缺口，检验 Safety 文件支持的 RAG |
| `D7-mock-02-constraint-conflict` | R2 / `use_rag` | 制造商清洗触发规则，检验 Manual 支持的 RAG |
| `D7-mock-04-capex-sensitivity` | R0 / `skip_rag` | 不依赖外部规则的经济计算对照 |
| `D7-mock-06-multisimulator` | R0 / `skip_rag` | 多工具复杂任务，同时检验上下文恢复与轨迹评价 |

这四题形成 2 个 R2 与 2 个 R0，覆盖两种RAG资料、计算题和复杂工具题。它是RAG链路的
synthetic development smoke，不是D1-D6跨领域分数比较集，也不能写入真实D7论文主结果。

系统阶段：

```powershell
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-typical-4-20260825-v1 --stage systems --require-complete-systems --case D7-mock-01-feasibility --case D7-mock-02-constraint-conflict --case D7-mock-04-capex-sensitivity --case D7-mock-06-multisimulator
```

系统完整后，用相同 run ID 和完全相同的 case 集合继续：

```powershell
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-typical-4-20260825-v1 --stage teachers --teacher-general-concurrency 1 --teacher-tools-concurrency 1 --codex-retries 1 --require-complete-systems --case D7-mock-01-feasibility --case D7-mock-02-constraint-conflict --case D7-mock-04-capex-sensitivity --case D7-mock-06-multisimulator
```

```powershell
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-typical-4-20260825-v1 --stage judges --codex-retries 1 --require-complete-systems --case D7-mock-01-feasibility --case D7-mock-02-constraint-conflict --case D7-mock-04-capex-sensitivity --case D7-mock-06-multisimulator
```

```powershell
python ae.py auto --benchmark-set d7_mock --run-id d7-mock-typical-4-20260825-v1 --stage report --require-complete-systems --case D7-mock-01-feasibility --case D7-mock-02-constraint-conflict --case D7-mock-04-capex-sensitivity --case D7-mock-06-multisimulator
```

最终查看 `runs/d7-mock-typical-4-20260825-v1/report.html`，论文图位于同目录的 `figures/`。
