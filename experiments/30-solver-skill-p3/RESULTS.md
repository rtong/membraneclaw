# E30 run status

| Run | 状态 | 是否可评分 | 说明 |
|---|---|---:|---|
| `solver-skill-p3-c00-c10-v1` | `invalid_preflight` | 否 | 只执行prepare，selection ID错误；没有模型调用 |
| `solver-skill-p3-c00-c10-v2` | `invalid_tool_surface` | 否 | 6份最终响应只调用calculator/http_get；目标WaterTAP工具0/6 |
| `solver-skill-p3-c00-c10-v3` | `planned` | 尚未运行 | 等待远程工具绑定恢复 |

v1/v2必须保留用于审计，但不得合并、覆盖或提交Judge。当前没有P3 Solver Skill效果结论。
两者已统一移动到 `archive/experiments/solver-skill-p3-invalid/runs/`。
