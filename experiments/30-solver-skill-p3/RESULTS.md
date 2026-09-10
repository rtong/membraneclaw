# E30 run status

| Run | 状态 | 是否可评分 | 说明 |
|---|---|---:|---|
| `solver-skill-p3-c00-c10-v1` | `invalid_preflight` | 否 | 只执行prepare，selection ID错误；没有模型调用 |
| `solver-skill-p3-c00-c10-v2` | `invalid_tool_surface` | 否 | 6份最终响应只调用calculator/http_get；目标WaterTAP工具0/6 |
| `solver-skill-p3-c00-c10-v3` | `completed` | 是 | C00最高52，C10最高100，best-of-3效应+48；平均配对效应+9.33 |

v1/v2必须保留用于审计，但不得与v3合并。两者已统一移动到
`archive/experiments/solver-skill-p3-invalid/runs/`。v3证明了开发题上的机制信号，但不证明
稳定性或跨题泛化。
