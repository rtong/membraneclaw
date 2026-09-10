# E31 run status

| Run | 状态 | 说明 |
|---|---|---|
| `solver-skill-p3-validation-3case-v3` | `completed_with_benchmark_confound` | 18/18 Solver和18/18 Judge完成；主指标-4.33，配对均值+6.22 |

逐题best-of-3：D6-6c-02为-36，D6-6c-03为+19，D6-6c-06为+4，胜/平/负为2/0/1。
按照预注册规则，本轮未通过继续门槛：平均逐题best-of-3不是正值，执行诊断也没有改善。

同时发现验证题的公开Prompt省略了开发题明确给出的代表性进料流量：膜单元1 kg/s、工厂
0.10 m3/s；Gold却使用这些值。D6-6c-02的C10主要使用工具默认0.3092 m3/s并因此受到
决定性扣分。该run必须冻结并保留，但不能作为无混杂的Skill迁移结论。

下一步不是扩大题量或进入C01/C11，而是版本化修正公开输入后，用相同冻结Skill和新run ID
重新进行三题验证。旧Prompt、v3结果和修正理由必须同时保留。
