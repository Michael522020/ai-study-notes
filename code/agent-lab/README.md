# agent-lab：三种 Agent 形态的实验台

> 对应笔记：[`07-Agent与智能体`](../../07-Agent与智能体/)

## 跑起来

```bash
python code/agent-lab/run.py eval            # 离线评测（10 个用例，无需 API key）
python code/agent-lab/run.py profile         # 打印三种形态的参数对照
python code/agent-lab/run.py chat --profile bot        # 真实 API 交互
python code/agent-lab/run.py chat --profile superagent # 长任务：问目标 + 验收标准
```

## 三种形态的差别在哪

| | Agent Bot | AI Agent | Super Agent |
|---|---|---|---|
| 输入 | 一句话问询 | 一个任务 | 目标 + 验收标准 |
| 循环深度 | 2~4 步 | 5~15 步 | 几十步、多阶段 |
| 工具面 | 7 个（窄） | 10 个 | 12 个 + 检查点 |
| 单次成本上限 | $0.02 | $0.60 | $3.00 |
| 结束条件 | 给出答案 / 转人工 | 给出答案 | **验收标准逐条通过** |
| 失败兜底 | 转人工 | 标注 needs_human | 检查点续跑 + 人工 |
| 关键文件 | `profiles/bot.py` | `profiles/agent.py` | `profiles/superagent.py` |

`runtime.py` 是三种形态共用的同一个 agentic loop。**形态差异不在代码结构里，
在系统提示、工具面、预算参数和验收标准这四处。**

## 离线评测用例（10 个）

| 用例 | 验证什么 |
|---|---|
| bot-知识库政策问答 | 查工具 → 回答，命中知识库 |
| bot-订单状态查询 | 单工具直达 |
| bot-金额必须走计算工具 | 涉及金额必须调 `calculate`，不许心算 |
| bot-越权承诺必须转人工 | 触发人工闸门，返回 `needs_human` + 待确认动作 |
| bot-入参非法时自修 | 校验失败 → 错误回喂 → 模型改对参数 |
| bot-预算闸门会拦下失控循环 | 步数超限 → `budget_exceeded` |
| agent-多步任务先拆解再执行 | 计划 → 执行 → 总结 |
| agent-工具不存在时报错并纠正 | 未知工具 → 报错 → 改用正确工具 |
| superagent-长任务走完分解执行验收 | 分解 → 3 步执行 → 验收通过 |
| superagent-验收不过不算完成 | 验收未过 → `verification_failed`，不算完成 |

## 关键工程约定

1. **带工具调用的回合 ≠ 结束**，只有纯文本回合才是最终回答。
2. **工具入参非法不崩**：错误信息回喂给模型让它自修，同一入参连续错 2 次就转人工。
3. **写操作默认干跑**：`create_ticket` 在未放行时只产出待确认草稿。
4. **预算三条线**：步数、工具调用次数、美元成本，任一越界即停并返回明确状态。
5. **thinking 块带 signature 原样回传**（preserved thinking），否则模型读不到自己的推理链。
