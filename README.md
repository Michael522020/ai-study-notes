# agent-learning

记录我在 agent / LLM 应用方向的学习笔记、读源码心得与可复用的脚手架。

## 目录约定

- `notes/` — 按日期或主题组织的学习笔记（Markdown）
- `examples/` — 配套的小示例代码
- `references/` — 论文 / 文档 / 链接

每个文件以日期开头，例如 `notes/2026-10-21-agent-construction.md`。

## 笔记索引

### 学习笔记（`notes/`）

| 文件 | 覆盖范围 |
|---|---|
| [`2026-10-21-agent-construction.md`](./notes/2026-10-21-agent-construction.md) | Agent 构建骨架 / ReAct / Plan & Execute / 反思 / 思维方式选型 |
| [`2026-10-31-tool-calling-and-mcp.md`](./notes/2026-10-31-tool-calling-and-mcp.md) | Function Calling 底层机制 / 工具描述设计 / 错误语义 / MCP 协议 / 工具治理 |
| [`2026-10-31-agent-memory-and-context.md`](./notes/2026-10-31-agent-memory-and-context.md) | 上下文预算 / 压缩技术 / 记忆分层 / 写入与检索 / 遗忘机制 |
| [`2026-10-31-agent-eval-and-observability.md`](./notes/2026-10-31-agent-eval-and-observability.md) | 轨迹埋点 / 五层评测指标 / LLM-as-Judge 校准 / 回放测试 / 回归门禁 |
| [`2026-10-31-multi-agent-orchestration.md`](./notes/2026-10-31-multi-agent-orchestration.md) | 协作拓扑 / 通信契约 / 角色与权限 / 错误传播与隔离 / 框架选型 |

### 项目笔记（`notes/projects/`）

| 文件 | 内容 |
|---|---|
| [`ft-agents.md`](./notes/projects/ft-agents.md) | 外贸全流程 Agent 系统 |
| [`minimax-chat.md`](./notes/projects/minimax-chat.md) | 浏览器 + 流式 LLM 最小工程模板 |

### 阅读顺序建议

```
构建范式 → 工具层 → 记忆/上下文 → 评测 → 多 Agent 编排
2026-10-21  →  10-31-a  →   10-31-b    →  10-31-c  →    10-31-d
```

后四篇是同一天写的，按「工具 → 状态 → 度量 → 拓扑」这条线读最顺：工具是手，记忆是脑，评测是尺子，编排是骨架。

## 风格

- 第一遍先抄定义，第二遍谈自己的理解和工程取舍
- 末尾留一句"一句话总结"，把最核心的心得收紧
- 代码片段尽量能独立跑通；不能跑的写明依赖
- 不上传任何环境配置、密钥、模型权重
