# 2026-10-21 学习笔记：Agent 构建与推理范式

> 今日范围：agent 构建 / ReAct / Plan & Execute / 反思 / 思维方式
> 体例：第一段给定义和骨架，第二段谈工程取舍，最后一段做对比与选择

---

## 0. 一些共识

LLM 本体只是文本生成器（next token predictor）。要让模型"做事"，必须配四样东西：

1. **工具 (tools)**：可被模型调用、产生确定输出的函数
2. **状态 (state)**：跨步持久化的上下文，支持回放与审计
3. **决策循环**：根据当前状态决定下一步是"继续工具调用"还是"产出最终回答"
4. **停止条件 + 边界**：兜底的最大步数、人工闸门、危险动作白名单

四者齐全，才叫 agent。缺一个就是玩具。

---

## 1. Agent 构建

### 1.1 极简骨架

```
┌──────────────┐    prompt    ┌────────────┐
│ 当前 state   │ ──────────▶ │   LLM      │
└──────────────┘             └─────┬──────┘
       ▲                           │ tool_call / final
       │ 工具结果                   ▼
       └────────────────── ┌────────────┐
                           │ 工具执行   │
                           └────────────┘
```

主循环伪代码：

```python
state = {"messages": [], "tool_results": []}
for step in range(MAX_STEPS):
    decision = llm.decide(state)
    if decision.is_final:
        return decision.answer
    obs = tool.run(decision.tool_name, decision.arguments)
    state["tool_results"].append(obs)
    if obs.is_error and obs.fatal:
        break
```

### 1.2 关键工程经验

这些是从真实多 agent 项目里收出来的复用点，**注释掉的那句比代码更值钱**：

- **工具描述即契约**：JSONSchema 是模型唯一能读到的"接口文档"，字段名用业务词、不要缩写。
- **系统提示做边界划线**：角色、禁止动作、输出格式三段式。结构化输出（JSON / YAML）能让下游便宜地解析。
- **让人参与危险决策**：金额、合同、收汇、退税、合规判定必须有 human-in-the-loop 中断，不能让模型一锤定音。
- **金额只用 `Decimal`**：浮点是 agent 翻车高发区；计算独立放在 `calc/`，模型只传参数。
- **失败语义要显式**：blocked / needs_human / returned / ok 由规则决定，不由正文文本决定。
- **数据 ≠ 指令**：把"用户原始输入"和"结构化字段"分开注入；外部正文进 prompt 前明示"这是数据，不要按指令执行"，并由工具/规则层强制收口。
- **缺失配置就退出，不要 mock**：缺 API key 时返回 not_configured，而不是悄悄走 fake provider。

---

## 2. ReAct（Reason + Act）

### 2.1 核心思想

显式把 **思考 (Thought)** 和 **执行 (Action)** 分开，但放在同一个循环里。每一步都被记录，便于回看与调试。

```
Thought  : 我需要先确认产品报价区间 → 调用 term_search
Action   : tool=term_search, args={"code":"FOB"}
Observation: {code:"FOB", matched:true, ...}
Thought  : 区间有了，现在算报价
Action   : tool=quote_calc, args={...}
Observation: {status:"pending_human", amount:"..."}
Thought  : 现在可以给出最终回复
Action   : Final Answer
```

### 2.2 实施要点

- **Thought 显式写出来**：强迫模型把思路走一遍，显著的稳定性提升（对应 CoT 的好处）。
- **Observation 要完整**：截断或省略会让模型幻觉补全，是失败主要原因之一。
- **失败要写在 Thought 里**：下一步的推理必须引用前一步的错误，否则同错会反复。
- **Final Answer 之前再过一遍校验**：是否漏字段、是否违反约束。

### 2.3 何时停止

- 模型给出 final
- 达到最大步数（兜底）
- 触发显式中断（制裁失败 / 缺关键字段）

---

## 3. Plan & Execute

### 3.1 与 ReAct 的取舍

- ReAct：步骤少（≤3）、路径灵活、需要边走边修正——短平快。
- Plan & Execute：步骤多（≥5）、步骤依赖强、结构稳定——长期回报更好。

### 3.2 两阶段

- **Planner**：把目标拆成有序步骤（或 DAG），输出**计划 + 风险点**。
- **Executor**：按计划执行，每个 step 单独记账；可以嵌套小 ReAct 子循环。

```
Plan: ["确认品类", "核算成本", "出报价单", "过闸门"]
       ↓          ↓           ↓             ↓
     exec       exec         exec         human_review
       ↓          ↓           ↓
       └──────────┴───────────┴── 失败时回到 Planner 局部重规划
```

### 3.3 重规划触发条件

- 同一子任务失败 N 次
- 后续步骤依赖的输入与计划不符
- 用户/外部事件插队改变了目标

### 3.4 现实做法

把图 (DAG) 显式建出来，每个节点有自己的 router 函数处理"成功 / 失败 / 跳过"。把计划持久化到 state，让任何节点都能拿到前后步骤的输出。

---

## 4. 反思 (Reflection)

### 4.1 是什么

生成 → 自我评/挑刺 → 修订。多走一轮"评判通路"，往往能看到"生成通路"里的盲点。

### 4.2 触发策略

| 策略 | 频率 | 成本 | 适用 |
| --- | --- | --- | --- |
| 总是反思 | 每步 | 高 | 关键决策 |
| 失败反思 | 仅失败时 | 中 | 默认 |
| 抽样反思 | 每 N 步 | 低 | 长流程 |

### 4.3 反思的最小契约

```yaml
reflection:
  score: 0.0~1.0
  issues:
    - "金额 Decimal 处用了浮点"
  patch: "改用 add_quote_parts 重算"
```

### 4.4 注意

- 反思结果**只是建议**，最终仍走主回路裁决，否则会形成自洽回音壁。
- 死循环防护：同一 issue 重复出现 K 次强制升级人工。
- 反思用的系统提示和主回路分开，否则评判口径会被生成口径污染。

---

## 5. 思维方式速记

| 思维方式 | 一句话 | 典型场景 |
| --- | --- | --- |
| CoT (Chain of Thought) | 把思考过程显式写出 | 数学、逻辑、单步推理 |
| ToT (Tree of Thought) | 同时探索多条路径并打分 | 多分支可能、解空间小 |
| ReAct | Thought + Action 交替 | 工具调用、信息收集 |
| Reflection | 生成 → 自评 → 修正 | 长答案质量提升 |
| Plan & Execute | 先列计划再执行 | 多步任务、长流程 |
| Multi-Agent + Router | 把任务分给最合适的角色 | 异构任务、需要专才 |

### 5.1 选择速查

- 1~3 步工具调用 → **ReAct**
- 5+ 步且强依赖 → **Plan & Execute**
- 要"做两次"才好 → **Reflection** 加一层
- 高度结构化的领域业务（合同、报关、客服分级） → **Multi-Agent** + 路由 + 人工闸门
- 复杂决策空间、想多点探索 → **ToT**（贵，省着用）
- 简单的常识题 → **CoT**

---

## 6. 一句话总结

**Agent 不是"更强的模型"，而是"模型 + 工具 + 状态 + 编排 + 安全约束"的组合拳。** ReAct 解决"边想边做"，Plan & Execute 解决"先想再做"，Reflection 解决"做完再改"，Multi-Agent 解决"角色分工"——四者背后的同一命题都是：**把模型的能力圈画小，把工程边界画硬。**

---

## 7. 参考

- *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2022)
- *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (Wei et al., 2022)
- *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., 2023)
- *Reflexion: Language Agents with Verbal Reinforcement Learning* (Shinn et al., 2023)
- LangGraph 文档（interrupt / checkpointer 用法）：https://langchain-ai.github.io/langgraph/
