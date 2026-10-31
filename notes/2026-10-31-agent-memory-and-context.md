# 2026-10-31 学习笔记：Agent 记忆与上下文工程

> 今日范围：上下文预算 / 压缩技术 / 记忆分层 / 记忆写入与检索 / 遗忘机制
> 体例：第一段给定义和骨架，第二段谈工程取舍，最后一段做对比与选择

---

## 0. 一些共识

先纠正一个普遍的错觉：**「记忆」不是「上下文」。**

- **上下文（context）**：这一次调用真正送进模型的那一串 token。它是有限的、临时的、用完即弃的。
- **记忆（memory）**：存在模型之外（数据库、向量库、文件系统），在需要时被检索回来、放进上下文。

所以「给 agent 加记忆」实际是两件事：
1. **决定什么值得被记住**（写入策略）
2. **决定这次要往上下文里放什么**（上下文工程）

很多团队做的「加记忆」只做了第一件，用一个向量库把对话全存进去，然后每轮把 top-k 塞回 prompt。结果是：记忆越积越脏，上下文越来越噪，agent 反而变笨。

另外三个共识：

1. **上下文是预算，不是仓库。** 每一次调用你都在做取舍：系统提示、工具定义、历史消息、检索内容、输出预留，五者抢同一个额度。
2. **丢失信息很少是因为「装不下」，多数是因为「选错了」。** 先怀疑选择策略，再怀疑容量。
3. **遗忘是功能不是 bug。** 一个只增不减的记忆系统，最终会退化成噪声发生器。

---

## 1. 上下文的构成与预算

### 1.1 上下文里到底有什么

```
┌─────────────────────────────────────────────┐
│ ① 系统提示 / 角色设定            固定，1~3K    │
├─────────────────────────────────────────────┤
│ ② 工具定义（JSON Schema）        随工具数增长  │
├─────────────────────────────────────────────┤
│ ③ 系统/开发者指令 + 运行时状态    固定+动态     │
├─────────────────────────────────────────────┤
│ ④ 历史消息（对话 + 工具调用 + 结果）  ← 大头    │
├─────────────────────────────────────────────┤
│ ⑤ 本轮检索到的内容（记忆/文档）     ← 常被低估  │
├─────────────────────────────────────────────┤
│ ⑥ 输出预留（max_tokens）          必须留够     │
└─────────────────────────────────────────────┘
```

**两个最容易被低估的**：

- **② 工具定义**：上一份笔记说过工具多会稀释质量。从 token 角度看，20 个工具的 schema 轻松吃掉 2~4K token。工具描述是**常驻成本**，每次调用都付。
- **⑤ 检索内容**：塞 10 个 chunk × 800 token = 8K，比历史消息还大。

### 1.2 预算分配

**不要把窗口用满。** 留 20~30% 的余量：

```python
from dataclasses import dataclass

@dataclass
class ContextBudget:
    total: int = 128_000        # 模型窗口
    reserve_output: int = 4_000 # 输出预留
    max_tool_defs_tokens: int = 4_000
    max_retrieval_tokens: int = 8_000
    max_history_tokens: int = 0  # 剩下的都给历史

    @property
    def usable(self) -> int:
        return self.total - self.reserve_output - self.max_tool_defs_tokens - self.max_retrieval_tokens
```

**分配原则**：

| 项 | 建议 | 理由 |
|---|---|---|
| 输出预留 | 10~20% | 截断是最糟的失败模式，答案被砍掉一半 |
| 工具定义 | 5~10% | 工具多就裁剪，别挤占其他 |
| 检索内容 | 15~25% | 宁少勿精，3~5 条高质量 > 20 条噪音 |
| 历史消息 | 剩余 | 弹性最大，可压缩 |

### 1.3 上下文腐化（Context Rot）

**论文**：*Lost in the Middle: How Language Models Use Long Contexts*（Liu et al., TACL 2024）—— 长上下文里，**开头和结尾的信息利用得最好，中间的最差**。

**但更近的观察更狠**：随着上下文变长，模型对**任何位置**的信息利用能力都会下降——不只是中间。上下文越长，整体注意力越涣散。

**工程含义**：

| 做法 | 说明 |
|---|---|
| **不要为了「也许有用」而塞** | 每加一段都在稀释所有其他内容 |
| **最重要的放两头** | 关键指令放系统提示末尾，或最后一条消息 |
| **主动做压缩** | 长度到了阈值就压，不要等撑爆 |
| **分段处理** | 很长的工作拆成多个独立会话，中间用摘要衔接 |

### 1.4 一个上下文预算器

```python
# 依赖: pip install tiktoken
# 可运行: python context_budget.py
import json
from typing import Any

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:                     # 离线环境兜底：粗略按字符估
    _ENC = None

def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, int(len(text) / 1.5))   # 中文约 1.5 字符/token 的粗估

def count_message_tokens(messages: list[dict]) -> int:
    """每条消息额外计 ~4 token 的角色/分隔开销"""
    return sum(count_tokens(m.get("content") or "") + 4 for m in messages)

def count_tool_tokens(tools: list[dict]) -> int:
    return count_tokens(json.dumps(tools, ensure_ascii=False))
```

---

## 2. 上下文压缩技术

### 2.1 五种压缩手段

| 手段 | 做什么 | 损失 | 成本 | 适合 |
|---|---|---|---|---|
| **截断** | 丢弃最早/最晚的消息 | 永久丢失 | 零 | 兜底 |
| **筛选** | 按规则/打分挑保留哪些 | 可能丢重要 | 低 | 有明确优先级时 |
| **摘要** | 把一段历史压成一段摘要 | 细节丢失 | 中~高 | 长对话 |
| **结构化** | 把消息转成结构化状态（而非自然语言） | 表达力下降 | 低 | 有明确状态机时 |
| **外置** | 把内容写到外部（scratchpad/文件），上下文只留引用 | 需要时再取回 | 低 | 大中间结果 |

### 2.2 摘要式压缩

**最朴素也最有效**：把旧的对话轮次压成摘要。

```python
SUMMARIZE_PROMPT = """请压缩以下 agent 工作历史，保留对后续步骤有用的信息。

必须保留：
1. 已确认的事实（订单号、金额、客户要求、已查到的字段）
2. 已经失败的操作及原因（避免重复尝试）
3. 尚未解决的问题和待办
4. 用户明确的约束条件

可以丢弃：
- 寒暄、过程性的工具调用细节
- 已被后续结果覆盖的中间值

输出格式：
【已知事实】
- ...
【已尝试】
- 操作X → 结果/失败原因
【待办】
- ...

工作历史：
{history}
"""

def compress_history(messages: list[dict], llm, keep_recent: int = 6,
                     max_chars: int = 3000) -> list[dict]:
    """
    保留最近 keep_recent 条原始消息，更早的历史压成一条摘要消息。
    """
    if len(messages) <= keep_recent:
        return messages

    head, tail = messages[:-keep_recent], messages[-keep_recent:]

    # 只把有内容的部分送去摘要（tool_calls 的 arguments 保留结构）
    summary = llm.invoke(SUMMARIZE_PROMPT.format(history=render_for_summary(head)))
    if len(summary.content) > max_chars:
        summary.content = summary.content[:max_chars]

    return [{"role": "system", "content": f"【历史摘要】\n{summary.content}"}] + tail
```

**摘要压缩的三个坑**：

1. **摘要会丢数字**。金额、订单号、日期是 agent 最需要的信息，摘要时必须显式要求保留（见上面 prompt 的「必须保留」列表）。
2. **摘要会形成错误回音壁**。一旦摘要里写错了「客户要的是蓝色」，后面所有轮次都会基于这个错误推理，且很难发现。**所以摘要要保留「原始消息可回溯」的指针。**
3. **摘要本身也会变长**。长对话要**递归摘要**（摘要的摘要），或者定期做「重置」——开一个新会话，只带一份摘要过去。

**递归摘要**：

```python
def recursive_compress(messages, llm, chunk_size=40, keep_recent=6):
    """把长历史分段摘要，摘要之间再摘要"""
    if len(messages) <= chunk_size:
        return compress_history(messages, llm, keep_recent)

    summaries = []
    for i in range(0, len(messages) - keep_recent, chunk_size):
        chunk = messages[i:i + chunk_size]
        summaries.append({"role": "system",
                          "content": llm.invoke(
                              SUMMARIZE_PROMPT.format(history=render_for_summary(chunk))).content})

    return recursive_compress(summaries + messages[-keep_recent:],
                              llm, chunk_size, keep_recent)
```

### 2.3 结构化压缩（把对话变成状态）

**思路**：agent 的历史里真正有价值的往往不是「我们聊了什么」，而是「当前处于什么状态、已确定哪些事实」。

```python
@dataclass
class AgentState:
    """agent 的结构化状态，替代大部分自然语言历史"""

    goal: str = ""
    confirmed_facts: dict = field(default_factory=dict)   # {"order_no": "20241015-001"}
    failed_attempts: list = field(default_factory=list)   # [{"tool":"query","error":"timeout"}]
    pending: list = field(default_factory=list)            # 待办
    constraints: list = field(default_factory=list)        # 用户约束
    turn_count: int = 0

    def render(self) -> str:
        lines = [f"【当前目标】{self.goal}"]
        if self.confirmed_facts:
            lines.append("【已确认事实】")
            lines += [f"- {k} = {v}" for k, v in self.confirmed_facts.items()]
        if self.failed_attempts:
            lines.append("【已尝试（勿重复）】")
            lines += [f"- {a['tool']}: {a['error']}" for a in self.failed_attempts]
        if self.constraints:
            lines.append("【约束】" + "；".join(self.constraints))
        if self.pending:
            lines.append("【待办】" + "；".join(self.pending))
        return "\n".join(lines)
```

**状态注入 prompt**：

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": state.render()},   # 状态作为一条消息注入
    {"role": "user", "content": current_user_input},
]
```

**好处**：
- 状态是**结构化的**，可以直接被代码检查（「金额字段有了吗？」）
- 不会被摘要的「信息损失」模糊掉
- token 可预测

**代价**：
- 需要你**明确知道哪些字段是重要的**（这本身是领域知识）
- 自然语言的微妙信息（「用户语气有点急」）会丢

**实践建议**：**结构化状态 + 保留最近几轮原文** 是最稳的组合。状态管「事实」，原文管「语气和细节」。

### 2.4 工具结果压缩

工具结果是上下文中**最容易失控**的部分。

```python
def compress_tool_result(name: str, raw: str, max_chars=1500,
                         query: str = "", embedder=None) -> str:
    """
    工具结果裁剪策略：
    1. 结构化结果 → 保留与查询相关的条目
    2. 长文本 → 抽取相关段落
    3. 兜底 → 头尾折叠
    """
    if len(raw) <= max_chars:
        return raw

    # 1. 尝试结构化：JSON 数组 → 按 query 相关性筛
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if embedder is not None and query:
                texts = [json.dumps(it, ensure_ascii=False) for it in data]
                embs = embedder.encode(texts, normalize_embeddings=True)
                q = embedder.encode([query], normalize_embeddings=True)
                scores = embs @ q[0]
                keep = np.argsort(-scores)[:20]
                kept = [data[i] for i in sorted(keep)]
                out = json.dumps({"total": len(data), "items": kept,
                                  "has_more": len(kept) < len(data),
                                  "note": "按查询相关性筛选前 20 条"},
                                 ensure_ascii=False)
                if len(out) <= max_chars:
                    return out
    except Exception:
        pass

    # 2. 兜底：头尾折叠
    head_n = int(max_chars * 0.6)
    tail_n = max_chars - head_n
    return (raw[:head_n] + f"\n...[省略 {len(raw)-max_chars} 字符]...\n" + raw[-tail_n:])
```

**更好的做法在工具侧**：工具就别返回 5000 条，让它支持 `limit` 和 `cursor`。**上下文压缩是补救，工具设计才是根治。**

### 2.5 外置（scratchpad）

**思路**：中间结果不进上下文，写到外部（文件、数据库、key-value），上下文里只留一个引用。

```python
# 大计算结果写文件，上下文只留路径
result_path = write_scratchpad(json.dumps(big_result))
messages.append({"role": "assistant", "content":
    f"已生成完整查询结果，共 {len(big_result)} 条，已保存至 {result_path}。"
    f"前 5 条：{json.dumps(big_result[:5], ensure_ascii=False)}"})
```

**适用**：网页抓取全文、批量查询结果集、生成的中间 JSON。

**配套需要一个 scratchpad 读取工具**，让 agent 需要时能取回来。

---

## 3. 记忆分层

### 3.1 四层记忆

借用认知科学的分类，在 agent 工程里非常贴切：

| 层 | 认知科学对应 | 存什么 | 存在哪 | 生命周期 |
|---|---|---|---|---|
| **工作记忆** | 短时记忆 | 当前任务的中间状态 | **上下文内** | 单次会话 |
| **情景记忆** | 情景记忆 | 「某时某地发生过什么」 | 事件日志 / 向量库 | 跨会话 |
| **语义记忆** | 语义记忆 | 客观事实、偏好、规则 | 知识库 / KV | 长期 |
| **程序性记忆** | 程序性记忆 | 做事的方法、SOP | 提示模板 / skill 文件 | 长期 |

**工程上的关键区分**：

> **情景记忆回答「发生过什么」，语义记忆回答「是什么」。**
> 用户说「上次我们讨论过 X 的报价」→ 查情景记忆。
> 用户说「X 的报价规则是什么」→ 查语义记忆。

混在一起检索会出乱子：把「上次讨论的临时报价数字」当成「标准报价规则」返回。

### 3.2 一个记忆条目该长什么样

```python
@dataclass
class Memory:
    id: str
    kind: str                 # "episodic" | "semantic" | "procedural"
    content: str
    subject: str = ""         # 主体：客户/订单/产品/项目
    tags: list = field(default_factory=list)
    source: str = ""          # 来源：会话ID / 文档路径
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    importance: float = 0.5   # 重要性 0~1
    confidence: float = 1.0   # 置信度 0~1（事实越确定越高）
    valid_until: float = None  # 时效性：政策类记忆要有过期时间
    supersedes: str = None     # 被哪条记忆取代（冲突消解用）
```

**四个容易被忽略的字段**：

| 字段 | 为什么重要 |
|---|---|
| `kind` | 决定它进哪个检索池、怎么用 |
| `importance` | 决定召回优先级和遗忘速度 |
| `confidence` | 「客户预算是 50 万」这种要能被后续修正 |
| `valid_until` | **政策会变**。没有时效性的记忆系统迟早给出过期答案 |

---

## 4. 记忆的写入策略

### 4.1 什么值得记

**判据（按优先级）**：

| 判据 | 说明 | 例子 |
|---|---|---|
| **不可从当前上下文推导** | 记下来是为了未来用 | 「客户的对接人叫李工」 |
| **跨会话仍然有效** | 只在本次会话有用就别记 | 「用户刚才打了个招呼」 |
| **用户明确要求记住** | 最强信号 | 「以后都叫我老王」 |
| **低频但高影响** | 平时不出现，一旦出现很关键 | 「这个客户对价格敏感」 |
| **纠错结果** | 模型/人工纠正过的，务必记 | 「这个产品不走海运」 |

**反例（不该记）**：

- 一次性的中间状态（订单查询的原始 JSON）
- 模型的猜测和推理过程
- 工具返回的大段原文（应该是引用，不是复制）
- 会过期的临时信息（除非带 `valid_until`）

### 4.2 何时记

**四种触发**：

```python
class MemoryWriter:
    def __init__(self, store, llm, extractor_prompt: str):
        self.store = store
        self.llm = llm
        self.extractor_prompt = extractor_prompt

    async def consider(self, turn: Turn, state: AgentState) -> list[Memory]:
        """
        在每轮结束后调用，判断这轮是否产生了值得记的东西。
        返回新写入的记忆。
        """
        candidates = await self._extract_candidates(turn)
        written = []
        for mem in candidates:
            if not self._is_worth(mem, state):
                continue
            if await self._is_duplicate_or_conflict(mem):
                continue
            written.append(await self.store.write(mem))
        return written

    async def _extract_candidates(self, turn: Turn) -> list[Memory]:
        prompt = self.extractor_prompt.format(
            user=turn.user_text, assistant=turn.assistant_text)
        raw = (await self.llm.invoke(prompt)).content
        return [Memory(**m) for m in parse_json_array(raw)]
```

**触发时机**：

| 时机 | 适合 |
|---|---|
| 每轮结束后 | 通用，但要防止过度记录 |
| **任务结束时** | 一次性任务的结论 |
| **用户明确说「记住」时** | 最高优先级 |
| **检测到纠错时** | 「不是这样的，应该是……」 |
| **跨天/长间隔时** | 会话开始时先做一次「记忆回顾」 |

**过度记录的危害**：记忆库污染 → 召回噪音 → 上下文变脏 → agent 变笨。**写入要保守，召回可以激进。**

### 4.3 冲突消解

**场景**：

```
3 月：客户预算是 50 万
9 月：客户预算调整到 80 万
```

如果两条都留着，检索时可能召回旧的。

**四种策略**：

| 策略 | 做法 | 适用 |
|---|---|---|
| **最新优先** | 同 subject+key 取 `created_at` 最大的 | 简单，覆盖大部分场景 |
| **显式取代** | 新记忆的 `supersedes` 指向旧记忆，检索时只返回未被取代的 | 精确，可追溯 |
| **并存带时间** | 两条都留，答案里说明「截至 X 时是 50 万」 | 需要历史演进分析 |
| **置信度加权** | 冲突时取 `confidence` 高的 | 事实确定度差别大时 |

**推荐「最新优先 + 显式取代」组合**：

```python
async def write_with_supersede(self, mem: Memory):
    # 找到同一 subject+key 的现有记忆
    existing = await self.store.find(subject=mem.subject, key=mem.key)
    for old in existing:
        if mem.created_at > old.created_at:
            mem.supersedes = old.id
            old.superseded_by = mem.id
    return await self.store.write(mem)
```

**不要做「向量相似度去重」**：「预算 50 万」和「预算 80 万」向量很相似，但它们是冲突不是重复。用 subject+key 这种**结构化键**做消解，语义相似度只用来做「大概重复」的粗筛。

---

## 5. 记忆的检索

### 5.1 检索不只是相似度

纯粹的向量检索会召回「看起来像但不相关」的旧记忆。生产可用的记忆检索通常是**多信号加权**：

```python
def retrieve_memories(query: str, store, embedder, k: int = 5,
                      now: float = None, half_life_days: float = 90.0) -> list[Memory]:
    now = now or time.time()
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]

    hits = store.knn_search(q_emb, top_k=k * 5)      # 先粗取
    scored = []
    for m in hits:
        sim = float(m.embedding @ q_emb)              # 语义相关
        recency = 0.5 ** ((now - m.created_at) / (half_life_days * 86400))  # 时间衰减
        importance = m.importance
        final = (0.55 * sim + 0.25 * recency + 0.20 * importance) * m.confidence
        # 被取代的记忆直接排除
        if m.superseded_by:
            continue
        scored.append((final, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]
```

**权重的直觉**：

- **相似度 0.55**：主导项，但不能单独说了算
- **时间衰减 0.25**：半衰期按业务定。偏好类记忆给长半衰期（一年），临时状态给短（一天）
- **重要性 0.20**：人工或规则标注的「这条特别重要」
- **置信度**：乘性因子，不确定的记忆整体降权

**时间衰减是必要的**：否则一条一年前的临时信息会和昨天的最新事实同等权重。

### 5.2 主体过滤（比相似度更重要）

**纯语义检索最常见的错误**：用户问「订单 20241015-001 的物流」，召回了一条讲「订单 20241020-008 的物流」的相似记忆。

**解法**：把**结构化约束**（主体、类型、时间范围）作为**硬过滤**加在语义检索之前：

```python
def retrieve_with_filters(query, store, embedder, subject=None, kind=None,
                          since_days=None, k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    filters = {}
    if subject:  filters["subject"] = subject
    if kind:     filters["kind"] = kind
    if since_days:
        filters["created_at"] = {"$gte": time.time() - since_days * 86400}

    hits = store.search(q_emb, top_k=k * 5, filters=filters)   # 先过滤后检索
    return hits[:k]
```

> **这正是 `ai-study-notes` 仓库里 RAG 笔记讲的「权限过滤必须在检索层做」**。记忆检索同理：主体过滤不能靠模型自觉，必须由检索层强制。

### 5.3 遗忘机制

**一个只增不减的记忆系统，最终是噪声发生器。**

**三种遗忘**：

| 类型 | 做什么 | 触发 |
|---|---|---|
| **衰减 (decay)** | 降低旧记忆的检索优先级 | 时间（见 §5.1） |
| **合并 (consolidate)** | 把多条相似记忆压成一条 | 相似度超阈值 |
| **淘汰 (evict)** | 直接删除 | 长期未被访问 / 价值分低于阈值 |

**合并（consolidation）**：

```python
async def consolidate(store, embedder, threshold=0.92):
    """把语义近似的记忆合并成一条，保留信息量最大的表述"""
    all_mem = store.all()
    for i, a in enumerate(all_mem):
        for b in all_mem[i+1:]:
            if a.kind != b.kind or a.subject != b.subject:
                continue
            sim = float(a.embedding @ b.embedding)
            if sim >= threshold:
                merged_content = llm_merge(a.content, b.content)  # LLM 合并表述
                await store.update(a.id, content=merged_content,
                                  created_at=max(a.created_at, b.created_at))
                await store.delete(b.id)
```

**淘汰（eviction）**：

```python
def should_evict(m: Memory, now: float) -> bool:
    # 从未被访问过，且已经过了新窗口期
    if m.access_count == 0 and now - m.created_at > 7 * 86400:
        return True
    # 长期冷门
    if m.access_count <= 1 and now - m.last_accessed > 180 * 86400:
        return True
    # 价值分太低且已过期
    if m.importance < 0.2 and m.valid_until and now > m.valid_until:
        return True
    return False
```

**关键**：**淘汰前先归档**，别直接删。冷记忆进「归档池」，召回时在主池找不到再去翻归档池。这样既控制了上下文噪音，又不丢信息。

### 5.4 记忆的写入时机与位置

**别只存在向量库里。** 记忆至少有三种归宿：

| 归宿 | 存什么 | 检索方式 | 例子 |
|---|---|---|---|
| **结构化 DB** | 明确的事实型记忆 | 精确查询（SQL/KV） | 「订单号」「预算是 80 万」 |
| **向量库** | 模糊的、需要语义匹配的 | 相似度 + 过滤 | 「客户偏好什么沟通方式」 |
| **文件 / 文档** | 大块的、原始的 | 检索摘要后回读 | 「上次的完整方案 PDF」 |

**实践**：**能用精确查询的，不要用向量检索**。向量检索擅长「大概像」，不擅长「就是它」。把结构化事实抽出来放 DB，需要的时候精确查；把长文本放文件，需要时检索+回读。

---

## 6. 上下文工程与提示词的关系

**一个重要的观念转变**：

> **提示词工程是上下文工程的一个子集。**
>
> 传统提示词工程关注「怎么把指令写清楚」。
> 上下文工程关注「**这一次调用，模型应该看到什么**」——包括指令、状态、历史、检索结果、工具输出、以及它们的**顺序**。

**上下文工程的完整清单**：

```
□ 系统提示：精简但完整，把最重要的规则放最后
□ 工具定义：按场景裁剪，控制常驻成本
□ 状态注入：结构化状态优于长对话历史
□ 历史压缩：保留最近 N 轮 + 摘要 + 状态
□ 检索内容：3~5 条，按相关度排，最重要的放两头
□ 输出预留：至少 10%
□ 顺序：关键约束别放中间
```

**一个具体的对比**：

```
❌ 传统做法：把 30 轮对话全塞进去
   → 上下文 15K token，中间信息丢失，还挤掉了工具定义

✅ 上下文工程：
   系统提示（精简，1.5K）
   + 工具定义（按场景裁剪，1.5K）
   + 结构化状态（已确认事实/约束/待办，0.3K）
   + 最近 4 轮原文（2K）
   + 历史摘要（0.8K）
   + 本轮检索（1.5K）
   + 输出预留（2K）
   = 9.6K，命中率高得多
```

---

## 7. 常见失败模式

| # | 失败模式 | 症状 | 解法 |
|---|---|---|---|
| 1 | **把所有对话都存进记忆** | 记忆库污染，召回噪音 | 写入保守，写入判据表 |
| 2 | **记忆只用向量检索** | 召回了相似但主体不对的东西 | 主体过滤做硬约束 |
| 3 | **没有时间衰减** | 旧信息和新信息同等权重 | 半衰期 + `valid_until` |
| 4 | **没有冲突消解** | 同时召回「预算 50 万」和「预算 80 万」 | subject+key 结构化键 + supersedes |
| 5 | **上下文塞满** | 输出被截断 / 中间信息丢失 | 留 20% 余量，主动压缩 |
| 6 | **摘要丢数字** | 金额、订单号错了 | 摘要 prompt 显式要求保留 |
| 7 | **摘要形成错误回音壁** | 错误被固化，很难发现 | 摘要保留原始消息指针 |
| 8 | **工具结果不裁剪** | 上下文被单个工具撑爆 | 工具侧分页 + 压缩兜底 |
| 9 | **没有结构化状态** | 每次都要从长历史里翻 | 引入状态对象 |
| 10 | **工具定义常驻太贵** | 每次调用都付 schema 的钱 | 按场景裁剪工具 |
| 11 | **记忆永不删除** | 检索质量随时间退化 | 衰减 + 合并 + 归档 |
| 12 | **用户说「忘记它」没实现** | 合规问题 | 提供显式遗忘接口 |
| 13 | **多用户记忆没隔离** | **串号泄露** | 记忆必须带 user_id/tenant_id，所有查询强制过滤 |
| 14 | **上下文顺序没设计** | 关键约束被忽略 | 重要的放两头 |

> **第 13 条是安全事故级别。** 记忆存储必须和权限体系同构——RAG 里的「权限过滤必须在检索层」，记忆检索完全一样。多租户场景下，`user_id` 过滤不能靠 prompt 里写「只返回该用户的数据」，必须是检索层的强制条件。

---

## 8. 一句话总结

**上下文工程决定「这一次模型看到什么」，记忆系统决定「它以后还知道什么」——前者是每次调用的预算控制，后者是跨会话的信息资产；而两者共用的第一性原理是：不是「存得越多越好」，而是「存得准、取得对、该忘的忘」。** 记忆系统的成熟度不看它存了多少条，看它在第 1000 次会话时是否还比第 10 次更准。

---

## 9. 参考

- *Lost in the Middle: How Language Models Use Long Contexts* (Liu et al., TACL 2024) — https://arxiv.org/abs/2307.03172
- *Context Rot: How Increasing Input Tokens Impacts LLM Performance* (2025) — Chroma 技术报告
- *Memory in the Age of AI Agents: A Survey* (2025) — https://arxiv.org/abs/2512.13564
- *Generative Agents: Interactive Simulacra of Human Behavior* (Park et al., UIST 2023) — 记忆流 / 反思机制
- *MemGPT: Towards LLMs as Operating Systems* (Packer et al., 2023) — https://arxiv.org/abs/2310.08560 （分页记忆的经典框架）
- *A-MEM: Agentic Memory for LLM Agents* (2025) — https://arxiv.org/abs/2502.12110 （记忆的自动化组织）
- *Reflexion*（记忆与自我改进）— 见 `notes/2026-10-21-agent-construction.md`
- *Self-RAG*（检索质量作为反思信号）— 见 `notes/2026-10-31-agent-eval-and-observability.md` 的相关讨论
- RAG 侧的上下文组装与「Lost in the Middle」实践 → `ai-study-notes` 仓库 `03-RAG/08-生成与提示工程.md`
