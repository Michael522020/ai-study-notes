# RAG 总览与架构

> **核心问题**：RAG 到底有哪些模块？每个模块的决策点在哪？

---

## 1. 问题定义

**为什么需要 RAG？**

| LLM 的局限 | RAG 的解法 |
|---|---|
| **知识截止**（不知道训练之后的事） | 检索实时数据 |
| **企业私有知识**（内部文档、工单、合同） | 检索企业知识库 |
| **幻觉**（编造不存在的事实） | 提供可溯源的证据 + 强制引用 |
| **无法更新**（重新训练成本极高） | 更新索引即可 |
| **无法解释**（不知道依据） | 返回原始片段 + 引用位置 |
| **长尾知识**（记住每个细节不现实） | 把细节存在外部，按需检索 |

**RAG 的核心主张**：

> **把「参数化知识」（parametric knowledge）和「非参数化知识」（non-parametric knowledge）分开。**
> - 参数化知识：模型权重里的通用能力
> - 非参数化知识：外部检索存储（向量库、图、SQL）

---

## 2. 三层范式

### 2.1 Naive RAG（2020）

**论文**：*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, Lewis et al., NeurIPS 2020

```
┌──────────────────────────────────────────────────┐
│ 离线（索引构建）                                     │
│   文档 → 切分 → 嵌入 → 向量库                       │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│ 在线（查询）                                        │
│   查询 → 嵌入 → 相似度检索 → Top-K 片段             │
│        → 拼进 prompt → LLM 生成答案                 │
└──────────────────────────────────────────────────┘
```

**结构**（论文里的公式）：

$$
\text{Probability}_{\theta,\eta}(z|x) \approx \sum_{z' \in \mathcal{Z}_{\text{retrieve}}} p_{\eta}(z'|x) \, p_\theta(z|x,z')
$$

其中：
- $x$：输入查询
- $z'$：检索到的文档
- $p_\eta(z'|x)$：检索器（$\eta$）的分布
- $p_\theta(z|x,z')$：生成器（$\theta$）的条件分布

**RAG-Sequence vs RAG-Token**：
- **RAG-Sequence**：整个文档作为一个整体检索，边缘化到文档级别
- **RAG-Token**：逐 token 边缘化，允许生成过程中「引用不同文档的不同部分」

**RAG 的三大优势**：
1. **可更新**：更新知识库即可
2. **可溯源**：能指出答案来自哪个文档
3. **参数高效**：不需要为每个任务微调

**Naive RAG 的四大问题**：

| # | 问题 | 后果 |
|---|---|---|
| 1 | **切分不当** | 语义被切断，检索到的片段看不懂 |
| 2 | **检索单一** | 只用向量检索，关键词匹配失败 |
| 3 | **上下文冗余** | Top-K 全塞进 prompt，噪声大、「中间迷失」 |
| 4 | **无引用/无法拒答** | 不知道依据，也不知道什么时候该说「不知道」 |

### 2.2 Advanced RAG（2023）

**针对 Naive RAG 的四个问题逐一加模块**：

```
┌──────────────────────── 索引侧 ────────────────────────┐
│  解析 → 切分优化 → 嵌入优化 → 元数据/过滤 → 索引        │
└────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────── 检索侧 ────────────────────────┐
│  查询改写 → 多路召回 → 融合 → 重排 → 上下文压缩         │
└────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────── 生成侧 ────────────────────────┐
│  提示组装 → 引用落地 → 拒答/校验 → 后处理              │
└────────────────────────────────────────────────────────┘
```

**这是目前 90% 生产系统的形态。**

### 2.3 Modular RAG（2024）

**论文/观点来源**：*The Power of Modular RAG Architecture*（Es, 2024）、LangChain / LlamaIndex 的框架化实践

**核心思想**：RAG 不应该是一个「黑盒流程」，而应该是一组**可插拔、可独立评测、可组合**的模块。

```
                    ┌─────────────────┐
   查询 ────────────→│  Query Router   │ ───→ 路由到不同策略
                    └─────────────────┘
                             ↓
        ┌────────────┬───────────┬────────────┐
        ↓            ↓           ↓            ↓
   [Naive RAG]  [Self-RAG]  [GraphRAG]  [Agentic]
   简单直答      自我反思      关系推理      多步自主
```

**收益**：
- 每个模块可以独立评测（"检索准不准" vs "生成好不好"）
- 可以按场景路由（简单问题走 Naive，复杂问题走 GraphRAG）
- 便于 A/B 测试和迭代

### 2.4 Agentic RAG（2025）

**核心思想**：**模型自己决定要不要检索、检索什么、检索几次。**

```
用户问题
   ↓
┌──────────────────────────────────────┐
│  Agent 循环                            │
│  ┌────────────────────────────────┐  │
│  │ 思考 (Reason)                   │  │
│  │  ↓                              │  │
│  │ 决定动作 (Act)                   │  │
│  │  ├─ 直接回答? → 输出            │  │
│  │  ├─ 检索? → 执行 → 观察          │  │
│  │  └─ 分解/改写? → 继续循环       │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

**代表工作**：
- **Self-RAG**（2023）：模型输出「反思 token」决定是否需要检索
- **CRAG / Corrective RAG**（2024）：评估检索质量，不满意则换策略
- **FLARE**（2023）：基于生成内容做主动检索
- **ReAct**（2022）：推理 + 行动交替的通用范式
- **Adaptive-RAG**（2024）：复杂度分类器路由不同策略

**代价**：延迟不可控、成本高、调试困难。

---

## 3. 完整数据流（工程视角）

### 3.1 离线：索引构建

```
原始文档（PDF/Word/HTML/数据库）
   ↓ ① 解析与清洗
结构化文本 + 元数据
   ↓ ② 切分
Chunk 列表（text + metadata）
   ↓ ③ 嵌入
向量 + 原文 + 元数据
   ↓ ④ 索引
向量库 + 倒排索引 + 元数据索引
```

**每个环节的决策点**：

| 环节 | 关键决策 |
|---|---|
| ① 解析 | 用什么解析器？表格怎么处理？要不要 OCR？ |
| ② 切分 | chunk 多大？按什么边界切？要不要加父块？ |
| ③ 嵌入 | 用哪个模型？维度多少？要不要量化？ |
| ④ 索引 | HNSW 还是 IVF？要不要 BM25 倒排？ |

### 3.2 在线：查询

```
用户查询
   ↓ ⑤ 查询理解
（可选：意图分类 / 改写 / 分解 / HyDE）
   ↓ ⑥ 多路召回
（向量检索 + BM25 + 关键词 + 元数据过滤）
   ↓ ⑦ 融合与重排
（RRF / 加权 / Cross-Encoder）
   ↓ ⑧ 上下文组装
（去重、压缩、排序、token 预算控制）
   ↓ ⑨ 生成
（prompt 模板 + 引用 + 拒答）
   ↓ ⑩ 后处理
（引用校验、格式化、流式返回）
```

**每个环节的决策点**：

| 环节 | 关键决策 |
|---|---|
| ⑤ 查询理解 | 要不要改写？要不要 HyDE？ |
| ⑥ 召回 | top_k 取多少？要不要多路？ |
| ⑦ 重排 | 用什么 reranker？取多少条进最终上下文？ |
| ⑧ 组装 | 上下文放哪（前面/中间/后面）？怎么去重？ |
| ⑨ 生成 | 什么模型？什么 prompt？强制引用？ |
| ⑩ 后处理 | 怎么验证引用是真的？ |

---

## 4. 上下文的位置效应（重要且反直觉）

**论文**：*Lost in the Middle: How Language Models Use Long Contexts*, Liu et al., TACL 2024

**核心发现**：

> **LLM 对上下文的利用呈 U 形曲线**：
> - 开头（beginning）的信息被利用得最好
> - 中间（middle）的信息被利用得最差
> - 结尾（ending）的信息被利用得较好（但比开头略差）

```
上下文利用率
  ↑
  │  ╱╲                    ╱╲
  │ ╱  ╲                  ╱  ╲
  │╱    ╲                ╱    ╲
  └──────╲──────────────╱──────→ 位置
  开头   中间（最差）      结尾
```

**对 RAG 的直接含义**：

| 实践 | 说明 |
|---|---|
| **最重要的片段放最前面或最后面** | 不要放中间 |
| **减少无关内容** | 噪声不仅无用，还会挤占「好位置」 |
| **排序策略要有讲究** | 按相关性降序放，**或**「最相关的放两头」 |
| **不要简单截断** | 宁可少放几条，也别放 20 条噪音 |

**实测建议**：
```
[最相关的 1-2 条] ← 放最前
  [其他相关片段]
  [用户的原始问题] ← 放最后（很多模型对最后的指令更敏感）
```

---

## 5. RAG vs 微调 vs 长上下文

| 方案 | 知识更新 | 成本 | 可溯源 | 适合 |
|---|---|---|---|---|
| **纯 LLM** | ❌ 需重训 | 低 | ❌ | 通用能力、创作 |
| **微调 (SFT)** | ❌ 需重训 | 高 | ❌ | **格式/风格/任务能力** |
| **长上下文** | ✅ 直接塞 | 高（token 成本） | ⚠️ 弱 | 少量文档、一次问答 |
| **RAG** | ✅ 更新索引 | 中 | ✅ | **知识密集型问答** |
| **RAG + 微调** | ✅ | 最高 | ✅ | 生产级最优 |

### 5.1 论文：长上下文 vs RAG

**论文**：*Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach*, Li et al., 2024

**核心结论**：
1. **RAG 在知识密集型任务上通常优于长上下文**（尤其成本/效果比）
2. **长上下文在小规模、查询相关性高时更简单**
3. **混合方案最优**：先用检索缩小候选范围，再用长上下文精读

**成本对比（量级）**：
```
一次 RAG 查询:
  检索 (10ms) + 嵌入 (5ms) + rerank (50ms) + LLM(2K token 输入, 500 token 输出)
  ≈ 总计 $0.001 ~ $0.01

一次长上下文查询:
  LLM(100K token 输入)
  ≈ 总计 $0.10 ~ $5.00     ← 贵 10~100 倍
```

### 5.2 混合方案

```
问题 → 判断复杂度
  ├─ 简单 → 短上下文 + 直接生成（便宜）
  └─ 复杂 → RAG 检索 → 长上下文精读候选
```

**论文**：*LongRAG*（2024）——把「检索」和「长上下文」结合：让模型能在长上下文里做「大海捞针」式的检索。

---

## 6. 什么时候不要用 RAG

这是一个重要但常被忽略的问题：

| 情况 | 更好的方案 |
|---|---|
| 任务是**创作/推理/代码生成**，不需要外部事实 | 直接 LLM（或微调） |
| 知识**高度结构化**（如订单状态、库存） | **SQL 查询 / API 调用**，不是向量检索 |
| 知识**极其动态**（秒级变化） | 实时 API，不要走离线索引 |
| 数据量**很小**（< 100 篇） | **直接把全文放进 context** |
| 数据量**很大但查询模式固定** | **微调 + 少样本**，比 RAG 便宜 |
| 需要**精确计算/统计** | 代码执行（Code Interpreter），不是 RAG |
| 需要**跨文档的全局总结** | **GraphRAG / 长上下文 map-reduce** |

> **工程结论**：**RAG 是「非结构化知识 + 模糊查询 + 可溯源需求」场景的解法**。结构化数据走 SQL/API，非结构化大规模才走 RAG。

---

## 7. RAG 的评估维度（先建立判断标准）

一个 RAG 系统的质量分解为：

$$
\text{Quality} = \underbrace{\text{Retrieval Quality}}_{\text{召回、排序}} \times \underbrace{\text{Generation Quality}}_{\text{忠实度、相关性}}
$$

**检索质量**：
- **召回率（Recall@k）**：相关的有没有被检回来
- **精确率（Precision@k）**：检回来的有多少是相关的
- **排序质量（NDCG/MRR）**：相关的排得够靠前吗
- **上下文相关性**：片段和查询真的相关吗

**生成质量**：
- **忠实度（Faithfulness）**：答案是否有依据（没有编造）
- **答案相关性（Answer Relevance）**：答案是否切题
- **上下文相关性（Context Relevance）**：检索到的内容是否有用
- **上下文精确率（Context Precision）**：有用的内容排前面吗

**关键认知**：
> **生成质量差，先查检索。** 90% 的「RAG 效果不好」是检索的问题，不是生成的问题。
> 模型拿到正确的证据时，生成质量通常没问题。

详见 [`10-评测与度量.md`](./10-评测与度量.md)。

---

## 8. 一条最小可用的 RAG 实现

```python
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

class NaiveRAG:
    def __init__(self, embed_model_name="BAAI/bge-m3",
                 llm_name="Qwen/Qwen2.5-7B-Instruct"):
        self.embedder = SentenceTransformer(embed_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype="auto", device_map="auto")

    def build_index(self, chunks: List[str], save_path="index.npy"):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            chunks, normalize_embeddings=True,
            batch_size=32, show_progress_bar=True)
        np.save(save_path, self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q = self.embedder.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, q[0])     # 余弦相似度（已归一化）
        idx = np.argsort(-scores)[:top_k]
        return [self.chunks[i] for i in idx]

    def generate(self, query: str, contexts: List[str]) -> str:
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = f"""请仅根据以下资料回答问题，并在答案中用 [1][2] 标注引用来源。
如果资料中没有答案，请直接说"根据现有资料无法回答"。

资料：
{ctx}

问题：{query}

答案："""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

**这个实现只有 30 行，但它已经是完整的 RAG 闭环。** 后续所有优化都是在某个环节上做加法。

---

## 9. 工程结论

1. **先做对，再做多**。一个跑通的 BM25 + 向量混合检索 + 重排的系统，比一个堆了十个模块但检索不准的系统强得多。
2. **检索质量是瓶颈**。排查 RAG 问题时，第一个动作永远是「人工看检索到了什么」。
3. **上下文要少而精**。20 条噪音不如 3 条精准（Lost in the Middle）。
4. **结构化数据不要硬塞进 RAG**。走 SQL/API。
5. **RAG 不是万能的**。小数据集、长上下文够用、创作类任务，直接 LLM 更好。
6. **从 Naive 开始，用评测驱动迭代**。没有评测集，就无法判断改动是进步还是退步。

---

## 🔗 相关

- 解析 → [`02-文档解析与预处理.md`](./02-文档解析与预处理.md)
- 切分 → [`03-切分策略.md`](./03-切分策略.md)
- 评测 → [`10-评测与度量.md`](./10-评测与度量.md)
- 高级架构 → [`09-高级RAG架构.md`](./09-高级RAG架构.md)
