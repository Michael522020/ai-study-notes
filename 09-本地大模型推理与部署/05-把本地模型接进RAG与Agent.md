# 把本地模型接进 RAG 与 Agent

> **核心问题**：本地模型能不能接进现有的 LLM 应用？如果能，改动量有多大？接进来之后，「成本闸门」还有意义吗？

本章以本仓库自带的 `code/agentkit` 为例，说明如何把 Ollama 作为一个**第三条 provider 通道**接进去——现有的两条是「真实 Claude API」和「离线确定性替身」。

**数据口径**：本章代码与接口描述基于本仓库 `code/agentkit/` 的实际实现（`provider.py` / `config.py` / `costs.py` / `types.py`）。**本章不新增仓库代码**，只给出接入设计与需要改动的位置。

---

## 1. 为什么「第三条通道」而不是「把本地模型换上去」

先看本仓库已有的两条通道：

| 通道 | 实现 | 用途 |
|---|---|---|
| **真实** | `AnthropicProvider` → Claude API | 效果评测、生产 |
| **离线替身** | `ScriptedProvider` → 按脚本回放 | 跑通编排逻辑、回归测试、CI |

`provider.py` 的文件头把这个边界写得很清楚：

> 离线替身**只替代模型**，工具是真的在跑、账是真的在算、轨迹是真的在写。
> 所以离线能证明「编排是对的」，**不能证明「模型能力够」**。

**接本地模型的正确姿势不是「替换」，而是「并列」**：

```
                    ┌──────────────────────────┐
                    │   Provider (Protocol)     │
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────┐   ┌───────────▼──────────┐   ┌────────▼─────────┐
│AnthropicProvider│   │  OllamaProvider(新)   │   │ScriptedProvider  │
│  真实云端 API    │   │  本机 OpenAI 兼容端点  │   │  离线确定性替身   │
│  能力最强 / 花钱  │   │  免费 / 弱 / 隐私      │   │  不花钱 / 不真实   │
└────────────────┘   └──────────────────────┘   └──────────────────┘
```

**三条通道各有明确的适用场景，不是替代关系：**

| 场景 | 用哪条 | 理由 |
|---|---|---|
| 生产主链路 | Anthropic | 能力差距是硬约束 |
| **敏感数据不能出网** | **Ollama** | 合规上的硬要求，不是优化 |
| **批量离线任务（脱敏、打标、归类）** | **Ollama** | 边际成本为零，可并发 |
| CI / 回归测试 | Scripted | 确定性，不能有网络抖动 |
| **本地开发不想花钱** | **Ollama** | 开发期反复调用，成本敏感 |
| 效果兜底 | Ollama 做降级 | 详见 §5 |

---

## 2. 接口对齐：`Provider` Protocol 要求什么

```python
class Provider(Protocol):
    name: str

    def generate(
        self, *,
        model: str,
        messages: Sequence[dict[str, Any]],
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
        effort: str = "high",
    ) -> LLMResponse: ...
```

**返回结构 `LLMResponse` 的四个字段**，决定了本地 provider 要做多少映射工作：

| 字段 | Anthropic 侧 | Ollama（OpenAI 兼容）侧 | 映射难度 |
|---|---|---|---|
| `blocks` | `TextBlock` / `ThinkingBlock` / `RedactedThinkingBlock` / `ToolUseBlock` | `message.content`（字符串）+ `message.thinking`（字符串）+ `message.tool_calls`（数组） | **中**：要把 OpenAI 的 `tool_calls` 数组拆成一个个 `ToolUseBlock` |
| `stop_reason` | `msg.stop_reason` | `finish_reason` | 低 |
| `usage` | `input_tokens` / `output_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens` | `usage.prompt_tokens` / `usage.completion_tokens`（**无 cache 字段**） | 低 |
| `parsed` | `parsed_output`（服务端按 schema 解析） | `response_format` 的 json_schema 输出 | 中 |

**几个需要显式处理的差异：**

| 差异 | 说明 | 怎么办 |
|---|---|---|
| **缓存 token** | Anthropic 有 prompt caching，读写分别计价；**Ollama 本地没有这个概念** | `cache_read_input_tokens` / `cache_creation_input_tokens` 恒为 0 |
| **thinking 签名** | Anthropic 的 `ThinkingBlock` 带 `signature`（多轮续接时要回传校验） | 本地模型没有签名的概念，`signature` 置空 |
| **effort 档位** | Anthropic 用 `output_config.effort`（low/medium/high/xhigh/max） | Ollama 的 `--think` 支持 `true/false/high/medium/low` —— **能对上 high/medium/low，xhigh/max 要降级映射** |
| **结构化输出** | `output_schema` → 服务端 `json_schema` | Ollama 的 OpenAI 兼容端点支持 `response_format: {"type": "json_schema", ...}`；**若版本不支持，必须退回「提示词里塞 schema + 客户端解析 + 失败重试」** |

---

## 3. 三个必须显式处理的配置问题

### 3.1 价格表里没有本地模型 → 直接抛异常

`costs.py` 的 `price_of()` 是**硬失败**的：

```python
def price_of(model: str) -> ModelPrice:
    try:
        return PRICING[model]
    except KeyError as exc:
        raise KeyError(f"模型 {model!r} 没有登记价格，无法核算成本。已登记：{sorted(PRICING)}")
```

注释写得很清楚：**「未知模型直接报错，不做『按最贵算』的保守估计」**。这是个好设计——它保证成本数字永远可信。

但接本地模型会直接撞上它：`qwen3:4b` 不在 `PRICING` 里 → **任何一次调用都抛 `KeyError`**。

**正确做法**：给本地模型登记一条**单价为 0** 的条目，而不是绕过 `price_of()`。

```python
PRICING = {
    ...,
    # 本地模型：边际成本为零。但**延迟与并发**才是真正的成本，见下方「工程结论」。
    "qwen3:4b": ModelPrice("qwen3:4b", 0.0, 0.0, 0.0, 0.0, 262_144, "local"),
}
```

**但要小心这带来的副作用**——见 §4。

### 3.2 `require_api_key()` 会拦人

`config.py` 里：

```python
def require_api_key(self) -> str:
    if not self.api_key:
        raise ConfigError(...)
```

真实 provider 在缺 key 时**不会静默降级**，而是带着修复命令抛错——这是 07 章「失败语义要显式」那条规则的落地。

Ollama 本地端点不需要真 key（OpenAI 兼容接口要求填一个占位符）。所以 `OllamaProvider` **不应该调用 `require_api_key()`**，而是：

- 默认填一个占位 key（`"ollama"`），或
- 允许 `api_key=None` 但要求 `base_url` 必须是 `localhost`/`127.0.0.1`（**防止把「无 key」误配成远端服务**）

> 💡 后者更稳：**本地通道的「无凭据」状态不应该和「远端通道的缺 key」混为一谈**。前者是正常配置，后者是配置错误。

### 3.3 本机环境的三个坑

| 坑 | 症状 | 修法 |
|---|---|---|
| **`python` 是 3.11.9，但 `pip` 指向 3.14** | `pip install X` 装到了另一个解释器上，`import` 不到 | 一律用 `python -m pip install X` |
| **控制台 ACP=936** | Python 打印中文花屏；**更糟的是脚本经管道传给 curl/子进程时中文被按 GBK 编码** | 跑脚本设 `PYTHONIOENCODING=utf-8`；发 HTTP 请求时用 `json.dumps(..., ensure_ascii=True)` 生成**纯 ASCII 请求体** |
| **没有 `ANTHROPIC_API_KEY`** | 真实 provider 起不来 | 这恰恰是本地通道存在的理由之一 |

> ⚠️ **第二条值得单独强调**，因为它在写笔记/做测试时真的踩到过：用 `curl` 发带中文的 JSON，Shell 管道把 Python 的 stdout 按 GBK 编码传出，**服务端收到的是乱码**。而**模型会对着乱码照样生成出一段通顺的回答**——不报错，只是答非所问。
>
> 本章的实测数据就是先踩了这个坑才重做的。**验收标准应该是「模型能复述你的问题」，而不是「请求返回了 200」。**

---

## 4. 成本闸门会失效——这是接本地模型最大的设计影响

本仓库 `code/agentlab/` 与 `code/hr-agent/` 都有**预算闸门**（超过 token 预算就中止）。它的依据是 `costs.py` 的美元单价。

**本地模型单价为 0 → 美元预算永远触发不了 → 闸门形同虚设。**

这不是 bug，是**换了成本单位之后必须换护栏**。本地通道真正的成本是：

| 真实成本 | 度量方式 | 对应的闸门设计 |
|---|---|---|
| **用户等待时间** | TTFT / 总生成时长 | 延迟闸门：超过 N 秒就中止或降级 |
| **本机资源占用** | 并发请求数 × 驻留模型数 | 并发闸门：`OLLAMA_NUM_PARALLEL` 限流 |
| **上下文膨胀** | input_tokens | **token 上限闸门（这个可以直接复用）** |
| **电费** | 整机功耗 × 时长 | 通常不必设闸门，但要记在账上 |

> 💡 **一条重要的设计结论**：
> **token 预算闸门在本地通道下依然有效**（token 数和单价是两回事），**美元预算闸门则完全失效**。
> 所以正确的做法是：**把闸门的判据从「美元」改成「token + 延迟」两个维度**，而不是把本地模型的成本也硬折算成美元（折算需要一个假想的电价假设，数字会变得不可验证——而本仓库的成本数字一贯要求「口径必须可追溯」）。

**同样地，`Usage` 里的 cache 字段在本地通道恒为 0** → 任何依赖「缓存命中率」的指标在本地通道下没有意义，做报表时要分通道统计，别把它们混在一张表里。

---

## 5. 什么该下发到本地，什么该留在云端

### 5.1 本地模型的三个真实优势

| 优势 | 说明 |
|---|---|
| **边际成本为零** | 批量任务（几千条打标）不用按 token 计价 |
| **数据不出网** | 合同、客户资料、源码、内部文档——**合规要求，不是偏好** |
| **无速率限制** | 云端 API 的 RPM/TPM 限流是本地不存在的约束 |

### 5.2 本地模型的两个硬劣势

| 劣势 | 量化（以本机 qwen3:4b 为例） |
|---|---|
| **能力差距** | 4B 级模型在复杂推理、长上下文、工具调用稳定性上与前沿闭源模型有量级差距 |
| **上下文长度受限** | 服务端实际生效的 `num_ctx` 受限于本机显存/内存（本机不修核显那件事就只有 4k，见第 01 章） |

### 5.3 一条好用的分界线

```
数据能不能出网？
├─ 不能 → 本地（且用更大的本地模型 + 人工复核）
└─ 能   → 任务需要前沿模型的能力吗？
          ├─ 需要 → 云端
          └─ 不需要（分类 / 抽取 / 改写 / 打标 / 路由）
                → 本地（量大、简单、重复 → 边际成本优势最大）
```

**一个具体的例子**：Agent 里的「路由」——判断用户问题该交给哪个下游模型——这是一个**分类任务**，用前沿大模型来做是浪费。本地 4B 模型足够，**而且它要读的全部是用户输入的摘要，正好落在「数据不出网」这一侧**。

---

## 6. 工程结论

1. **本地模型是并列的第三条通道，不是替换**。`Provider` Protocol 的存在就是为了让这件事变成「加一个实现」而不是「重写一层」。
2. **必须给本地模型登记 0 美元价格**，否则 `price_of()` 会直接抛异常。而且**不要**因此就认为成本闸门还有效——**美元闸门在本地通道下完全失效**。
3. **把护栏从「美元」换成「token + 延迟 + 并发」**。token 闸门可以直接复用，延迟和并发是新加的。
4. **本地通道的「无 key」状态要和远端的「缺 key」分开**。本地没有凭据是正常配置，远端缺 key 是配置错误——前者不该抛错。
5. **本机三个环境坑**：`python -m pip` 而不是 `pip`；`PYTHONIOENCODING=utf-8`；**发 HTTP 时用 `ensure_ascii=True` 生成 ASCII 请求体**——中文乱码不会报错，只会答非所问。
6. **分界线是「数据能不能出网」+「任务要不要前沿能力」**，不是「本地模型好不好」。

---

## 🔗 与其他主线的关系

| 主题 | 去哪看 |
|---|---|
| 上一章：本地模型的对齐边界 | [`./04-本地模型的对齐边界.md`](./04-本地模型的对齐边界.md) |
| 上一章：思考模型与三处 think 开关 | [`./03-思考模型与本地推理的工程问题.md`](./03-思考模型与本地推理的工程问题.md) |
| provider 抽象的原始实现（`provider.py` / `costs.py` / `config.py`） | [`../code/agentkit/`](../code/agentkit/) |
| 成本模型与定价口径的原始说明 | [`../07-Agent与智能体/01-AI-Agent-通论与工程落地.md`](../07-Agent与智能体/01-AI-Agent-通论与工程落地.md) |
| 失败语义要显式（为什么缺 key 不静默降级） | [`../07-Agent与智能体/01-AI-Agent-通论与工程落地.md`](../07-Agent与智能体/01-AI-Agent-通论与工程落地.md) |
| RAG 侧的检索与重排（本地模型常用于重排） | [`../03-RAG/06-检索与重排.md`](../03-RAG/06-检索与重排.md) |
| RAG 侧的评测与度量 | [`../03-RAG/10-评测与度量.md`](../03-RAG/10-评测与度量.md) |
