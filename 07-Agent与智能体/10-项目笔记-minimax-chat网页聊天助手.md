# 10 · 项目笔记：minimax-chat 网页聊天助手

> **定位**：能直接抄走的一个最小可用的"浏览器 + 流式 LLM"工程模板
> **来源**：作者本机私有项目（仓库未公开），本笔记为公开学习笔记；并入时已按公开分级脱敏本机绝对路径，**不含任何密钥、凭据或真实模型地址**。
> 本章原为独立仓库 `agent-learning` 的项目笔记，合并至 07 主线时仅调整标题与索引格式。

---

## 1. 它是什么

浏览器里和多轮聊天的页面。后端是 FastAPI，把 MiniMax 的 OpenAI 兼容 SSE 包一层再喂给前端。前端用原生 HTML/CSS/JS（**不打包**），按节流渲染 Markdown。

> 说明：本笔记里不写任何 `.env` 里的真实密钥、真实模型名 host、示例 BASE_URL，详见配套文档或仓库自身的 `报告.md`。

典型流程：浏览器组装 messages → `POST /api/chat` → 校验裁剪 → 后端开一个 httpx 流式请求 → 逐行解析 → 每个 delta 转成 SSE 推到浏览器 → 浏览器按 60ms / 40 字重绘 → 收到 `done` 或用户点"停止"。

---

## 2. 顶层架构

```
server/
├─ config.py            —— 读 .env，校参，回退有警告不退出
├─ minimax_client.py    —— httpx 流式 + 行解析，产出统一内部事件
├─ main.py              —— 路由、消息裁剪、内部事件→SSE、托管 web/
└─ web/
   ├─ index.html        —— 骨架 + 主题闪屏抑制脚本
   ├─ styles.css        —— 深色 / 浅色 / 抽屉
   └─ app.js            —— 存储、Markdown、网络、视图、交互
```

三段清晰分责：**配置只读不联网；客户端只做流；路由做协议转换与裁剪**。

---

## 3. 最值得抄的几个设计

### 3.1 边收边推 + 断开即停（流式最关键的两件事）

```python
# server/minimax_client.py（要点复述）
queue: asyncio.Queue[str | object] = asyncio.Queue()

async def pump():
    async for line in response.aiter_lines():
        await queue.put(line)
    await queue.put(_END)
pump_task = asyncio.create_task(pump())

while True:
    # 等下一行最多 100ms，期间探测浏览器断开
    if await client_gone():
        await _close_response(response)
        raise ClientDisconnected()
    try:
        item = await asyncio.wait_for(queue.get(), timeout=0.1)
    except asyncio.TimeoutError:
        continue
    ...
```

要点：

- **不能等上游收完再返回**，否则打字机效果没了
- **断开要立刻 aclose 上游**，否则继续吃 token，浪费钱
- 主循环里 `wait_for(..., 0.1)` 短超时是关键——既不阻塞渲染，也能在思考间歇里及时发现断开

### 3.2 SSE 协议契约极简

内部事件只有四种，前端只看到这四种：

```
delta_thinking  → thinking   （进默认折叠的"思考过程"）
delta_content   → content    （正文）
error           → error      （红条 + 重试）
done            → done       （不带 text，流的边界）
```

错误一定是**先 error 后 done**——这样前端不会卡在加载态。**空文本不向前端发**，节省 paint 次数。

### 3.3 ASGI 2.3 的坑

```python
class SseResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await self.stream_response(send)   # 只写
        except OSError:
            logger.info("浏览器连接已关闭")
```

Starlette 的 `StreamingResponse` 会**另起一个任务读 receive**——而业务里的 `is_disconnected()` 也在读 receive，**两处抢同一条消息**。解法是**自定义响应只调用 `stream_response(send)`，不再主动读 receive**，把断开探测交给业务层在循环内做。

### 3.4 历史裁剪三道关

```python
# server/main.py:prepare_messages 摘要
- 单条 ≤ 100_000 字，否则直接 400
- 空白 content 丢弃
- 多个 system 只留第一条，且放最前
- 对话只留最近 40 条
- 总字符超 120_000 时从最旧的对话丢，但最后一条 user 一定留着
- 至少要有一条 user，否则拒绝
```

教训：**LLM 上下文不是"越长越好"，是"越相关越好"**。业务侧裁剪往往比模型侧注意力窗口更关键。

### 3.5 前端节流渲染

```js
// app.js 节流要点
PAINT_INTERVAL_MS = 60;
PAINT_CHAR_STEP   = 40;
SCROLL_GAP        = 80;
```

- **60ms 或每 40 字**才刷一次 DOM，否则每个 token 都重排 layout 会卡
- 只有距底部 < 80px 时才"自动跟随滚动"——用户向上回看时不被打断
- 已生成的文字在停止 / 刷新后仍在；不显示空气泡

### 3.6 输入法与 IME 兼容

```js
input.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" || event.shiftKey) return;
    if (event.isComposing || event.keyCode === 229) return;  // 组字中
    event.preventDefault();
    Actions.send();
});
```

中文输入法组字（候选词）时按 Enter 是"选词"不是"发送"。漏掉这一行就会反复误发。

### 3.7 Markdown 安全三连

```
marked.parse(...)  →  DOMPurify.sanitize(..., FORBID_TAGS, FORBID_ATTR)  →  innerHTML
```

`FORBID_TAGS` 黑名单 `[img, iframe, object, embed, form, input, style, script]`、`FORBID_ATTR: ["style"]`。模型输出的 markdown 仍可能带 `<script>` 或 `onerror=`，必须清洗后插入。**绝不能让"助手消息"成为前端 XSS 入口**。

### 3.8 CDN 三级降级

```js
// app.js ensureLibraries()
jsDelivr → BootCDN → unpkg → 纯文本回退（页面不白屏）
```

标记主题 `data-hljs="dark|light"` 来切换 highlight.js 的 css，深浅色切换只动 `disabled`。

### 3.9 localStorage 版本化

```js
var STORAGE_KEY = "minimax_chat_sessions_v1";
```

`version: 1` 字段显式保留；`normalizeData()` 在读取时**永远走兼容性转换**，老数据不会让页面崩溃。`clearAll()` 不是删一个 key，而是重建一份空数据并 `saveNow()`——避免读取到旧格式留下的脏数据。

---

## 4. 配置与启动期常见雷

- **占位符识别**：`""`、`your_api_key_here`、`your-api-key`、`sk-xxx` 都被视为未配置——`_PLACEHOLDER_KEYS` 列出，避免有人把示例文件拷过去就当成已配置
- **非法配置回退 + 警告**：温度超 [0,2]、max_tokens 超界、模型不在白名单等都回退到默认，但**不退出进程**——只把警告打到 startup log，方便排查
- **脱敏入库**：上游错误时日志先 `_redact(raw, MINIMAX_API_KEY)` 再写，最多 500 字；前端只看到中文短说明
- **断开的状态机**：`controller.aborted` 时 stop 分支把 streaming=false、stopped=true、error 清空；前端用专门的 `msg-error` 与 `msg-note("已停止生成")` 区分显示

---

## 5. 目录导读

| 想看什么 | 看哪里 |
| --- | --- |
| 环境变量、默认、校验 | `server/config.py` |
| httpx 流式与行解析 | `server/minimax_client.py` |
| 路由、消息裁剪、SSE 头 | `server/main.py` |
| 前端骨架、主题闪屏 | `server/web/index.html` |
| 流式渲染、节流、停止 | `server/web/app.js` |
| 复现清单 | `.cursor/skills/minimax-web-chat/SKILL.md` |
| 字段级契约 | `.cursor/skills/minimax-web-chat/reference.md` |
| 完整复现报告 | `报告.md` |

---

## 6. 与 04–08 章的对照

| 对应章节 | 在 minimax-chat 中的体现 |
| --- | --- |
| 04 · Agent 构建 | 不是 agent，是 streaming chat；工具只有"调用模型"这一个 |
| 04 · ReAct / Plan & Execute | 都没有——单轮对话不需要 |
| 04 · Reflection | 没有；但 thinking 和 content 分两个通道收，可视为模型自带的"先想后说" |
| 04 · 思维方式 | 单一 CoT；UI 层做了大量"防抖 / 丢弃 / 降级"，是同一种思路在前端 |

主要可学项其实是**工程面而非算法面**——怎样把"一个会吐 SSE 的接口"做成一个"在手机上也能用、断网也不会爆、模型再抽风也不会卡死"的网页。

---

## 7. 一句话总结

**这个项目的精髓是"边收边推 + 断开即停 + 错误降级 + 离线友好"。** 算法几乎不用写，所有聪明都花在"流怎么不断又不空转、用户怎么告别也不卡死、CDN 挂了怎么不白屏"——这是 streaming LLM 应用最容易翻车也最容易藏住水平的地方。

---

## 8. ⚠️ 不要做的事

- 不要把 `.env`、`示例 BASE_URL`、占位符以外的任何真实凭据写进笔记 / commit / 日志
- 不要在前端任何地方放过密钥、Token、Cookie——所有鉴权只能放服务端
- 不要在没读 `SseResponse.__call__` 之前把业务代码里的 `await self.stream_response(send)` 改成 `super().__call__(...)`，否则会两个 receive 抢消息
- 不要让 CDN 全部失败后页面变白——必须降级到纯文本 + `[hidden] { display: none !important; }`
