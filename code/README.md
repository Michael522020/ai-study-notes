# 代码：AI Agent 与 AI 招聘应用的可运行实现

> 与 [`05-Agent与智能体`](../05-Agent与智能体/) 、[`06-AI招聘应用`](../06-AI招聘应用/) 两章笔记配套。
> **笔记讲的是「为什么这么做」，这里的代码是「照着做一遍」。**

## 两个可运行项目

| 目录 | 对应笔记 | 跑什么 |
|---|---|---|
| `agent-lab/` | 05 章（AI Agent / Agent Bot / Super Agent） | 三种 Agent 形态的运行时、工具层、离线评测集 |
| `hr-agent/` | 06 章（简历筛查 / 模拟面试） | JD 能力模型、筛查流水线、模拟面试引擎、偏差审计 |

两者共用 `agentkit/` 里的 provider 抽象、成本核算、轨迹记录与 JSON Schema 校验。

## 零依赖跑起来

```bash
# 离线评测（不需要 API key，不需要 pip install）
python code/agent-lab/run.py eval
python code/hr-agent/run.py eval

# 看一份完整的筛查报告与面试报告
python code/hr-agent/run.py eval --show-report
```

Windows 控制台默认编码是 ACP=936，中文会花屏。脚本里已经 `reconfigure(encoding="utf-8")`，
如果仍然乱码，加一个环境变量：

```bash
PYTHONIOENCODING=utf-8 python code/hr-agent/run.py eval
```

## 跑真实模型

```bash
pip install -r code/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
python code/agent-lab/run.py chat --profile bot
python code/agent-lab/run.py chat --profile superagent
```

`chat` 里的 Super Agent 会问「目标」和「验收标准」，然后自己拆解、执行、验收。

## 跑测试

```bash
cd code
python -m pytest            # pyproject.toml 里已配好 pythonpath
```

当前基线：**agent-lab 21 项 + hr-agent 25 项测试全绿**；
两套离线评测 `pass_rate = 1.0`，hr-agent 的硬门槛判对率、证据可回指率、偏差不变性均为 100%。

---

## ⚠️ 关于「离线评测」的重要边界

这套离线评测用 `ScriptedProvider` 替代模型：**工具是真的在跑、账是真的在算、轨迹是真的在写**，
只有「模型这一步会做什么」是写死的脚本。

所以它能证明：

- 编排逻辑正确（工具分发、错误回喂、闸门触发、停止条件）；
- 账目正确（步数、工具调用、成本的累计与上限）；
- 状态语义正确（`ok` / `error` / `blocked` / `needs_human` 的划分）；
- 结论一致性正确（屏蔽前后得分必须相等、证据必须能回指原文）。

它**不能**证明模型能力够。模型答得好不好，必须用真实 API 跑同一套断言 ——
那一栏指标（答案质量、真实延迟、真实成本）在离线报表里是空的，这是刻意的。

## 目录

```
code/
├── agentkit/              共享内核
│   ├── config.py          配置 + 预算闸门
│   ├── costs.py           定价表与成本核算
│   ├── provider.py        AnthropicProvider（真实） / ScriptedProvider（离线）
│   ├── schema.py          JSON Schema 严格化与校验
│   ├── tracing.py         JSONL 轨迹
│   └── types.py           provider 无关的数据类型
├── agent-lab/
│   ├── agentlab/
│   │   ├── tools.py       工具层：定义即契约
│   │   ├── runtime.py     手写 agentic loop
│   │   ├── profiles/      bot / agent / superagent
│   │   └── evals/         离线评测集
│   └── tests/
├── hr-agent/
│   ├── hragent/
│   │   ├── extract.py     txt / md / docx / pdf 文本抽取
│   │   ├── rules.py       能力图谱 + 敏感字段词典
│   │   ├── jd.py          JD → 能力模型
│   │   ├── matching.py    能力匹配 + 证据回指校验
│   │   ├── screening.py   筛查流水线
│   │   ├── bias.py        敏感字段屏蔽与偏差审计
│   │   ├── interview.py   模拟面试引擎
│   │   ├── report.py      Markdown 报告
│   │   └── evals/         合成夹具与评测
│   └── tests/
└── pyproject.toml         pytest 的 pythonpath 配置
```

## 依赖

- **离线模式**：只需要 Python 3.10+（标准库即可跑通全部测试与评测）。
- **真实模式**：`anthropic>=2.7`（`requirements.txt`）。
- `hr-agent` 的 PDF 抽取是自带的尽力实现，**只处理有文本层的 PDF**；
  扫描件会返回 `needs_ocr`，不会静默返回空字符串。

## 数据声明

`hr-agent` 中的 JD 与简历**全部为合成虚构数据**，不含任何真实个人信息、
不来自任何真实招聘渠道，仅用于跑通流程与回归测试。
