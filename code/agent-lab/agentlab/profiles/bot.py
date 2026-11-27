"""Agent Bot：面向终端用户的产品化对话形态。

产品视角的关键差异：**Bot 的失败成本是「用户体验差」，不是「任务没做完」**。
所以 Bot 的设计重心是：
- 场景收窄（只答售前售后政策、订单状态）；
- 一轮成本封顶（Bot 并发量大，单轮成本 × QPS 才是账单）；
- 随时可转人工，且转人工必须是**产品动作**而不是报错；
- 输出长度受控（超过 4 行就该给按钮/链接，而不是继续写小作文）。

不要用 Bot 的形态做 Agent 的活：Bot 的循环被刻意压到 3~4 步，
任务稍微复杂一点就会「兜底转人工」，那不是 Agent。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentkit.config import Budget
from agentkit.provider import Provider
from agentkit.tracing import Tracer

from ..runtime import AgentRuntime, AgentResult
from ..tools import ToolRegistry, build_registry

BOT_TOOLS = [
    "search_knowledge",
    "list_orders",
    "lookup_order",
    "calculate",
    "draft_reply",
    "create_ticket",
    "escalate_to_human",
]

BOT_SYSTEM_PROMPT = """你是外贸业务团队的客服助手，服务对象是海外 B 端客户。

回答规则：
1. 只回答业务政策、订单状态、售后流程三类问题；其余一律说明超出服务范围。
2. 涉及金额、补偿比例、账期的数字，必须先用 calculate 工具算，再写进回答；不要心算。
3. 任何补偿、折扣、退款承诺都超出你的权限 —— 说明需要主管审批，并转人工。
4. 不确定的事就说不知道，不要编造政策条款。知识库里没有的，不要当成有。
5. 回答控制在 4 行以内，给出下一步动作（查订单号 / 转人工 / 看政策）。

工具使用：
- 需要政策条款 → search_knowledge；
- 需要订单状态 → lookup_order（不确定订单号时先 list_orders）；
- 只出草稿不发送，需要发信时用 draft_reply。
"""

BOT_BUDGET = Budget(max_steps=4, max_tool_calls=6, max_cost_usd=0.02)


@dataclass
class BotProfile:
    """Bot 的产品参数。这些数字进监控看板，不是可选项。"""

    name: str = "agent-bot"
    tools: list[str] = field(default_factory=lambda: list(BOT_TOOLS))
    max_steps: int = 4
    max_tool_calls: int = 6
    max_cost_usd: float = 0.02
    #: 单轮成本红线：超过就降级到 FAQ 直答
    cost_alert_usd: float = 0.01
    #: 连续两轮没解决就转人工
    escalate_after_turns: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tools": list(self.tools),
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_cost_usd": self.max_cost_usd,
            "cost_alert_usd": self.cost_alert_usd,
            "escalate_after_turns": self.escalate_after_turns,
        }


def build_bot(
    provider: Provider,
    *,
    profile: BotProfile | None = None,
    tracer: Tracer | None = None,
    model: str = "claude-opus-5-5",
    effort: str = "medium",
    allow_side_effects: bool = False,
) -> AgentRuntime:
    prof = profile or BotProfile()
    registry: ToolRegistry = build_registry(prof.tools)
    return AgentRuntime(
        provider=provider,
        tools=registry,
        system_prompt=BOT_SYSTEM_PROMPT,
        model=model,
        effort=effort,
        budget=Budget(
            max_steps=prof.max_steps,
            max_tool_calls=prof.max_tool_calls,
            max_cost_usd=prof.max_cost_usd,
        ),
        tracer=tracer,
        allow_side_effects=allow_side_effects,
    )


def run_bot_turn(
    runtime: AgentRuntime,
    user_input: str,
    *,
    history: list[dict[str, Any]] | None = None,
) -> AgentResult:
    """跑一轮对话并做产品层后处理。

    后处理不是装饰：Bot 的输出必须始终「可执行」，哪怕模型返回的是一句道歉。
    """
    result = runtime.run(user_input, history=history)
    if result.status == "completed" and not result.final_text.strip():
        result.status = "needs_human"
        result.stop_reason = "empty_answer"
    return result
