"""通用 AI Agent：能拆解任务、多步推进、记住中间结论。

与 Bot 的分界线是**目标的不确定性**：Bot 面对的是「已经知道答案范围的问题」，
Agent 面对的是「目标清楚但路径未知的问题」——需要先规划，边走边修正。

工程上多出来的东西：
1. 规划与执行的分离（plan_tasks → execute_task），让中间产物可检查、可回放；
2. 跨轮记忆（AgentMemory），下一轮不重复问已经查过的事实；
3. 更宽的预算，但同时更严的**单任务成本上限**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentkit.config import Budget
from agentkit.provider import Provider
from agentkit.tracing import Tracer

from ..runtime import AgentRuntime, AgentResult
from ..tools import ToolRegistry, build_registry

AGENT_TOOLS = [
    "search_knowledge",
    "read_knowledge_doc",
    "list_orders",
    "lookup_order",
    "calculate",
    "draft_reply",
    "plan_tasks",
    "execute_task",
    "check_acceptance",
    "escalate_to_human",
]

AGENT_SYSTEM_PROMPT = """你是外贸业务团队的运营助手，需要完成多步骤任务。

工作方式：
1. 先把任务拆成 3~6 个子任务并用 plan_tasks 登记，再逐条用 execute_task 执行；
   不要一上来就动手，遇到依赖关系要按顺序来。
2. 需要外部事实（政策、订单）必须查工具，不许凭印象作答。
3. 涉及金额计算用 calculate 工具，不心算。
4. 每个子任务完成后，把关键结论写进 output —— 后续步骤要靠它，不要重复查。
5. 写操作（建单、外发）只出草稿，由人确认；需要承诺金额时转人工。
6. 全部子任务完成后，用 check_acceptance 对照验收标准自检一遍。
"""


@dataclass
class AgentMemory:
    """跨轮记忆。

    只存**结论性事实**（键值对），不存整段对话 —— 整段对话由 history 负责。
    这是上下文预算管理的第一步：可结构化的别用自然语言塞。
    """

    facts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def remember(self, note: str) -> None:
        self.notes.append(note)

    def as_system_context(self) -> str:
        if not self.facts and not self.notes:
            return ""
        lines = []
        if self.facts:
            lines.append("已知事实：")
            lines.extend(f"- {k}: {v}" for k, v in self.facts.items())
        if self.notes:
            lines.append("过程记录：")
            lines.extend(f"- {n}" for n in self.notes[-10:])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"facts": dict(self.facts), "notes": list(self.notes)}


@dataclass
class AgentProfile:
    name: str = "ai-agent"
    tools: list[str] = field(default_factory=lambda: list(AGENT_TOOLS))
    max_steps: int = 14
    max_tool_calls: int = 28
    max_cost_usd: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tools": list(self.tools),
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_cost_usd": self.max_cost_usd,
        }


def build_agent(
    provider: Provider,
    *,
    profile: AgentProfile | None = None,
    tracer: Tracer | None = None,
    model: str = "claude-opus-5-5",
    effort: str = "high",
    allow_side_effects: bool = False,
) -> AgentRuntime:
    prof = profile or AgentProfile()
    registry: ToolRegistry = build_registry(prof.tools)
    return AgentRuntime(
        provider=provider,
        tools=registry,
        system_prompt=AGENT_SYSTEM_PROMPT,
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


def run_agent_task(
    runtime: AgentRuntime,
    goal: str,
    *,
    memory: AgentMemory | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AgentResult:
    """执行一个完整任务，并把已知事实沉淀进记忆。

    记忆的注入方式很关键：作为 system 追加，而不是塞进 user 消息 ——
    塞进 user 消息会让模型把记忆当成「用户说的」，而不是「系统已知的」。
    """
    mem = memory or AgentMemory()
    extra = mem.as_system_context()
    result = runtime.run(goal, history=history, extra_system=extra or None)
    for call in result.tool_calls:
        if call.name in ("execute_task", "plan_tasks") and call.status == "ok":
            mem.remember(f"{call.name}: {call.message}")
    return result
