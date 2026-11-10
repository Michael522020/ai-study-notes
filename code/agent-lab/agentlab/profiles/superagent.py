"""Super Agent：长任务自治形态。

形态特征（一句话：**目标模糊、路径未知、结果要验收**）：
- 输入不是一句话问询，而是一个「目标 + 验收标准」；
- 过程跨越多阶段，允许中途失败、中断续跑；
- 结束不由模型说「我做完了」决定，而由**验收标准逐条判定**决定。

四个必备机制：
1. 分解 —— 目标拆成可独立验证的子任务；
2. 检查点 —— 每完成一步落盘，进程挂了从断点续跑；
3. 上下文治理 —— 长任务必然撑爆上下文，必须有压缩策略；
4. 验收 —— 机器可判的标准交给 check_acceptance，不靠模型自述。

与「Agent 跑得更久」的区别：Agent 是**一次会话内把一件事做完**，
Super Agent 是**把一件事做成一个可中断、可恢复、可审计的作业**。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agentkit.config import Budget
from agentkit.provider import Provider
from agentkit.tracing import Tracer

from ..runtime import STATUS_COMPLETED, STATUS_NEEDS_HUMAN, AgentResult, AgentRuntime
from ..tools import ToolRegistry, build_registry

SUPER_AGENT_TOOLS = [
    "search_knowledge",
    "read_knowledge_doc",
    "list_orders",
    "lookup_order",
    "calculate",
    "plan_tasks",
    "execute_task",
    "check_acceptance",
    "save_checkpoint",
]

SUPER_AGENT_SYSTEM_PROMPT = """你是长任务执行器，接收一个目标和一组验收标准。

执行纪律：
1. 第一步用 plan_tasks 把目标拆成 3~6 个可独立验证的子任务；
2. 彼此独立的子任务，在**同一轮**里并行发起多个 execute_task 调用，不要排队；
3. 每个子任务完成后用 save_checkpoint 保存进度；
4. 涉及金额用 calculate，不心算；
5. 全部子任务完成后用 check_acceptance 逐条对照验收标准；
6. 任何一步卡住且无法自行解决，调用 escalate_to_human 说明卡点，不要硬撑。
"""

#: 本地压缩阈值：上下文里累计多少字符后触发历史裁剪
DEFAULT_COMPACT_THRESHOLD = 6000


@dataclass
class SuperAgentProfile:
    name: str = "super-agent"
    tools: list[str] = field(default_factory=lambda: list(SUPER_AGENT_TOOLS))
    max_steps: int = 40
    max_tool_calls: int = 60
    max_cost_usd: float = 3.0
    checkpoint_path: Path | None = None
    compact_threshold: int = DEFAULT_COMPACT_THRESHOLD
    #: 保留最近多少条消息不被裁剪
    compact_keep_last: int = 6

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tools": list(self.tools),
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_cost_usd": self.max_cost_usd,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


@dataclass
class SuperAgentResult:
    status: str
    goal: str
    plan: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    final_text: str = ""
    cost_usd: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    turns: list[AgentResult] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal,
            "plan": self.plan,
            "completed_steps": self.completed_steps,
            "verification": self.verification,
            "final_text": self.final_text,
            "cost_usd": round(self.cost_usd, 6),
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "error": self.error,
        }


class SuperAgent:
    """长任务执行器：阶段状态机 + 检查点 + 验收。"""

    def __init__(
        self,
        provider: Provider,
        *,
        profile: SuperAgentProfile | None = None,
        tracer: Tracer | None = None,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        allow_side_effects: bool = False,
    ) -> None:
        self.provider = provider
        self.profile = profile or SuperAgentProfile()
        self.tracer = tracer
        self.model = model
        self.effort = effort
        self.allow_side_effects = allow_side_effects
        self._budget = Budget(
            max_steps=self.profile.max_steps,
            max_tool_calls=self.profile.max_tool_calls,
            max_cost_usd=self.profile.max_cost_usd,
        )

    # -- 检查点 -------------------------------------------------------------

    def _load_checkpoint(self, goal: str) -> dict[str, Any]:
        path = self.profile.checkpoint_path
        if not path or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if data.get("goal") != goal:
            return {}
        return data

    def _save_checkpoint(self, state: dict[str, Any]) -> None:
        path = self.profile.checkpoint_path
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)  # 原子替换，避免半截文件

    # -- 上下文治理 ---------------------------------------------------------

    def compact_history(
        self, messages: list[dict[str, Any]], *, keep_last: int | None = None
    ) -> list[dict[str, Any]]:
        """裁剪历史：保留首条用户目标与最近若干条消息。

        注意这是**保底式**裁剪，不是语义摘要 —— 它保证不炸上下文，但会丢细节。
        生产上更好的两个选择：
        1. 服务端压缩（Compaction beta，API 侧自动摘要历史并回传 compaction 块）；
        2. 让模型自己写摘要再替换历史（多一次调用，质量更好）。
        笔记 05 章「上下文治理」一节有取舍表。
        """
        keep_last = keep_last if keep_last is not None else self.profile.compact_keep_last
        if len(messages) <= keep_last + 1:
            return messages
        head = messages[:1]
        tail = messages[-keep_last:]
        dropped = len(messages) - len(head) - len(tail)
        marker = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"[上下文压缩] 已裁剪 {dropped} 条历史消息，关键结论请写在 output 字段里。",
                }
            ],
        }
        return head + [marker] + tail

    def _history_size(self, messages: Sequence[dict[str, Any]]) -> int:
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)

    # -- 阶段运行 -----------------------------------------------------------

    def _runtime(self, tools: list[str], system_suffix: str = "") -> AgentRuntime:
        registry: ToolRegistry = build_registry(tools)
        system = SUPER_AGENT_SYSTEM_PROMPT
        if system_suffix:
            system = f"{system}\n\n{system_suffix}"
        return AgentRuntime(
            provider=self.provider,
            tools=registry,
            system_prompt=system,
            model=self.model,
            effort=self.effort,
            budget=self._budget,
            tracer=self.tracer,
            allow_side_effects=self.allow_side_effects,
        )

    def _run_phase(
        self,
        runtime: AgentRuntime,
        prompt: str,
        history: list[dict[str, Any]],
    ) -> AgentResult:
        if self._history_size(history) > self.profile.compact_threshold:
            history = self.compact_history(history)
            if self.tracer:
                self.tracer.event("context_compacted", threshold=self.profile.compact_threshold)
        return runtime.run(prompt, history=history)

    # -- 主流程 -------------------------------------------------------------

    def run(
        self,
        goal: str,
        acceptance_criteria: list[str],
        *,
        resume: bool = False,
    ) -> SuperAgentResult:
        state = self._load_checkpoint(goal) if resume else {}
        plan: list[str] = list(state.get("plan", []))
        completed: list[str] = list(state.get("completed_steps", []))
        history: list[dict[str, Any]] = []
        turns: list[AgentResult] = []
        total_cost = 0.0
        total_steps = 0
        total_tools = 0
        verification: dict[str, Any] = dict(state.get("verification", {}))

        if self.tracer:
            self.tracer.event("superagent_start", goal=goal, resumed=bool(state))

        # ---- 阶段一：分解 -------------------------------------------------
        if not plan:
            plan_runtime = self._runtime(
                ["plan_tasks", "search_knowledge", "read_knowledge_doc", "list_orders", "lookup_order"]
            )
            plan_prompt = (
                f"目标：{goal}\n"
                f"验收标准：\n" + "\n".join(f"- {c}" for c in acceptance_criteria) + "\n\n"
                "请先制定子任务计划。"
            )
            plan_result = self._run_phase(plan_runtime, plan_prompt, history)
            turns.append(plan_result)
            total_cost += plan_result.cost_usd
            total_steps += plan_result.steps
            total_tools += len(plan_result.tool_calls)
            history = []
            for call in plan_result.tool_calls:
                if call.name == "plan_tasks" and call.status == "ok":
                    plan = list(call.input.get("steps") or [])
            if not plan:
                self._save_checkpoint(
                    {"goal": goal, "plan": [], "completed_steps": completed, "verification": verification}
                )
                return SuperAgentResult(
                    status="failed",
                    goal=goal,
                    plan=[],
                    completed_steps=completed,
                    cost_usd=total_cost,
                    steps=total_steps,
                    tool_calls=total_tools,
                    turns=turns,
                    error="规划阶段没有产出子任务清单",
                )

        # ---- 阶段二：逐条执行（含并行提示） --------------------------------
        for step in plan:
            if step in completed:
                continue
            exec_runtime = self._runtime(
                ["execute_task", "save_checkpoint", "calculate", "search_knowledge",
                 "read_knowledge_doc", "lookup_order", "list_orders"]
            )
            step_prompt = (
                f"目标：{goal}\n"
                f"当前子任务：{step}\n"
                f"验收标准：\n" + "\n".join(f"- {c}" for c in acceptance_criteria) + "\n\n"
                "请完成这个子任务，用 execute_task 回填产出，用 save_checkpoint 保存进度。"
            )
            step_result = self._run_phase(exec_runtime, step_prompt, history)
            turns.append(step_result)
            total_cost += step_result.cost_usd
            total_steps += step_result.steps
            total_tools += len(step_result.tool_calls)
            if step_result.status == STATUS_NEEDS_HUMAN:
                self._save_checkpoint(
                    {"goal": goal, "plan": plan, "completed_steps": completed, "verification": verification}
                )
                return SuperAgentResult(
                    status=STATUS_NEEDS_HUMAN,
                    goal=goal,
                    plan=plan,
                    completed_steps=completed,
                    verification=verification,
                    cost_usd=total_cost,
                    steps=total_steps,
                    tool_calls=total_tools,
                    turns=turns,
                    error="执行阶段请求人工接管",
                )
            if step_result.status != STATUS_COMPLETED:
                self._save_checkpoint(
                    {"goal": goal, "plan": plan, "completed_steps": completed, "verification": verification}
                )
                return SuperAgentResult(
                    status=step_result.status,
                    goal=goal,
                    plan=plan,
                    completed_steps=completed,
                    verification=verification,
                    cost_usd=total_cost,
                    steps=total_steps,
                    tool_calls=total_tools,
                    turns=turns,
                    error=step_result.error,
                )
            completed.append(step)

        # ---- 阶段三：验收 -------------------------------------------------
        verify_runtime = self._runtime(["check_acceptance", "execute_task", "save_checkpoint"])
        verify_prompt = (
            f"目标：{goal}\n"
            f"已完成子任务：\n" + "\n".join(f"- {s}" for s in completed) + "\n"
            f"验收标准：\n" + "\n".join(f"- {c}" for c in acceptance_criteria) + "\n\n"
            "请用 check_acceptance 逐条判定验收标准是否达成。"
        )
        verify_result = self._run_phase(verify_runtime, verify_prompt, history)
        turns.append(verify_result)
        total_cost += verify_result.cost_usd
        total_steps += verify_result.steps
        total_tools += len(verify_result.tool_calls)
        for call in verify_result.tool_calls:
            if call.name == "check_acceptance" and call.status == "ok":
                verification = {
                    "criteria": call.input.get("criteria", []),
                    "results": call.input.get("results", []),
                }

        passed = bool(verification.get("results")) and all(verification["results"])
        status = STATUS_COMPLETED if passed else "verification_failed"
        self._save_checkpoint(
            {
                "goal": goal,
                "plan": plan,
                "completed_steps": completed,
                "verification": verification,
                "cost_usd": total_cost,
            }
        )
        if self.tracer:
            self.tracer.event("superagent_end", status=status, cost_usd=total_cost)
        return SuperAgentResult(
            status=status,
            goal=goal,
            plan=plan,
            completed_steps=completed,
            verification=verification,
            final_text=verify_result.final_text,
            cost_usd=total_cost,
            steps=total_steps,
            tool_calls=total_tools,
            turns=turns,
            error=None if passed else "验收标准未全部通过",
        )


def build_super_agent(
    provider: Provider,
    *,
    profile: SuperAgentProfile | None = None,
    tracer: Tracer | None = None,
    model: str = "claude-opus-5-5",
    effort: str = "high",
    allow_side_effects: bool = False,
) -> SuperAgent:
    return SuperAgent(
        provider,
        profile=profile,
        tracer=tracer,
        model=model,
        effort=effort,
        allow_side_effects=allow_side_effects,
    )
