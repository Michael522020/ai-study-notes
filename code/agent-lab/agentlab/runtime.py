"""Agent 运行时：手写的 agentic loop。

为什么要手写而不用 SDK 的 `tool_runner`：
托管版（`client.beta.messages.tool_runner`）确实更省事，但它把「模型看到什么、
工具返回什么、为什么停」这三件事藏起来了。本项目的目的正是把这三件事摊开讲清楚，
所以循环自己写；SDK 版作为对照写在 05 章的表格里。

循环的四要素（缺一即玩具）：
1. 工具 —— 可调用、输出确定；
2. 状态 —— 跨步持久化，可回放可审计；
3. 决策循环 —— 下一步是「调工具」还是「给答案」；
4. 停止条件与边界 —— 最大步数、预算闸门、人工闸门、危险动作白名单。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from agentkit.config import Budget, BudgetExceeded, Settings
from agentkit.costs import estimate_cost
from agentkit.provider import Provider, ProviderError
from agentkit.schema import validate
from agentkit.tracing import MemoryTracer, Tracer
from agentkit.types import LLMResponse, Usage

from .tools import ToolContext, ToolRegistry, ToolResult

# 运行结束状态。区分「正常完成」和「被闸门拦下」——运营看板要分开统计。
STATUS_COMPLETED = "completed"
STATUS_NEEDS_HUMAN = "needs_human"
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
STATUS_MAX_STEPS = "max_steps"
STATUS_FAILED = "failed"


@dataclass
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    status: str
    message: str = ""
    duration_ms: float = 0.0
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "status": self.status,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
            "is_error": self.is_error,
        }


@dataclass
class AgentResult:
    status: str
    final_text: str = ""
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    error: str | None = None
    #: 人工闸门拦下时，待人工确认的动作
    pending_action: dict[str, Any] | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETED

    @property
    def tool_error_count(self) -> int:
        return sum(1 for c in self.tool_calls if c.is_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_text": self.final_text,
            "steps": self.steps,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_error_count": self.tool_error_count,
            "cost_usd": round(self.cost_usd, 6),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
            },
            "stop_reason": self.stop_reason,
            "error": self.error,
            "pending_action": self.pending_action,
            "duration_ms": round(self.duration_ms, 2),
        }


class AgentRuntime:
    """通用 agentic loop。三种形态（Bot / Agent / Super Agent）共用它，差异在配置与系统提示。"""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        *,
        system_prompt: str,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        budget: Budget | None = None,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
        max_tokens: int = 16000,
        allow_side_effects: bool = False,
        #: 同一个工具入参连续失败几次后不再重试，防止「无效调用循环」
        max_identical_tool_errors: int = 2,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.model = model
        self.effort = effort
        self.budget = budget or Budget()
        self.tracer = tracer or MemoryTracer()
        self.settings = settings or Settings.from_env()
        self.max_tokens = max_tokens
        self.allow_side_effects = allow_side_effects
        self.max_identical_tool_errors = max_identical_tool_errors

    # -- 内部 ---------------------------------------------------------------

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            allow_side_effects=self.allow_side_effects,
            dry_run=not self.allow_side_effects,
            tracer=self.tracer,
        )

    def _dispatch(self, call: Any, ctx: ToolContext) -> ToolResult:
        spec = self.tools.get(call.name)
        if spec is None:
            return ToolResult(
                status="error",
                message=(
                    f"没有名为 {call.name} 的工具。可用工具：{self.tools.names()}。"
                    "请使用列表中的工具名。"
                ),
                error_code="unknown_tool",
            )
        errors = validate(call.input, spec.input_schema)
        if errors:
            return ToolResult(
                status="error",
                message="工具入参校验失败，请按 schema 修正后重试：" + "；".join(errors),
                error_code="invalid_input",
            )
        try:
            result = spec.handler(dict(call.input), ctx)
        except Exception as exc:  # 工具内部异常也要变成可自修的错误，不能炸掉整个 Agent
            return ToolResult(
                status="error",
                message=f"工具执行异常：{exc}",
                error_code="tool_exception",
            )
        if result.status not in ("ok", "error", "blocked", "needs_human"):
            return ToolResult(
                status="error",
                message=f"工具返回了未知状态 {result.status}",
                error_code="bad_status",
            )
        return result

    # -- 主循环 -------------------------------------------------------------

    def run(
        self,
        user_input: str,
        *,
        history: Sequence[dict[str, Any]] | None = None,
        extra_system: str | None = None,
    ) -> AgentResult:
        started = time.time()
        system = self.system_prompt
        if extra_system:
            system = f"{system}\n\n{extra_system}"

        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

        tool_calls: list[ToolCallRecord] = []
        total_usage = Usage()
        total_cost = 0.0
        error_counts: dict[str, int] = {}
        final_text = ""
        stop_reason = ""
        #: 本次运行的步数（与共享的 Budget 累计步数区分开）
        steps_this_run = 0

        self.tracer.event("run_start", model=self.model, tools=self.tools.names())

        try:
            while True:
                self.budget.charge_step()
                steps_this_run += 1
                resp: LLMResponse = self.provider.generate(
                    model=self.model,
                    messages=messages,
                    system=system,
                    tools=self.tools.provider_specs(),
                    max_tokens=self.max_tokens,
                    effort=self.effort,
                )
                total_usage = total_usage.merge(resp.usage)
                cost = estimate_cost(resp.model or self.model, resp.usage)
                total_cost += cost
                self.budget.charge_cost(cost)

                self.tracer.event(
                    "model_turn",
                    step=self.budget.steps,
                    stop_reason=resp.stop_reason,
                    tool_calls=[c.name for c in resp.tool_calls],
                    usage={
                        "input": resp.usage.input_tokens,
                        "output": resp.usage.output_tokens,
                        "cache_read": resp.usage.cache_read_input_tokens,
                    },
                    cost_usd=round(cost, 6),
                )

                messages.append(resp.to_message())

                if not resp.tool_calls:
                    final_text = resp.text
                    stop_reason = resp.stop_reason
                    break

                ctx = self._tool_context()
                tool_result_blocks: list[dict[str, Any]] = []
                needs_human_action: dict[str, Any] | None = None

                for call in resp.tool_calls:
                    self.budget.charge_tool_call()
                    t0 = time.time()
                    result = self._dispatch(call, ctx)
                    duration_ms = (time.time() - t0) * 1000
                    record = ToolCallRecord(
                        name=call.name,
                        input=dict(call.input),
                        status=result.status,
                        message=result.message,
                        duration_ms=duration_ms,
                        is_error=result.status == "error",
                    )
                    tool_calls.append(record)
                    self.tracer.event("tool_call", **record.to_dict())

                    if result.status == "error":
                        key = f"{call.name}:{json.dumps(call.input, sort_keys=True, ensure_ascii=False)}"
                        error_counts[key] = error_counts.get(key, 0) + 1
                        if error_counts[key] > self.max_identical_tool_errors:
                            needs_human_action = needs_human_action or {
                                "tool": call.name,
                                "input": dict(call.input),
                                "reason": f"同一工具入参连续失败 {error_counts[key]} 次",
                            }
                    elif result.status == "needs_human":
                        needs_human_action = {
                            "tool": call.name,
                            "input": dict(call.input),
                            "reason": result.message,
                            "preview": result.data,
                        }

                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": result.to_text(),
                            "is_error": result.status == "error",
                        }
                    )

                # 一次 assistant 回合里的多个 tool_use，必须在**同一条** user 消息里回结果
                messages.append({"role": "user", "content": tool_result_blocks})

                if needs_human_action is not None:
                    self.tracer.event("human_gate", **needs_human_action)
                    return AgentResult(
                        status=STATUS_NEEDS_HUMAN,
                        final_text=final_text,
                        steps=steps_this_run,
                        tool_calls=tool_calls,
                        cost_usd=total_cost,
                        usage=total_usage,
                        stop_reason="human_gate",
                        pending_action=needs_human_action,
                        duration_ms=(time.time() - started) * 1000,
                    )

        except BudgetExceeded as exc:
            self.tracer.event("budget_exceeded", reason=str(exc))
            return AgentResult(
                status=STATUS_BUDGET_EXCEEDED,
                final_text=final_text,
                steps=steps_this_run,
                tool_calls=tool_calls,
                cost_usd=total_cost,
                usage=total_usage,
                stop_reason="budget",
                error=str(exc),
                duration_ms=(time.time() - started) * 1000,
            )
        except ProviderError as exc:
            self.tracer.event("provider_error", reason=str(exc))
            return AgentResult(
                status=STATUS_FAILED,
                final_text=final_text,
                steps=steps_this_run,
                tool_calls=tool_calls,
                cost_usd=total_cost,
                usage=total_usage,
                stop_reason="provider_error",
                error=str(exc),
                duration_ms=(time.time() - started) * 1000,
            )

        self.tracer.event("run_end", status=STATUS_COMPLETED, steps=self.budget.steps)
        return AgentResult(
            status=STATUS_COMPLETED,
            final_text=final_text,
            steps=steps_this_run,
            tool_calls=tool_calls,
            cost_usd=total_cost,
            usage=total_usage,
            stop_reason=stop_reason,
            duration_ms=(time.time() - started) * 1000,
        )
