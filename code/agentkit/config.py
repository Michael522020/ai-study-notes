"""运行期配置与预算闸门。

所有配置来自环境变量；`Settings.from_env()` 在缺少 API key 时**不**降级为假 provider，
而是让调用方显式选择 `ScriptedProvider`（离线替身）。理由见 05-Agent与智能体/01 章
「失败语义要显式」一节：缺配置返回 not_configured，而不是悄悄走 fake。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 默认模型：Claude Opus 5.5（见 05 章「模型选型与成本」）
DEFAULT_MODEL = "claude-opus-5-5"
# 便宜档：Claude Haiku 5.5，用于分类 / 抽取 / 改写这类窄任务
CHEAP_MODEL = "claude-haiku-5-5"

# effort 档位：low < medium < high < xhigh < max
# 2026 年起 `budget_tokens` 已被 `output_config.effort` 取代（thinking 的固定预算参数在
# Opus 5.5 / Sonnet 5.5 / Fable 5.2 上直接 400）。
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class ConfigError(RuntimeError):
    """配置缺失或非法。调用方应显式捕获并向上返回，不要吞掉。"""


@dataclass(frozen=True)
class Settings:
    """一次运行所需的全部外部配置。"""

    model: str = DEFAULT_MODEL
    fast_model: str = CHEAP_MODEL
    effort: str = "high"
    max_tokens: int = 16000
    api_key: str | None = None
    trace_path: Path | None = None
    #: 离线替身模式下是否允许「写操作」类工具真执行。默认否 —— 只出草稿，不落库不外发。
    allow_side_effects: bool = False

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        trace = os.environ.get("AGENTKIT_TRACE")
        settings = cls(
            model=os.environ.get("AGENTKIT_MODEL", DEFAULT_MODEL),
            fast_model=os.environ.get("AGENTKIT_FAST_MODEL", CHEAP_MODEL),
            effort=os.environ.get("AGENTKIT_EFFORT", "high"),
            max_tokens=int(os.environ.get("AGENTKIT_MAX_TOKENS", "16000")),
            api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            trace_path=Path(trace) if trace else None,
            allow_side_effects=os.environ.get("AGENTKIT_ALLOW_SIDE_EFFECTS") == "1",
        )
        if settings.effort not in EFFORT_LEVELS:
            raise ConfigError(
                f"AGENTKIT_EFFORT={settings.effort!r} 非法，可选：{', '.join(EFFORT_LEVELS)}"
            )
        if overrides:
            settings = settings.replace(**overrides)  # type: ignore[arg-type]
        return settings

    def replace(self, **changes: object) -> "Settings":
        from dataclasses import replace as _replace

        return _replace(self, **changes)  # type: ignore[arg-type]

    def require_api_key(self) -> str:
        """真实模式取 key。缺失时抛 ConfigError，附带可直接执行的修复命令。"""
        if not self.api_key:
            raise ConfigError(
                "缺少 ANTHROPIC_API_KEY。\n"
                "  · 真实运行：export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  · 或先执行 `ant auth login`（SDK 会自动读取本地 profile，无需导出变量）\n"
                "  · 离线跑通编排逻辑：改用 ScriptedProvider，不要在生产代码里 mock。"
            )
        return self.api_key


@dataclass
class Budget:
    """预算闸门：步数、工具调用次数、美元成本三条线任一越界即停。

    停止时抛 BudgetExceeded，调用方负责把它翻译成业务语义
    （Bot 形态转成「已转人工」，Agent 形态转成 needs_human）。
    """

    max_steps: int = 12
    max_tool_calls: int = 24
    max_cost_usd: float = 0.50
    max_seconds: float = 300.0
    spent_cost_usd: float = 0.0
    spent_tool_calls: int = 0
    steps: int = 0

    def charge_step(self) -> None:
        # 先判后计：步数永远不会超过 max_steps，计数器不会「先越界再报错」
        if self.steps + 1 > self.max_steps:
            raise BudgetExceeded(f"步数超限：{self.steps + 1} > {self.max_steps}")
        self.steps += 1

    def charge_tool_call(self) -> None:
        self.spent_tool_calls += 1
        if self.spent_tool_calls > self.max_tool_calls:
            raise BudgetExceeded(
                f"工具调用次数超限：{self.spent_tool_calls} > {self.max_tool_calls}"
            )

    def charge_cost(self, usd: float) -> None:
        self.spent_cost_usd += usd
        if self.spent_cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"成本超限：${self.spent_cost_usd:.4f} > ${self.max_cost_usd:.4f}"
            )

    def snapshot(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "max_steps": self.max_steps,
            "tool_calls": self.spent_tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "cost_usd": round(self.spent_cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
        }


class BudgetExceeded(RuntimeError):
    """预算闸门触发。属于可预期的业务结果，不是崩溃。"""
