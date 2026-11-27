"""Provider 抽象：真实 Claude API 与离线确定性替身。

为什么要两个实现：
- `AnthropicProvider` 走真实 API，用于效果评测与生产；
- `ScriptedProvider` 按脚本返回动作，用于**离线跑通编排逻辑、回归测试、CI**。

关键约定：离线替身**只替代模型**，工具是真的在跑、账是真的在算、轨迹是真的在写。
所以离线能证明「编排是对的」，**不能证明「模型能力够」** —— 这条边界在笔记里反复强调，
评测报告里也分列两个指标组。

另有一条硬规则：真实 provider 永远不会在缺 key 时静默降级。
`Settings.require_api_key()` 会带着修复命令抛 `ConfigError`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from .config import Settings
from .schema import strict_schema
from .types import (
    LLMResponse,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)


class ProviderError(RuntimeError):
    """provider 调用失败。属于可预期的错误路径，调用方应转成业务语义。"""


class ScriptExhausted(ProviderError):
    """离线脚本用完了 —— 通常意味着用例设计与运行时行为不一致，是 bug 而不是业务结果。"""


class Provider(Protocol):
    """模型供应商的统一接口。"""

    name: str

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
        effort: str = "high",
    ) -> LLMResponse: ...


def _thinking_param(display: str | None) -> dict[str, Any]:
    """自适应思考。2026 年起固定 thinking 预算（budget_tokens）已废弃。"""
    param: dict[str, Any] = {"type": "adaptive"}
    if display:
        param["display"] = display
    return param


class AnthropicProvider:
    """真实 Claude API。

    刻意手写 agentic loop 所需的最小请求面，而不是直接用 SDK 的 tool_runner：
    循环本身是这份笔记要讲清楚的东西，藏在 SDK 里就讲不清楚了。
    SDK 的 `client.beta.messages.tool_runner` 作为托管版替代方案在笔记中做对比。
    """

    name = "anthropic"

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        *,
        thinking_display: str | None = None,
    ) -> None:
        self._settings = settings or Settings.from_env()
        self._thinking_display = thinking_display
        if client is None:
            import anthropic  # 延迟导入：离线模式不需要这个依赖

            client = anthropic.Anthropic(api_key=self._settings.require_api_key())
        self._client = client

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
        effort: str = "high",
    ) -> LLMResponse:
        output_config: dict[str, Any] = {"effort": effort}
        thinking = _thinking_param(self._thinking_display)

        try:
            if output_schema is not None:
                # 结构化输出：模型直接返回符合 schema 的对象，不需要正则抠 JSON
                output_config["format"] = {
                    "type": "json_schema",
                    "schema": strict_schema(output_schema),
                }
                msg = self._client.messages.parse(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=list(messages),
                    output_config=output_config,
                    thinking=thinking,
                )
                parsed = None
                for block in msg.content:
                    if getattr(block, "type", None) == "text":
                        parsed = getattr(block, "parsed_output", None)
                        if parsed is not None:
                            break
                response = LLMResponse(
                    blocks=self._convert_blocks(msg.content),
                    stop_reason=msg.stop_reason or "end_turn",
                    usage=self._convert_usage(msg.usage),
                    model=msg.model,
                    parsed=parsed,
                    raw=msg,
                )
                if parsed is None:
                    raise ProviderError(
                        "请求了结构化输出但没有拿到 parsed_output，"
                        "通常是 schema 不合法或模型拒答，请检查 stop_reason。"
                    )
                return response

            request: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": list(messages),
                "output_config": output_config,
                "thinking": thinking,
            }
            if tools:
                # tool_choice 用 auto：2026 年起在 Opus 5.5 / Sonnet 5.5 / Fable 5.2 上
                # 强制 tool_choice=any|tool 会直接 400，改用「严格 schema + auto」。
                request["tools"] = [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "input_schema": strict_schema(t["input_schema"]),
                        "strict": True,
                    }
                    for t in tools
                ]
                request["tool_choice"] = {"type": "auto"}

            msg = self._client.messages.create(**request)
            return LLMResponse(
                blocks=self._convert_blocks(msg.content),
                stop_reason=msg.stop_reason or "end_turn",
                usage=self._convert_usage(msg.usage),
                model=msg.model,
                raw=msg,
            )
        except ProviderError:
            raise
        except Exception as exc:  # 统一成 ProviderError，保留原异常链
            raise ProviderError(f"模型调用失败：{exc}") from exc

    @staticmethod
    def _convert_blocks(content: Iterable[Any]) -> list[Any]:
        blocks: list[Any] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                blocks.append(TextBlock(block.text))
            elif btype == "thinking":
                blocks.append(
                    ThinkingBlock(
                        thinking=getattr(block, "thinking", ""),
                        signature=getattr(block, "signature", ""),
                    )
                )
            elif btype == "redacted_thinking":
                blocks.append(RedactedThinkingBlock(data=getattr(block, "data", "")))
            elif btype == "tool_use":
                blocks.append(
                    ToolUseBlock(id=block.id, name=block.name, input=dict(block.input))
                )
        return blocks

    @staticmethod
    def _convert_usage(usage: Any) -> Usage:
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )


@dataclass
class ScriptedStep:
    """离线脚本的一步。

    - text: 直接给最终回答
    - tool: 发起一次工具调用
    - structured: 返回结构化对象
    - thinking: 只回一段思考（用于验证 thinking 块回放链路）
    """

    text: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    structured: Any | None = None
    thinking: str | None = None


class ScriptedProvider:
    """离线确定性替身。按脚本逐步返回动作。

    它**不是**模型的替身，只是「模型这一步会做什么」的固定回放：
    工具仍然真实执行，成本仍然按估算记账，轨迹仍然完整写盘。
    """

    name = "scripted"

    def __init__(
        self,
        script: Sequence[dict[str, Any] | ScriptedStep],
        *,
        model: str = "claude-opus-5-5",
    ) -> None:
        self._script = list(script)
        self._index = 0
        self.model = model
        #: 记录每次请求，测试里断言「模型看到了什么」
        self.calls: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return max(0, len(self._script) - self._index)

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
        effort: str = "high",
    ) -> LLMResponse:
        self.calls.append(
            {
                "model": model,
                "messages": [dict(m) for m in messages],
                "system": system,
                "tools": [t["name"] for t in tools] if tools else [],
                "output_schema": output_schema,
                "effort": effort,
            }
        )
        if self._index >= len(self._script):
            raise ScriptExhausted(
                f"离线脚本已用尽（第 {self._index + 1} 次调用）。"
                "要么是运行时多走了一步，要么是脚本少写了一步。"
            )
        step = self._script[self._index]
        self._index += 1
        if isinstance(step, ScriptedStep):
            step = {
                k: v
                for k, v in {
                    "text": step.text,
                    "tool": (
                        {"name": step.tool_name, "input": step.tool_input}
                        if step.tool_name
                        else None
                    ),
                    "structured": step.structured,
                    "thinking": step.thinking,
                }.items()
                if v is not None
            }
        return self._render(step)

    def _render(self, step: dict[str, Any]) -> LLMResponse:
        blocks: list[Any] = []
        if step.get("thinking"):
            blocks.append(ThinkingBlock(thinking=step["thinking"], signature="offline"))
        tool = step.get("tool")
        if tool:
            blocks.append(
                ToolUseBlock(
                    id=f"toolu_offline_{self._index}",
                    name=tool["name"],
                    input=dict(tool.get("input") or {}),
                )
            )
        if step.get("structured") is not None:
            blocks.append(TextBlock(json.dumps(step["structured"], ensure_ascii=False)))
        if step.get("text"):
            blocks.append(TextBlock(step["text"]))
        if not blocks:
            raise ScriptExhausted(f"离线脚本第 {self._index} 步既没有动作也没有文本")
        return LLMResponse(
            blocks=blocks,
            stop_reason="tool_use" if tool else "end_turn",
            # 离线替身报告一个**名义** token 数：够跑通成本闸门的记账逻辑，
            # 又不至于把成本顶到预算红线（真实调用请用 AnthropicProvider）。
            usage=Usage(input_tokens=420, output_tokens=90),
            model=self.model,
            parsed=step.get("structured"),
        )
