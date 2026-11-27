"""Provider 无关的数据类型。

刻意不直接用 anthropic SDK 的类型：离线替身（ScriptedProvider）要和真实 provider
产出同构的结果，否则离线跑通的测试在真实模式下不成立。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ThinkingBlock:
    thinking: str
    #: 回放时必须原样带回，否则模型读不到自己的推理链
    signature: str = ""
    type: Literal["thinking"] = "thinking"


@dataclass
class RedactedThinkingBlock:
    data: str
    type: Literal["redacted_thinking"] = "redacted_thinking"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass
class LLMResponse:
    """一次模型调用的归一化结果。"""

    blocks: list[TextBlock | ThinkingBlock | ToolUseBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    #: 结构化输出（output_schema 模式）解析后的对象；未请求结构化输出时为 None
    parsed: Any | None = None
    #: 供应商原始响应，排查用
    raw: Any | None = None

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, TextBlock)).strip()

    @property
    def thinking(self) -> str:
        return "\n".join(
            b.thinking for b in self.blocks if isinstance(b, ThinkingBlock)
        ).strip()

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.blocks if isinstance(b, ToolUseBlock)]

    def to_message(self) -> dict[str, Any]:
        """转回 provider 无关的 assistant 消息，供下一轮拼接。

        thinking 块必须原样回传（含 signature），否则模型读不到自己的推理链；
        这是 2026 年 preserved thinking 的硬要求。
        """
        content: list[dict[str, Any]] = []
        for b in self.blocks:
            if isinstance(b, TextBlock):
                content.append({"type": "text", "text": b.text})
            elif isinstance(b, ThinkingBlock):
                content.append(
                    {
                        "type": "thinking",
                        "thinking": b.thinking,
                        "signature": b.signature,
                    }
                )
            elif isinstance(b, RedactedThinkingBlock):
                content.append({"type": "redacted_thinking", "data": b.data})
            else:
                content.append(
                    {
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.input,
                    }
                )
        return {"role": "assistant", "content": content}
