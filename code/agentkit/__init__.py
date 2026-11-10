"""共享内核：配置、类型、成本、校验、轨迹。

设计原则（与仓库既有笔记一脉相承）：
- 缺配置就报错退出，不静默走 fake provider；
- 成本、步数、工具调用次数都是一等公民，运行时强制记账；
- 数据与指令分离：外部正文进 prompt 前标注为「数据」。
"""

from .config import Budget, ConfigError, Settings
from .costs import PRICING, ModelPrice, estimate_cost
from .types import (
    LLMResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

__all__ = [
    "Budget",
    "ConfigError",
    "LLMResponse",
    "ModelPrice",
    "PRICING",
    "Settings",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "Usage",
    "estimate_cost",
]
