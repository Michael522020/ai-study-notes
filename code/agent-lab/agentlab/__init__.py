"""agent-lab：三种 Agent 形态的可运行实验台。

- `tools`    工具层（定义即契约）
- `runtime`  手写 agentic loop
- `profiles` Bot / Agent / Super Agent 三种形态
- `evals`    离线确定性评测集
"""

from .runtime import (
    STATUS_BUDGET_EXCEEDED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_MAX_STEPS,
    STATUS_NEEDS_HUMAN,
    AgentResult,
    AgentRuntime,
)
from .tools import ToolRegistry, ToolSpec, build_registry

__all__ = [
    "AgentResult",
    "AgentRuntime",
    "STATUS_BUDGET_EXCEEDED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_MAX_STEPS",
    "STATUS_NEEDS_HUMAN",
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
]
