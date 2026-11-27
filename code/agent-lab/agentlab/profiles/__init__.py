"""三种 Agent 形态的 profile。

形态差异不在「会不会调工具」，而在四件事：
1. 工具面大小（给多了乱调，给少了绕路）；
2. 循环深度（Bot 3~4 步，Super Agent 几十步）；
3. 预算闸门（Bot 一轮成本必须封顶，否则并发上来账单失控）；
4. 停止与兜底（Bot 随时可转人工，Super Agent 要有验收标准兜底）。
"""

from .agent import AgentProfile, build_agent
from .bot import BotProfile, build_bot
from .superagent import SuperAgentProfile, build_super_agent

__all__ = [
    "AgentProfile",
    "BotProfile",
    "SuperAgentProfile",
    "build_agent",
    "build_bot",
    "build_super_agent",
]
