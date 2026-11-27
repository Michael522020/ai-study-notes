"""agent-lab 命令行入口。

    python -m agentlab eval                # 跑离线评测集（不需要 API key）
    python -m agentlab profile             # 打印三种形态的参数对照
    python -m agentlab chat --profile bot  # 真实 API 交互（需要 ANTHROPIC_API_KEY）
"""

from __future__ import annotations

import argparse
import json
import sys

from .evals.run_eval import main as eval_main
from .profiles.agent import AgentProfile
from .profiles.bot import BotProfile
from .profiles.superagent import SuperAgentProfile


def _print_profiles() -> None:
    rows = [
        ("bot", BotProfile().to_dict()),
        ("agent", AgentProfile().to_dict()),
        ("superagent", SuperAgentProfile().to_dict()),
    ]
    keys = sorted({k for _, d in rows for k in d})
    width = max(len(k) for k in keys) + 2
    print(" " * 12 + "".join(k.ljust(width) for k in keys))
    for name, d in rows:
        cells = []
        for k in keys:
            v = d.get(k)
            cells.append(("" if v is None else str(v)).ljust(width))
        print(name.ljust(12) + "".join(cells))


def _chat(profile: str) -> int:
    from agentkit.config import Settings
    from agentkit.provider import AnthropicProvider

    settings = Settings.from_env()
    provider = AnthropicProvider(settings)
    if profile == "bot":
        from .profiles.bot import build_bot

        runtime = build_bot(provider)
    elif profile == "agent":
        from .profiles.agent import build_agent

        runtime = build_agent(provider)
    else:
        from .profiles.superagent import build_super_agent

        runtime = build_super_agent(provider)
        goal = input("目标：").strip()
        criteria = [c.strip() for c in input("验收标准（分号分隔）：").split("；") if c.strip()]
        result = runtime.run(goal, criteria)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"已进入 {profile} 模式，输入 quit 退出。")
    history: list[dict] = []
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ("quit", "exit", ":q"):
            break
        if not user:
            continue
        result = runtime.run(user, history=history)
        print(f"bot[{result.status}] {result.final_text}")
        print(
            f"    步数={result.steps} 工具={len(result.tool_calls)} "
            f"成本=${result.cost_usd:.6f} 延迟={result.duration_ms:.0f}ms"
        )
        if result.pending_action:
            print(f"    待人工确认：{result.pending_action}")
        history.append({"role": "user", "content": [{"type": "text", "text": user}]})
        history.append({"role": "assistant", "content": [{"type": "text", "text": result.final_text}]})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentlab", description="Agent 形态实验台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="跑离线评测集")
    p_eval.add_argument("--profile", choices=["bot", "agent", "superagent"])
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("--report")

    sub.add_parser("profile", help="打印三种形态参数对照")

    p_chat = sub.add_parser("chat", help="真实 API 交互")
    p_chat.add_argument("--profile", choices=["bot", "agent", "superagent"], default="bot")

    args = parser.parse_args(argv)
    if args.command == "eval":
        argv2 = ["--json", "--report", str(args.report)] if args.report else []
        if args.profile:
            argv2 += ["--profile", args.profile]
        if args.json:
            argv2.append("--json")
        return eval_main(argv2)
    if args.command == "profile":
        _print_profiles()
        return 0
    if args.command == "chat":
        return _chat(args.profile)
    return 1


if __name__ == "__main__":
    sys.exit(main())
