"""评测执行器。

用法：
    python -m agentlab.evals.run_eval              # 跑全部用例
    python -m agentlab.evals.run_eval --profile bot # 只跑某个形态
    python -m agentlab.evals.run_eval --json       # 输出 JSON 报表
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agentkit.provider import ScriptedProvider
from agentkit.tracing import MemoryTracer

from ..profiles.agent import build_agent
from ..profiles.bot import build_bot
from ..profiles.superagent import build_super_agent
from .cases import CASES
from .metrics import CaseResult, aggregate, evaluate_case, render_table


def _run_bot(case: dict[str, Any]) -> Any:
    provider = ScriptedProvider(case["script"])
    runtime = build_bot(provider, tracer=MemoryTracer())
    return runtime.run(case["user"])


def _run_agent(case: dict[str, Any]) -> Any:
    provider = ScriptedProvider(case["script"])
    runtime = build_agent(provider, tracer=MemoryTracer())
    return runtime.run(case["user"])


def _run_superagent(case: dict[str, Any]) -> Any:
    provider = ScriptedProvider(case["script"])
    super_agent = build_super_agent(provider, tracer=MemoryTracer())
    result = super_agent.run(case["user"], case.get("acceptance", []))
    # 统一成 AgentResult 的字段视图，方便用同一套断言
    return _SuperAgentView(result)


class _SuperAgentView:
    """把 SuperAgentResult 适配成 AgentResult 的字段视图。"""

    def __init__(self, result: Any) -> None:
        self._r = result
        self.status = result.status
        self.final_text = result.final_text
        self.steps = result.steps
        self.cost_usd = result.cost_usd
        self.duration_ms = 0.0
        self.error = result.error
        self.pending_action = None

    @property
    def tool_calls(self) -> list[Any]:
        calls: list[Any] = []
        for turn in self._r.turns:
            calls.extend(turn.tool_calls)
        return calls

    @property
    def tool_error_count(self) -> int:
        return sum(1 for c in self.tool_calls if c.is_error)


def run_case(case: dict[str, Any]) -> CaseResult:
    runner = {"bot": _run_bot, "agent": _run_agent, "superagent": _run_superagent}.get(
        case["profile"]
    )
    if runner is None:
        raise ValueError(f"未知 profile：{case['profile']}")
    result = runner(case)
    return evaluate_case(case, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="agent-lab 离线评测")
    parser.add_argument("--profile", choices=["bot", "agent", "superagent"], help="只跑某个形态")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报表")
    parser.add_argument("--report", type=Path, help="把 JSON 报表写到文件")
    args = parser.parse_args(argv)

    cases = [c for c in CASES if not args.profile or c["profile"] == args.profile]
    results = [run_case(c) for c in cases]
    summary = aggregate(results)

    if args.json:
        payload = {"summary": summary, "cases": [r.to_dict() for r in results]}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = render_table(results) + "\n\n" + json.dumps(summary, ensure_ascii=False, indent=2)

    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "cases": [r.to_dict() for r in results]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n报表已写入 {args.report}")

    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
