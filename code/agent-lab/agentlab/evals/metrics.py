"""评测指标与断言。

三层指标，分开看：
1. **行为指标** —— 状态对不对、工具调对没有、答案含不含关键信息（离线可测）
2. **效率指标** —— 步数、工具调用次数、每次工具错误、每任务成本（离线可测）
3. **质量指标** —— 答案本身好不好（离线测不了，必须真实 API + 人工/LLM 评审）

第三层不在离线评测里，这是刻意的：用一个写死的脚本去断言「模型答得好不好」是自欺欺人。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class CaseResult:
    name: str
    profile: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    status: str = ""
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    called_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile,
            "passed": self.passed,
            "reasons": self.reasons,
            "status": self.status,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "cost_usd": round(self.cost_usd, 6),
            "duration_ms": round(self.duration_ms, 2),
            "called_tools": self.called_tools,
        }


def evaluate_case(case: dict[str, Any], result: Any) -> CaseResult:
    """按 expect 字典断言一个用例。返回逐条失败原因，不抛异常。"""
    expect = case.get("expect", {}) or {}
    reasons: list[str] = []
    called = [c.name for c in result.tool_calls]

    cr = CaseResult(
        name=case["name"],
        profile=case["profile"],
        passed=True,
        status=result.status,
        steps=result.steps,
        tool_calls=len(result.tool_calls),
        tool_errors=result.tool_error_count,
        cost_usd=result.cost_usd,
        duration_ms=result.duration_ms,
        called_tools=called,
    )

    if "status" in expect and result.status != expect["status"]:
        reasons.append(f"状态期望 {expect['status']}，实际 {result.status}")

    for tool in expect.get("tools_called", []):
        if tool not in called:
            reasons.append(f"期望调用工具 {tool}，实际调用 {called}")

    if "forbidden_tools" in expect:
        for tool in expect["forbidden_tools"]:
            if tool in called:
                reasons.append(f"不该调用工具 {tool}")

    for frag in expect.get("final_text_contains", []):
        if frag not in result.final_text:
            reasons.append(f"最终回答缺少片段 {frag!r}")

    if "pending_action_contains" in expect:
        pending = result.pending_action or {}
        blob = str(pending.get("reason", "")) + str(pending.get("preview", ""))
        if expect["pending_action_contains"] not in blob:
            reasons.append(
                f"待人工动作缺少片段 {expect['pending_action_contains']!r}（实际 {blob!r}）"
            )

    if "min_tool_errors" in expect and result.tool_error_count < expect["min_tool_errors"]:
        reasons.append(
            f"期望至少 {expect['min_tool_errors']} 次工具错误，实际 {result.tool_error_count}"
        )

    if "max_steps" in expect and result.steps > expect["max_steps"]:
        reasons.append(f"步数 {result.steps} 超过上限 {expect['max_steps']}")

    if "max_cost_usd" in expect and result.cost_usd > expect["max_cost_usd"]:
        reasons.append(f"成本 ${result.cost_usd:.6f} 超过上限 ${expect['max_cost_usd']}")

    cr.passed = not reasons
    cr.reasons = reasons
    return cr


def aggregate(results: Sequence[CaseResult]) -> dict[str, Any]:
    """汇总成一张报表。"""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    by_profile: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_profile.setdefault(
            r.profile, {"cases": 0, "passed": 0, "steps": 0, "cost_usd": 0.0, "tool_errors": 0}
        )
        bucket["cases"] += 1
        bucket["passed"] += 1 if r.passed else 0
        bucket["steps"] += r.steps
        bucket["cost_usd"] += r.cost_usd
        bucket["tool_errors"] += r.tool_errors
    for bucket in by_profile.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["cases"], 4) if bucket["cases"] else 0.0
        bucket["avg_steps"] = round(bucket["steps"] / bucket["cases"], 2) if bucket["cases"] else 0.0
        bucket["avg_cost_usd"] = round(bucket["cost_usd"] / bucket["cases"], 6) if bucket["cases"] else 0.0
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)

    costs = sorted(r.cost_usd for r in results)
    p95 = costs[min(len(costs) - 1, int(len(costs) * 0.95))] if costs else 0.0
    return {
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "avg_steps": round(sum(r.steps for r in results) / total, 2) if total else 0.0,
        "avg_tool_errors": round(sum(r.tool_errors for r in results) / total, 2) if total else 0.0,
        "p95_cost_usd": round(p95, 6),
        "by_profile": by_profile,
    }


def render_table(results: Sequence[CaseResult]) -> str:
    """终端可读的表格。"""
    header = f"{'用例':<34}{'形态':<11}{'结果':<6}{'状态':<20}{'步':>3}{'工具':>5}{'错误':>5}{'成本$':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        status = r.status
        if not r.passed and r.reasons:
            status = f"{status} ({r.reasons[0][:28]})"
        lines.append(
            f"{r.name[:32]:<34}{r.profile:<11}{mark:<6}{status:<20}"
            f"{r.steps:>3}{r.tool_calls:>5}{r.tool_errors:>5}{r.cost_usd:>10.6f}"
        )
    return "\n".join(lines)
