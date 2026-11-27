"""运行时单元测试：闸门、错误语义、回放完整性。"""

from __future__ import annotations

from agentkit.provider import ScriptedProvider
from agentkit.tracing import MemoryTracer
from agentkit.types import TextBlock, ThinkingBlock, ToolUseBlock

from agentlab.profiles.bot import build_bot
from agentlab.runtime import (
    STATUS_BUDGET_EXCEEDED,
    STATUS_COMPLETED,
    STATUS_NEEDS_HUMAN,
)


def _bot(script):
    provider = ScriptedProvider(script)
    tracer = MemoryTracer()
    return build_bot(provider, tracer=tracer), tracer


def test_unknown_tool_is_reported_to_model():
    runtime, _ = _bot(
        [
            {"tool": {"name": "nope_tool", "input": {}}},
            {"tool": {"name": "lookup_order", "input": {"order_id": "A123"}}},
            {"text": "订单 A123 状态 shipped。"},
        ]
    )
    result = runtime.run("查订单 A123")
    assert result.status == STATUS_COMPLETED
    assert result.tool_error_count == 1
    # 错误信息必须回喂给模型：轨迹里应能看到工具错误
    assert any(c.status == "error" for c in result.tool_calls)


def test_invalid_input_is_reported_to_model():
    runtime, _ = _bot(
        [
            {"tool": {"name": "lookup_order", "input": {}}},
            {"tool": {"name": "lookup_order", "input": {"order_id": "A123"}}},
            {"text": "已查到。"},
        ]
    )
    result = runtime.run("查订单")
    assert result.status == STATUS_COMPLETED
    assert result.tool_error_count >= 1


def test_budget_gate_stops_runaway_loop():
    runtime, _ = _bot([{"tool": {"name": "list_orders", "input": {}}}] * 10)
    result = runtime.run("查所有订单")
    assert result.status == STATUS_BUDGET_EXCEEDED
    assert result.steps <= 4


def test_human_gate_intercepts_side_effect():
    runtime, tracer = _bot(
        [
            {
                "tool": {
                    "name": "create_ticket",
                    "input": {"order_id": "B456", "issue": "延迟未出库"},
                }
            }
        ]
    )
    result = runtime.run("给 B456 建单")
    assert result.status == STATUS_NEEDS_HUMAN
    assert result.pending_action["tool"] == "create_ticket"
    assert tracer.find("human_gate")


def test_thinking_blocks_are_replayed_unchanged():
    """preserved thinking：thinking 块必须带 signature 原样回传。"""
    blocks = [ThinkingBlock(thinking="先查订单", signature="sig-123")]
    message = {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "y"}]}
    assert message["content"][0]["signature"] == "y"
    assert blocks[0].signature == "sig-123"


def test_tool_results_share_one_user_message():
    runtime, _ = _bot(
        [
            {
                "tool": {"name": "lookup_order", "input": {"order_id": "A123"}},
            },
            {"text": "查到了。"},
        ]
    )
    result = runtime.run("查订单")
    assert result.status == STATUS_COMPLETED
    assert [c.name for c in result.tool_calls] == ["lookup_order"]


def test_trace_records_turns():
    runtime, tracer = _bot([{"text": "你好"}])
    runtime.run("hi")
    assert "run_start" in tracer.kinds()
    assert "model_turn" in tracer.kinds()
    assert "run_end" in tracer.kinds()


def test_final_answer_ends_the_loop():
    """模型给出最终回答就该停 —— 不能因为脚本里还有内容就继续跑。"""
    provider = ScriptedProvider([{"text": "第一段"}, {"text": "第二段"}])
    runtime = build_bot(provider, tracer=MemoryTracer())
    result = runtime.run("说两句")
    assert result.status == STATUS_COMPLETED
    assert "第一段" in result.final_text
    assert "第二段" not in result.final_text
    assert provider.remaining == 1, "多余的一轮脚本不应被消耗"
