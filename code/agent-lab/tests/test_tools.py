"""工具层单元测试。"""

from __future__ import annotations

import pytest

from agentlab.tools import ToolContext, build_registry, safe_eval


def test_safe_eval_arithmetic():
    assert safe_eval("18400 * 0.03") == pytest.approx(552.0)
    assert safe_eval("(100 + 20) / 4") == pytest.approx(30.0)
    assert safe_eval("2 ** 10") == pytest.approx(1024.0)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "1 if True else 2",
        "2 ** 100000",
        "999999999999 * 999999999999",
    ],
)
def test_safe_eval_rejects_unsafe(expr):
    with pytest.raises(Exception):
        safe_eval(expr)


def test_search_knowledge_hits():
    registry = build_registry(["search_knowledge"])
    result = registry.get("search_knowledge").handler(
        {"query": "延迟,补偿", "top_k": 2}, ToolContext()
    )
    assert result.status == "ok"
    assert result.data, "应至少命中一篇文档"
    assert any("补偿" in d["content"] for d in result.data)


def test_lookup_order_not_found_is_error():
    registry = build_registry(["lookup_order"])
    result = registry.get("lookup_order").handler({"order_id": "ZZZ"}, ToolContext())
    assert result.status == "error"
    assert result.error_code == "not_found"


def test_create_ticket_is_dry_run_by_default():
    registry = build_registry(["create_ticket"])
    result = registry.get("create_ticket").handler(
        {"order_id": "B456", "issue": "延迟未出库"}, ToolContext(allow_side_effects=False)
    )
    assert result.status == "needs_human"
    assert result.data["would_create"]["order_id"] == "B456"


def test_create_ticket_writes_when_allowed():
    from agentlab.tools import TICKETS

    before = len(TICKETS)
    registry = build_registry(["create_ticket"])
    result = registry.get("create_ticket").handler(
        {"order_id": "B456", "issue": "延迟未出库"}, ToolContext(allow_side_effects=True)
    )
    assert result.status == "ok"
    assert len(TICKETS) == before + 1


def test_registry_unknown_tool():
    registry = build_registry(["lookup_order"])
    assert registry.get("does_not_exist") is None
    assert "lookup_order" in registry.names()
