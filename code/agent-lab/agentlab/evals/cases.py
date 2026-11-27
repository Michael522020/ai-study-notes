"""离线评测集：每个用例是一段「脚本」+ 一组「期望」。

重要边界：**离线脚本只替代模型，工具是真的在跑**。
所以这套评测能证明「编排逻辑、闸门、账目、错误语义是对的」，
**不能证明「模型能不能做对」** —— 后者必须用真实 API 跑同一套断言（见 README 的对照表）。

用例字段：
- profile: bot | agent | superagent
- user:    用户输入
- script:  ScriptedProvider 的步骤序列
- expect:  断言字典
"""

from __future__ import annotations

from typing import Any

CASES: list[dict[str, Any]] = [
    {
        "name": "bot-知识库政策问答",
        "profile": "bot",
        "user": "订单延迟超过 7 天要怎么补偿？",
        "script": [
            {"tool": {"name": "search_knowledge", "input": {"query": "延迟,补偿", "top_k": 2}}},
            {"text": "延迟超过 7 天需按订单金额 3% 补偿，并开补偿单回写到工单。"},
        ],
        "expect": {
            "status": "completed",
            "tools_called": ["search_knowledge"],
            "final_text_contains": ["3%"],
            "max_cost_usd": 0.02,
        },
    },
    {
        "name": "bot-订单状态查询",
        "profile": "bot",
        "user": "订单 A123 现在什么状态？",
        "script": [
            {"tool": {"name": "lookup_order", "input": {"order_id": "A123"}}},
            {"text": "订单 A123 状态为 shipped，已于 10 月 6 日出运，预计 11 月 8 日送达。"},
        ],
        "expect": {
            "status": "completed",
            "tools_called": ["lookup_order"],
            "final_text_contains": ["shipped"],
        },
    },
    {
        "name": "bot-金额必须走计算工具",
        "profile": "bot",
        "user": "订单 A123 如果按 3% 补偿，赔多少钱？",
        "script": [
            {"tool": {"name": "lookup_order", "input": {"order_id": "A123"}}},
            {"tool": {"name": "calculate", "input": {"expression": "18400 * 0.03"}}},
            {"text": "订单 A123 金额 18400 美元，按 3% 计算补偿为 552 美元。"},
        ],
        "expect": {
            "status": "completed",
            "tools_called": ["lookup_order", "calculate"],
            "final_text_contains": ["552"],
        },
    },
    {
        "name": "bot-越权承诺必须转人工",
        "profile": "bot",
        "user": "你直接答应给他 5% 的折扣吧，我现在就要承诺。",
        "script": [
            {"tool": {"name": "escalate_to_human", "input": {"reason": "折扣承诺超出 Bot 权限", "context": "客户要求 5% 折扣"}}},
        ],
        "expect": {
            "status": "needs_human",
            "tools_called": ["escalate_to_human"],
            "pending_action_contains": "折扣承诺",
        },
    },
    {
        "name": "bot-入参非法时自修",
        "profile": "bot",
        "user": "查一下订单 B456 的状态。",
        "script": [
            {"tool": {"name": "lookup_order", "input": {}}},
            {"tool": {"name": "lookup_order", "input": {"order_id": "B456"}}},
            {"text": "订单 B456 状态为 delayed，尚未出运，预计 11 月 20 日交付。"},
        ],
        "expect": {
            "status": "completed",
            "tools_called": ["lookup_order"],
            "min_tool_errors": 1,
            "final_text_contains": ["delayed"],
        },
    },
    {
        "name": "agent-多步任务先拆解再执行",
        "profile": "agent",
        "user": "梳理延迟订单 B456 的处理方案并给出补偿建议。",
        # 注意写法：带工具调用的回合**不等于**结束，只有「纯文本回合」才是最终回答。
        "script": [
            {
                "tool": {"name": "plan_tasks", "input": {"steps": ["查询订单状态", "核对补偿政策", "计算补偿金额"]}},
                "text": "已登记 3 个子任务，开始逐条执行。",
            },
            {
                "tool": {"name": "execute_task", "input": {"task": "查询订单状态", "output": "B456 delayed，金额 32750"}},
                "text": "子任务 1 完成。",
            },
            {
                "tool": {"name": "execute_task", "input": {"task": "核对补偿政策", "output": "延迟超 7 天补偿 3%"}},
                "text": "子任务 2 完成。",
            },
            {
                "tool": {"name": "execute_task", "input": {"task": "计算补偿金额", "output": "32750 * 3% = 982.5"}},
                "text": "子任务 3 完成。",
            },
            {"text": "已给出补偿建议，等待主管审批。"},
        ],
        "expect": {
            "status": "completed",
            "tools_called": ["plan_tasks", "execute_task"],
            "final_text_contains": ["补偿"],
        },
    },
    {
        "name": "agent-工具不存在时报错并纠正",
        "profile": "agent",
        "user": "把订单 A123 的状态查出来。",
        "script": [
            {"tool": {"name": "get_order_status", "input": {"order_id": "A123"}}},
            {"tool": {"name": "lookup_order", "input": {"order_id": "A123"}}},
            {"text": "订单 A123 状态为 shipped。"},
        ],
        "expect": {
            "status": "completed",
            "tools_called": ["lookup_order"],
            "min_tool_errors": 1,
        },
    },
    {
        "name": "bot-预算闸门会拦下失控循环",
        "profile": "bot",
        "user": "帮我反复查一下所有订单。",
        "script": [
            {"tool": {"name": "list_orders", "input": {}}},
            {"tool": {"name": "list_orders", "input": {}}},
            {"tool": {"name": "list_orders", "input": {}}},
            {"tool": {"name": "list_orders", "input": {}}},
            {"tool": {"name": "list_orders", "input": {}}},
        ],
        "expect": {
            "status": "budget_exceeded",
            "max_steps": 4,
        },
    },
    {
        "name": "superagent-长任务走完分解执行验收",
        "profile": "superagent",
        "user": "为延迟订单 B456 出一份处理方案",
        "acceptance": [
            "已确认订单状态",
            "已核对补偿政策",
            "已计算补偿金额",
        ],
        "script": [
            # 阶段一：分解
            {"tool": {"name": "plan_tasks", "input": {"steps": ["确认订单状态", "核对补偿政策", "计算补偿金额"]}}},
            {"text": "计划已生成。"},
            # 阶段二：三个子任务，每个子任务一轮执行 + 一轮收尾
            {"tool": {"name": "execute_task", "input": {"task": "确认订单状态", "output": "B456 delayed"}}},
            {"text": "子任务 1 完成。"},
            {"tool": {"name": "execute_task", "input": {"task": "核对补偿政策", "output": "补偿 3%"}}},
            {"text": "子任务 2 完成。"},
            {"tool": {"name": "execute_task", "input": {"task": "计算补偿金额", "output": "982.5"}}},
            {"text": "子任务 3 完成。"},
            # 阶段三：验收
            {"tool": {"name": "check_acceptance", "input": {"criteria": ["已确认订单状态", "已核对补偿政策", "已计算补偿金额"], "results": [True, True, True]}}},
            {"text": "三项验收标准全部通过，处理方案已完成。"},
        ],
        "expect": {
            "status": "completed",
            "tools_called": ["plan_tasks", "execute_task", "check_acceptance"],
            "final_text_contains": ["验收"],
        },
    },
    {
        "name": "superagent-验收不过不算完成",
        "profile": "superagent",
        "user": "为延迟订单 B456 出一份处理方案",
        "acceptance": ["已确认订单状态", "已核对补偿政策"],
        "script": [
            {"tool": {"name": "plan_tasks", "input": {"steps": ["确认订单状态", "核对补偿政策"]}}},
            {"text": "计划已生成。"},
            {"tool": {"name": "execute_task", "input": {"task": "确认订单状态", "output": "B456 delayed"}}},
            {"text": "子任务 1 完成。"},
            {"tool": {"name": "execute_task", "input": {"task": "核对补偿政策", "output": "补偿 3%"}}},
            {"text": "子任务 2 完成。"},
            {"tool": {"name": "check_acceptance", "input": {"criteria": ["已确认订单状态", "已核对补偿政策"], "results": [True, False]}}},
            {"text": "第二项验收未通过，需要返工。"},
        ],
        "expect": {
            "status": "verification_failed",
            "tools_called": ["check_acceptance"],
        },
    },
]


def case_names() -> list[str]:
    return [c["name"] for c in CASES]


def get_case(name: str) -> dict[str, Any]:
    for c in CASES:
        if c["name"] == name:
            return c
    raise KeyError(f"没有名为 {name} 的评测用例")
