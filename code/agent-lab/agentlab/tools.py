"""工具层：定义即契约。

三条硬规则（笔记里反复出现）：
1. **工具描述是模型唯一能读到的接口文档** —— 字段名用业务词，参数名不缩写；
2. **入参必须校验**，校验失败要把错误信息喂回模型让它自修，而不是让 Agent 崩掉；
3. **写操作必须显式标注**（`side_effect=True`），默认走人工闸门，只出草稿不落库不外发。

工具集合共用一套，按 profile 选择性挂载 —— 「工具面」本身就是产品设计：
给多了模型会乱调，给少了它绕路。
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 夹具数据（离线评测与单元测试共用；线上换成真实 API / DB）
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "shipping_delay": {
        "title": "发货延迟处理流程",
        "content": (
            "订单超过 72 小时未出库，先在系统标记 delayed，再同步给客户。"
            "延迟超过 7 天需按订单金额 3% 补偿，补偿单号回写到工单。"
        ),
        "keywords": ["发货", "延迟", "出库", "delay", "72"],
    },
    "invoice_policy": {
        "title": "发票与收款政策",
        "content": (
            "PI 在合同签署后 24 小时内发出。尾款到账后 3 个工作日内开票，"
            "增值税专票抬头必须与合同一致，不一致时先暂停开票并联系财务。"
        ),
        "keywords": ["发票", "开票", "PI", "尾款", "invoice"],
    },
    "warranty": {
        "title": "售后与质保",
        "content": (
            "标准质保 12 个月。质保期内非人为损坏免费换货，"
            "需客户提供带 SN 的照片；人为损坏按成本价 6 折收费。"
        ),
        "keywords": ["售后", "质保", "换货", "warranty", "SN"],
    },
    "payment_terms": {
        "title": "付款方式与账期",
        "content": (
            "新客户默认 T/T 30% 定金 + 70% 见提单副本付款；"
            "合作满 3 笔且无逾期可申请 L/C at sight。老客户最长账期 60 天。"
        ),
        "keywords": ["付款", "账期", "定金", "T/T", "L/C", "尾款"],
    },
    "moq_capacity": {
        "title": "起订量与产能",
        "content": (
            "常规起订量 500 件，产线日产能 3000 件，旺季（11 月至次年 2 月）产能上浮 20%。"
            "低于起订量走小单线，交付周期增加 7 天。"
        ),
        "keywords": ["起订量", "MOQ", "产能", "交期", "交付"],
    },
}

ORDERS: dict[str, dict[str, Any]] = {
    "A123": {
        "order_id": "A123",
        "customer": "Nordic Retail AB",
        "status": "shipped",
        "amount_usd": 18400,
        "placed_at": "2026-10-02",
        "shipped_at": "2026-10-06",
        "eta": "2026-11-08",
        "tracking": "DHL-9938211",
    },
    "B456": {
        "order_id": "B456",
        "customer": "Pacific Home Goods",
        "status": "delayed",
        "amount_usd": 32750,
        "placed_at": "2026-09-28",
        "shipped_at": None,
        "eta": "2026-11-20",
        "tracking": None,
    },
    "C789": {
        "order_id": "C789",
        "customer": "Alpine Outdoor GmbH",
        "status": "in_production",
        "amount_usd": 9200,
        "placed_at": "2026-10-25",
        "shipped_at": None,
        "eta": "2026-11-28",
        "tracking": None,
    },
}

#: 工单落库 —— 演示「只写内存，不真落库」的干跑语义
TICKETS: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# 安全算术求值
# ---------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """只允许算术运算的表达式求值。模型给的算式不可信，必须过白名单。"""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("只支持数字常量")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 16 or left ** right > 1e12):
                raise ValueError("幂运算超出允许范围")
            result = _BINOPS[type(node.op)](left, right)
            if abs(result) > 1e12:
                raise ValueError("结果超出允许范围")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](_eval(node.operand))
        raise ValueError("表达式包含不允许的语法")

    return _eval(ast.parse(expression, mode="eval"))


# ---------------------------------------------------------------------------
# 工具契约
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """工具返回值。状态语义由规则决定，不由正文文本决定。"""

    status: str  # ok | error | blocked | needs_human
    data: Any = None
    message: str = ""
    error_code: str | None = None

    def to_text(self) -> str:
        import json

        payload = {"status": self.status, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        if self.error_code:
            payload["error_code"] = self.error_code
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class ToolContext:
    """工具执行上下文：账、轨迹、人工闸门都从这里拿。"""

    allow_side_effects: bool = False
    dry_run: bool = True
    tracer: Any = None

    def note(self, kind: str, **fields: Any) -> None:
        if self.tracer is not None:
            self.tracer.event(kind, **fields)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], ToolResult]
    #: 是否产生外部副作用（建单、发信、改数据）
    side_effect: bool = False

    def to_provider_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# 具体工具
# ---------------------------------------------------------------------------


def _search_knowledge(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    query = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k") or 3)
    if not query:
        return ToolResult(status="error", message="query 不能为空", error_code="invalid_input")
    terms = [t.lower() for t in query.replace("，", ",").split(",") if t.strip()]
    hits = []
    for doc in KNOWLEDGE_BASE.values():
        haystack = (doc["title"] + " " + doc["content"] + " " + " ".join(doc["keywords"])).lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            hits.append((score, doc))
    hits.sort(key=lambda x: -x[0])
    data = [
        {"title": d["title"], "content": d["content"], "matched_keywords": d["keywords"]}
        for _, d in hits[:top_k]
    ]
    return ToolResult(
        status="ok",
        data=data,
        message=f"命中 {len(data)} 篇知识库文档" if data else "知识库中没有相关文档",
    )


def _lookup_order(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    order_id = str(args.get("order_id", "")).strip().upper()
    order = ORDERS.get(order_id)
    if not order:
        return ToolResult(
            status="error",
            message=f"订单 {order_id} 不存在",
            error_code="not_found",
        )
    return ToolResult(status="ok", data=order, message=f"订单 {order_id} 状态 {order['status']}")


def _calculate(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return ToolResult(status="error", message="expression 不能为空", error_code="invalid_input")
    try:
        value = safe_eval(expression)
    except Exception as exc:
        return ToolResult(status="error", message=f"表达式无法计算：{exc}", error_code="invalid_input")
    return ToolResult(status="ok", data={"value": value}, message=f"{expression} = {value}")


def _draft_reply(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    order_id = str(args.get("order_id", "")).strip().upper()
    tone = str(args.get("tone", "professional"))
    order = ORDERS.get(order_id)
    if not order:
        return ToolResult(status="error", message=f"订单 {order_id} 不存在", error_code="not_found")
    templates = {
        "professional": f"您好 {order['customer']}，订单 {order_id} 当前状态为 {order['status']}。",
        "apologetic": f"非常抱歉给您带来不便，订单 {order_id} 目前状态为 {order['status']}。",
    }
    body = templates.get(tone, templates["professional"])
    return ToolResult(
        status="ok",
        data={"order_id": order_id, "tone": tone, "draft": body, "sent": False},
        message="已生成回复草稿（未发送）",
    )


def _create_ticket(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    order_id = str(args.get("order_id", "")).strip().upper()
    issue = str(args.get("issue", "")).strip()
    urgency = str(args.get("urgency", "normal")).strip()
    if not order_id or not issue:
        return ToolResult(status="error", message="order_id 与 issue 均不能为空", error_code="invalid_input")
    if order_id not in ORDERS:
        return ToolResult(status="error", message=f"订单 {order_id} 不存在", error_code="not_found")
    if not ctx.allow_side_effects:
        # 干跑：给模型一份「如果执行会怎样」的预览，真实动作留给人工闸门
        return ToolResult(
            status="needs_human",
            data={"would_create": {"order_id": order_id, "issue": issue, "urgency": urgency}},
            message="建单属于写操作，已生成待人工确认的工单草稿，未落库",
        )
    ticket = {"ticket_id": f"T{len(TICKETS) + 1:04d}", "order_id": order_id, "issue": issue, "urgency": urgency}
    TICKETS.append(ticket)
    return ToolResult(status="ok", data=ticket, message=f"工单 {ticket['ticket_id']} 已创建")


def _escalate_to_human(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    reason = str(args.get("reason", "")).strip()
    if not reason:
        return ToolResult(status="error", message="reason 不能为空", error_code="invalid_input")
    return ToolResult(
        status="needs_human",
        data={"reason": reason, "context": args.get("context", "")},
        message="已请求人工接管",
    )


def _list_orders(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    data = [
        {"order_id": o["order_id"], "customer": o["customer"], "status": o["status"], "amount_usd": o["amount_usd"]}
        for o in ORDERS.values()
    ]
    return ToolResult(status="ok", data=data, message=f"共 {len(data)} 笔订单")


def _read_knowledge_doc(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    doc_id = str(args.get("doc_id", "")).strip()
    doc = KNOWLEDGE_BASE.get(doc_id)
    if not doc:
        return ToolResult(status="error", message=f"文档 {doc_id} 不存在", error_code="not_found")
    return ToolResult(status="ok", data=doc, message=doc["title"])


def _plan_tasks(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    steps = args.get("steps")
    if not isinstance(steps, list) or not steps:
        return ToolResult(status="error", message="steps 必须是非空数组", error_code="invalid_input")
    cleaned = [str(s).strip() for s in steps if str(s).strip()]
    if len(cleaned) != len(steps):
        return ToolResult(status="error", message="steps 中存在空项", error_code="invalid_input")
    return ToolResult(status="ok", data={"steps": cleaned}, message=f"已记录 {len(cleaned)} 个子任务")


def _execute_task(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = str(args.get("task", "")).strip()
    output = args.get("output")
    if not task:
        return ToolResult(status="error", message="task 不能为空", error_code="invalid_input")
    ctx.note("task_executed", task=task)
    return ToolResult(status="ok", data={"task": task, "output": output}, message=f"子任务完成：{task}")


def _check_acceptance(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    criteria = args.get("criteria")
    results = args.get("results")
    if not isinstance(criteria, list) or not isinstance(results, list):
        return ToolResult(
            status="error", message="criteria 与 results 必须都是数组", error_code="invalid_input"
        )
    if len(criteria) != len(results):
        return ToolResult(
            status="error",
            message="criteria 与 results 长度不一致",
            error_code="invalid_input",
        )
    detail = [{"criterion": c, "passed": bool(r)} for c, r in zip(criteria, results)]
    passed = all(d["passed"] for d in detail)
    return ToolResult(
        status="ok",
        data={"all_passed": passed, "detail": detail},
        message="验收通过" if passed else "验收未通过",
    )


def _save_checkpoint(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    state = args.get("state")
    if not isinstance(state, dict):
        return ToolResult(status="error", message="state 必须是对象", error_code="invalid_input")
    return ToolResult(status="ok", data={"saved": state}, message="检查点已保存")


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

_OBJECT = {"type": "object"}
_STRING = {"type": "string"}

_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_knowledge",
        description=(
            "在业务知识库中检索政策与流程文档（如发货延迟、发票、质保、付款账期）。"
            "参数 query 用自然语言描述要查什么，逗号分隔多个关键词可提高召回。"
            "返回命中的文档标题与正文；查不到时返回空列表，不要臆造。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {**_STRING, "description": "检索关键词或自然语言问题"},
                "top_k": {"type": "integer", "description": "返回条数，默认 3", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
        handler=_search_knowledge,
    ),
    ToolSpec(
        name="read_knowledge_doc",
        description="按文档 ID 精确读取一篇知识库文档全文。",
        input_schema={
            "type": "object",
            "properties": {"doc_id": {**_STRING, "description": "文档 ID，如 shipping_delay"}},
            "required": ["doc_id"],
        },
        handler=_read_knowledge_doc,
    ),
    ToolSpec(
        name="list_orders",
        description="列出当前客户名下的全部订单及其状态与金额。",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_list_orders,
    ),
    ToolSpec(
        name="lookup_order",
        description="按订单号查询订单详情：状态、金额、出运时间、ETA、运单号。",
        input_schema={
            "type": "object",
            "properties": {"order_id": {**_STRING, "description": "订单号，如 A123"}},
            "required": ["order_id"],
        },
        handler=_lookup_order,
    ),
    ToolSpec(
        name="calculate",
        description=(
            "计算一个算术表达式（只支持 + - * / // % ** 和括号）。"
            "涉及金额、比例、补偿的计算必须用本工具，不要心算。"
        ),
        input_schema={
            "type": "object",
            "properties": {"expression": {**_STRING, "description": "算术表达式，如 18400 * 0.03"}},
            "required": ["expression"],
        },
        handler=_calculate,
    ),
    ToolSpec(
        name="draft_reply",
        description=(
            "生成一条给客户的回复草稿。只生成不发送，sent 恒为 false。"
            "语气可选 professional / apologetic。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {**_STRING, "description": "订单号"},
                "tone": {"type": "string", "enum": ["professional", "apologetic"], "description": "语气"},
            },
            "required": ["order_id"],
        },
        handler=_draft_reply,
    ),
    ToolSpec(
        name="create_ticket",
        description=(
            "为订单创建售后工单。这是有副作用的写操作：默认只产出待人工确认的草稿，"
            "只有明确放行才会真正建单。",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {**_STRING, "description": "订单号"},
                "issue": {**_STRING, "description": "问题描述"},
                "urgency": {"type": "string", "enum": ["low", "normal", "high"], "description": "紧急程度"},
            },
            "required": ["order_id", "issue"],
        },
        handler=_create_ticket,
        side_effect=True,
    ),
    ToolSpec(
        name="escalate_to_human",
        description=(
            "当问题超出自身能力、涉及金额承诺或客户情绪激烈时，转人工接管。"
            "调用后流程会暂停等待人工，不要继续编造答案。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "reason": {**_STRING, "description": "转人工的原因"},
                "context": {**_STRING, "description": "已收集到的上下文摘要"},
            },
            "required": ["reason"],
        },
        handler=_escalate_to_human,
        side_effect=True,
    ),
    ToolSpec(
        name="plan_tasks",
        description="登记一份子任务清单，用于长任务的进度追踪。",
        input_schema={
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"}, "description": "子任务描述列表"}
            },
            "required": ["steps"],
        },
        handler=_plan_tasks,
    ),
    ToolSpec(
        name="execute_task",
        description="执行一个已登记的子任务并回填产出。",
        input_schema={
            "type": "object",
            "properties": {
                "task": {**_STRING, "description": "子任务描述"},
                "output": {"type": "string", "description": "子任务产出"},
            },
            "required": ["task"],
        },
        handler=_execute_task,
    ),
    ToolSpec(
        name="check_acceptance",
        description=(
            "对验收标准逐条判定。criteria 与 results 一一对应，全部为 true 才算通过。"
            "这是长任务收尾的自检环节，不要跳过。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "criteria": {"type": "array", "items": {"type": "string"}, "description": "验收标准列表"},
                "results": {"type": "array", "items": {"type": "boolean"}, "description": "逐条判定结果"},
            },
            "required": ["criteria", "results"],
        },
        handler=_check_acceptance,
    ),
    ToolSpec(
        name="save_checkpoint",
        description="保存长任务的检查点状态，用于中断后从断点续跑。",
        input_schema={
            "type": "object",
            "properties": {"state": {"type": "object", "description": "可序列化的状态对象"}},
            "required": ["state"],
        },
        handler=_save_checkpoint,
    ),
]


def build_registry(names: list[str] | None = None) -> "ToolRegistry":
    """按名字构建注册表；names 为 None 时挂载全部工具。"""
    selected = [t for t in _TOOL_SPECS if names is None or t.name in names]
    return ToolRegistry(selected)


@dataclass
class ToolRegistry:
    tools: list[ToolSpec] = field(default_factory=list)

    def get(self, name: str) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def provider_specs(self) -> list[dict[str, Any]]:
        return [t.to_provider_spec() for t in self.tools]

    def names(self) -> list[str]:
        return [t.name for t in self.tools]
