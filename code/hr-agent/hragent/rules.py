"""词典与正则：能力图谱、敏感字段、简历字段抽取。

这一层是**规则后端**的知识来源。它同时有三个用途：
1. 离线评测的确定性基线（没有 API 也能跑通全流程）；
2. 真实模型输出的**校验器** —— 模型抽出来的能力项要能对回词典，对不上就标红；
3. 敏感字段的兜底屏蔽 —— 即便模型抽取层漏了，规则层还能挡住年龄、婚育这类字段。

关键词一律小写匹配；中文没有大小写问题，英文技能名统一小写。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 能力图谱
# ---------------------------------------------------------------------------

#: key → (中文名, 关键词, 描述, 默认权重, 期望水平 1~5)
COMPETENCY_LEXICON: dict[str, tuple[str, list[str], str, float, int]] = {
    # --- 外贸业务 ---
    "customer_dev": (
        "客户开发与获客",
        ["客户开发", "获客", "询盘", "leads", "线索", "开发客户", "陌拜", "展会"],
        "主动开发新客户，管理从线索到成交的全过程",
        1.2,
        4,
    ),
    "quotation": (
        "报价与成本核算",
        ["报价", "成本核算", "fob", "cif", "exw", "退税", "核价", "毛利"],
        "核算产品成本与税费，给出有竞争力的报价",
        1.1,
        3,
    ),
    "order_followup": (
        "跟单与交付管理",
        ["跟单", "交期", "生产进度", "出货", "交付", "排产"],
        "跟进订单从下单到出货的全链路进度",
        1.1,
        3,
    ),
    "supply_chain": (
        "供应链与物流",
        ["供应链", "物流", "清关", "报关", "海运", "空运", "仓储", "货代"],
        "组织物流与清关环节，控制运输成本与时效",
        0.9,
        3,
    ),
    "payment_risk": (
        "收汇与风险控制",
        ["收汇", "信用证", "l/c", "t/t", "汇率", "风控", "账期", "坏账"],
        "管理回款节奏与信用风险",
        1.0,
        3,
    ),
    "cross_border": (
        "跨境平台运营",
        ["amazon", "亚马逊", "shopee", "lazada", "独立站", "shopify", "跨境电商", "平台运营"],
        "在跨境平台上做选品、listing 与流量运营",
        0.9,
        3,
    ),
    "compliance": (
        "合规与产品认证",
        ["认证", "ce ", "rohs", "iso", "reach", "合规", "报关单据"],
        "确保产品与单据符合目标市场法规",
        0.8,
        3,
    ),
    "data_tool": (
        "数据工具与分析",
        ["excel", "vlookup", "数据透视", "power bi", "tableau", "sql", "数据分析"],
        "用表格与 BI 工具做经营分析与日常管理",
        0.9,
        3,
    ),
    # --- AI / 工程 ---
    "python_dev": (
        "Python 工程能力",
        ["python", "django", "fastapi", "flask", "pytest", "pandas", "numpy"],
        "用 Python 交付可维护的服务与数据处理逻辑",
        1.3,
        4,
    ),
    "llm_app": (
        "大模型应用开发",
        ["llm", "大模型", "prompt", "提示词", "claude", "gpt", "多模态", "token"],
        "基于大模型构建可用应用，处理成本与效果权衡",
        1.3,
        4,
    ),
    "agent_dev": (
        "Agent 与工具编排",
        ["agent", "智能体", "工具调用", "function call", "mcp", "工作流", "编排"],
        "设计带工具调用与状态管理的智能体流程",
        1.2,
        4,
    ),
    "rag": (
        "RAG 与知识库",
        ["rag", "检索增强", "向量", "embedding", "知识库", "rerank", "分块"],
        "搭建检索增强的知识问答系统",
        1.1,
        3,
    ),
    "deployment": (
        "部署与工程化",
        ["docker", "部署", "kubernetes", "ci/cd", "流水线", "监控", "日志", "边缘"],
        "把应用稳定部署到生产并持续运维",
        1.0,
        3,
    ),
    "vision": (
        "计算机视觉",
        ["yolo", "目标检测", "opencv", "视觉", "图像", "分割", "多模态"],
        "用视觉模型解决检测与识别类问题",
        0.9,
        3,
    ),
    # --- 通用软技能 ---
    "communication": (
        "沟通与跨部门协作",
        ["沟通", "跨部门", "协作", "团队", "对齐", "协调"],
        "与客户、工厂、货代、内部团队高效协作",
        1.0,
        4,
    ),
    "english": (
        "英文工作能力",
        ["英语", "英文", "cet-6", "cet6", "六级", "雅思", "托福", "business english", "商务英语"],
        "用英文处理邮件、谈判与文档",
        1.0,
        3,
    ),
    "ownership": (
        "结果导向与owner意识",
        ["owner", "主导", "负责", "从0到1", "从 0 到 1", "独立负责", "推动落地"],
        "端到端负责一件事并对结果负责",
        1.0,
        4,
    ),
}

SOFT_SKILL_LEXICON = {
    "沟通表达": ["沟通", "表达", "汇报", "演讲", "客户关系"],
    "抗压": ["抗压", "压力", "高强度", "加班"],
    "学习能力": ["学习能力", "快速学习", "自驱", "自省"],
    "团队协作": ["团队", "协作", "配合", "跨部门"],
    "结果导向": ["结果导向", "业绩", "目标达成", "kpi"],
}


def find_competency_hits(text: str) -> dict[str, list[str]]:
    """文本里命中了哪些能力维度，以及命中的关键词。"""
    low = text.lower()
    hits: dict[str, list[str]] = {}
    for key, (_name, keywords, _desc, _w, _lv) in COMPETENCY_LEXICON.items():
        matched = [kw for kw in keywords if kw in low]
        if matched:
            hits[key] = matched
    return hits


# ---------------------------------------------------------------------------
# 敏感字段（招聘场景的法律与伦理红线）
# ---------------------------------------------------------------------------

#: (字段名, 正则) —— 命中即屏蔽
SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    ("age", r"(?:年龄\s*[:：]?\s*)?(\d{2})\s*岁"),
    ("birth_year", r"(?:出生(?:年月|日期)?\s*[:：]?\s*)((?:19|20)\d{2})"),
    ("gender", r"性别\s*[:：]?\s*[男女]"),
    ("gender", r"(?<![一-龥])(男|女)(?![一-龥])[，,、\s]*\d{2}\s*岁"),
    ("marital", r"婚(?:姻|姻状况?|生)\s*[:：]?\s*[^\n，,。；;]{0,8}"),
    ("marital", r"(?<![一-龥])(未婚|已婚|离异|丧偶)(?![一-龥])"),
    ("ethnicity", r"民族\s*[:：]?\s*[^\n，,。；;]{0,8}"),
    ("native_place", r"(?:籍贯|出生地|户口|户籍)\s*[:：]?\s*[^\n，,。；;]{0,10}"),
    ("political", r"政治面貌\s*[:：]?\s*[^\n，,。；;]{0,8}"),
    ("photo", r"(?:近期)?(?:照片|免冠照|证件照)"),
    ("id_card", r"身份证(?:号)?\s*[:：]?\s*[0-9Xx*]{6,20}"),
    ("health", r"(?:健康状况|体检结果|既往病史|疾病史)\s*[:：]?\s*[^\n，,。；;]{0,12}"),
    ("family", r"(?:家庭成员|父亲|母亲|配偶|子女)\s*[:：]?\s*[^\n，,。；;]{0,12}"),
]

#: 这些字段在评分前必须屏蔽（法律明确禁止或高争议）
MUST_MASK_FIELDS = ("age", "gender", "marital", "ethnicity", "native_place", "political", "photo", "id_card", "health")


# ---------------------------------------------------------------------------
# 简历字段解析
# ---------------------------------------------------------------------------

EDUCATION_LEVELS: list[tuple[str, int]] = [
    ("博士", 5),
    ("硕士", 4),
    ("研究生", 4),
    ("本科", 3),
    ("学士", 3),
    ("大专", 2),
    ("专科", 2),
    ("高中", 1),
    ("中专", 1),
]

_CITY_LIST = [
    "北京", "上海", "广州", "深圳", "杭州", "苏州", "南京", "武汉", "成都", "西安",
    "青岛", "宁波", "义乌", "厦门", "东莞", "佛山", "汕头", "中山", "嘉兴", "泉州",
    "石家庄", "郑州", "长沙", "合肥", "天津", "重庆", "沈阳", "大连", "哈尔滨",
]

_YEARS_PATTERNS = [
    re.compile(r"(\d+)\s*[-~到至]\s*(\d+)\s*年"),
    re.compile(r"(\d+)\s*年(?:以上)?(?:相关|工作|从业|运营|开发|管理|经验)"),
    re.compile(r"(\d+)\s*年"),
]

_SALARY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[-~到至]\s*(\d+(?:\.\d+)?)\s*[kK]")
_SALAY_RANGE_WAN = re.compile(r"(\d+(?:\.\d+)?)\s*[-~到至]\s*(\d+(?:\.\d+)?)\s*万")


def extract_education(text: str) -> tuple[str, int]:
    """返回 (学历名称, 等级)。等级：博士5 硕士4 本科3 大专2 高中1，未知0。"""
    for name, level in EDUCATION_LEVELS:
        if name in text:
            return name, level
    return "未知", 0


def extract_years(text: str) -> float:
    """工作年限。取区间下限 —— 招聘口径里「3-5年经验」通常指至少 3 年。"""
    for pattern in _YEARS_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return 0.0


def extract_city(text: str) -> str | None:
    for city in _CITY_LIST:
        if city in text:
            return city
    return None


def extract_salary_range(text: str) -> tuple[float, float] | None:
    """期望薪资区间，单位 K/月。"""
    m = _SALARY_PATTERN.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _SALAY_RANGE_WAN.search(text)
    if m:
        return float(m.group(1)) * 10, float(m.group(2)) * 10
    return None


def extract_salary_expectation_k(text: str) -> float | None:
    """候选人期望月薪（K）。取区间下限，缺失则 None。"""
    rng = extract_salary_range(text)
    return rng[0] if rng else None


def find_soft_skills(text: str) -> list[str]:
    out = []
    for name, keywords in SOFT_SKILL_LEXICON.items():
        if any(k in text for k in keywords):
            out.append(name)
    return out
