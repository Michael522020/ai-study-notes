"""模拟面试引擎。

产品形态：**按能力模型出题 → 逐题追问 → 按维度打分 → 生成反馈报告**。

和真实面试相比，模拟面试的价值不在「像人」，而在三点：
1. **可重复** —— 同一份简历、同一套模型，每次问的题一样，候选人之间可比；
2. **可控** —— 想验证「这个候选人到底会不会 RAG」，可以只问 RAG 那一维；
3. **低风险** —— 候选人先在这里练一轮，正式面试不再浪费在「基础问题没准备」上。

因此引擎的核心不是「生成得像样的问题」，而是**问题必须绑定到能力维度**，
**评分必须能引用候选人的原话**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agentkit.provider import Provider

from .model import (
    Answer,
    CompetencyModel,
    DimensionScore,
    InterviewEvaluation,
    InterviewSession,
    Question,
)

# ---------------------------------------------------------------------------
# 题库（规则后端）
# ---------------------------------------------------------------------------

#: key → (问题, 期望要点, 追问)
QUESTION_BANK: dict[str, tuple[str, list[str], str]] = {
    "customer_dev": (
        "请用一个具体案例讲你开发新客户的过程：从线索来源到首单成交，中间你做了哪些动作？",
        ["线索", "客户", "开发", "成交", "渠道", "转化"],
        "那批线索最终的成交转化率大概是多少？卡在什么地方？",
    ),
    "quotation": (
        "一笔海外订单从收到询盘到报出最终价格，中间你要核算哪些成本项？",
        ["成本", "报价", "税费", "运费", "汇率", "毛利"],
        "如果海运费在这个季度涨了 15%，你的报价会怎么调？",
    ),
    "order_followup": (
        "描述一次你处理交期延误的完整过程，谁在什么节点给了你什么信息？",
        ["交期", "生产", "排产", "进度", "客户", "沟通"],
        "如果客户要求你保证一个你不敢保证的交期，你会怎么回复？",
    ),
    "supply_chain": (
        "你在物流与清关环节做过哪些具体决策？举一个你改变了供应商选择的例子。",
        ["物流", "清关", "货代", "仓", "成本", "时效"],
        "那次的成本变化最终是怎么传导到报价上的？",
    ),
    "payment_risk": (
        "一个新客户要求账期 60 天，你会看哪些信息来决定是否给？",
        ["账期", "信用", "风险", "记录", "定金", "保险"],
        "如果这家客户之前有过一次逾期，你的决策会怎么变？",
    ),
    "cross_border": (
        "讲一个你在跨境平台上做选品或 listing 的例子，你用什么数据做判断？",
        ["选品", "listing", "平台", "数据", "转化", "关键词"],
        "你的判断最终带来了什么可量化的变化？",
    ),
    "compliance": (
        "产品要进入欧盟市场，认证流程里哪些环节最容易出问题？",
        ["认证", "合规", "标准", "文件", "检测", "标识"],
        "如果客户要求跳过某项认证，你会怎么处理？",
    ),
    "data_tool": (
        "你日常用表格或 BI 工具做哪些分析？请举一个改变了决策的例子。",
        ["数据", "分析", "报表", "指标", "excel", "bi"],
        "那个例子里，如果数据本身是错的你会怎么发现？",
    ),
    "python_dev": (
        "讲一个你用 Python 解决业务问题的项目：你怎么拿到数据、怎么处理、怎么交付？",
        ["python", "数据", "接口", "脚本", "部署", "测试"],
        "这段代码如果数据量涨 100 倍，你会先改哪里？",
    ),
    "llm_app": (
        "你做过的 LLM 应用是怎么把「模型能力」变成「业务可用」的？中间的坑有哪些？",
        ["模型", "prompt", "评测", "成本", "延迟", "幻觉", "结构化"],
        "你怎么衡量效果是变好了还是只是看起来变好了？",
    ),
    "agent_dev": (
        "一个 Agent 要调工具才能干活。你怎么保证它调错工具时不会造成实际损失？",
        ["工具", "校验", "权限", "人工", "回滚", "状态"],
        "如果一个工具调用连续失败三次，你会怎么处理？",
    ),
    "rag": (
        "知识库问答总是答不准。你会怎么定位问题出在检索还是生成？",
        ["检索", "分块", "召回", "重排", "引用", "评测"],
        "你怎么构造评测集来判断改动是有效的？",
    ),
    "deployment": (
        "描述一个你把应用部署到生产并持续运维的过程，出了问题你怎么定位？",
        ["部署", "监控", "日志", "回滚", "灰度", "告警"],
        "你上线前一定会做哪一件事？",
    ),
    "vision": (
        "讲一个你用视觉模型做检测或识别的项目，数据和准确率是怎么解决的？",
        ["检测", "数据", "标注", "准确率", "量化", "阈值"],
        "如果现场环境光线变了，你的方案会怎么退化？",
    ),
    "communication": (
        "举一个你和客户或内部团队产生分歧的例子，最后是怎么推动到结论的？",
        ["沟通", "分歧", "对齐", "客户", "结论", "书面"],
        "如果对方始终不接受你的结论，你会怎么收尾？",
    ),
    "english": (
        "请用英文介绍一次你处理客户投诉的经历。",
        ["customer", "apologize", "refund", "resolve", "email"],
        "如果客户的英文邮件里有法律措辞，你会怎么处理？",
    ),
    "ownership": (
        "讲一件你从 0 负责到落地的事，你在里面承担了什么角色？",
        ["负责", "主导", "落地", "从0到1", "结果", "协作"],
        "这件事里如果重来一次，你会改哪个决定？",
    ),
}

DEFAULT_QUESTION_TEMPLATE = (
    "请结合你的实际经历，谈谈你在「{name}」这块做过什么，以及你现在的水平处于什么位置。",
    ["负责", "项目", "结果", "经验"],
    "能再举一个更具体的例子吗？",
)

# ---------------------------------------------------------------------------
# 后端协议
# ---------------------------------------------------------------------------


class InterviewerBackend(Protocol):
    name: str

    def plan(
        self, model: CompetencyModel, resume_text: str, *, max_questions: int
    ) -> list[Question]: ...

    def follow_up(self, question: Question, answer: str, model: CompetencyModel) -> str: ...

    def evaluate(
        self, answers: list[Answer], model: CompetencyModel
    ) -> InterviewEvaluation: ...


# ---------------------------------------------------------------------------
# 规则后端
# ---------------------------------------------------------------------------


class RuleInterviewerBackend:
    """确定性出题与评分。用于离线回归与基线对照。"""

    name = "rule"

    def plan(
        self, model: CompetencyModel, resume_text: str, *, max_questions: int = 6
    ) -> list[Question]:
        ordered = sorted(
            model.competencies, key=lambda c: (not c.required, -c.weight, c.key)
        )
        questions: list[Question] = []
        for comp in ordered:
            bank = QUESTION_BANK.get(comp.key)
            if bank:
                text, points, follow = bank
            else:
                text, points, follow = DEFAULT_QUESTION_TEMPLATE
                text = text.format(name=comp.name)
            questions.append(
                Question(
                    dimension=comp.key,
                    level=comp.min_level,
                    question=text,
                    expected_points=list(points),
                    follow_up=follow,
                )
            )
            if len(questions) >= max_questions:
                break
        return questions

    def follow_up(self, question: Question, answer: str, model: CompetencyModel) -> str:
        """追问策略：过短 → 追问补充；中等 → 不追问；充分展开 → 顺着细节追问。"""
        stripped = answer.strip()
        if len(stripped) < 20:
            return "这个回答比较简短，能补充一个具体的事例或数字吗？"
        if len(stripped) < 60:
            return ""
        return question.follow_up or ""

    def evaluate(
        self, answers: list[Answer], model: CompetencyModel
    ) -> InterviewEvaluation:
        by_dim: dict[str, list[Answer]] = {}
        for a in answers:
            by_dim.setdefault(a.question.dimension, []).append(a)

        dims: list[DimensionScore] = []
        strengths: list[str] = []
        gaps: list[str] = []

        for comp in model.competencies:
            group = by_dim.get(comp.key)
            if not group:
                continue
            text = " ".join(a.text for a in group)
            points = group[0].question.expected_points or ["负责", "项目"]
            hit = [p for p in points if p.lower() in text.lower()]
            coverage = len(hit) / len(points)
            has_number = any(ch.isdigit() for ch in text)
            level = 1 + int(round(coverage * 3.0)) + (1 if has_number and coverage >= 0.5 else 0)
            level = max(1, min(5, level))
            score = level / 5.0
            comment = f"命中要点 {len(hit)}/{len(points)}" + ("，回答含具体数字" if has_number else "")
            dims.append(
                DimensionScore(
                    dimension=comp.key,
                    level=level,
                    score=score,
                    comment=comment,
                    evidence=group[0].text[:120],
                )
            )
            name = comp.name
            if level >= 4:
                strengths.append(f"{name}（{level}/5）")
            elif level <= 2:
                gaps.append(f"{name}（{level}/5）")

        if dims:
            weights = {c.key: c.weight for c in model.competencies}
            total_w = sum(weights.get(d.dimension, 1.0) for d in dims)
            overall = (
                sum(d.score * weights.get(d.dimension, 1.0) for d in dims) / total_w * 100
                if total_w
                else 0.0
            )
        else:
            overall = 0.0

        if overall >= 75:
            recommendation = "推荐进入下一轮"
        elif overall >= 55:
            recommendation = "有条件推荐（需补充薄弱维度证据）"
        else:
            recommendation = "不推荐"

        return InterviewEvaluation(
            overall=round(overall, 2),
            dimensions=dims,
            strengths=strengths,
            gaps=gaps,
            recommendation=recommendation,
            comments=["评分基于要点覆盖度与具体程度，用于横向可比，不等同于最终录用判断。"],
        )


# ---------------------------------------------------------------------------
# 模型后端
# ---------------------------------------------------------------------------

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "description": "能力 key，必须来自能力模型"},
                    "level": {"type": "integer", "description": "考察水平 1~5"},
                    "question": {"type": "string", "description": "面试题原文"},
                    "expected_points": {"type": "array", "items": {"type": "string"}},
                    "follow_up": {"type": "string", "description": "可能的追问"},
                },
                "required": ["dimension", "level", "question", "expected_points"],
            },
        }
    },
    "required": ["questions"],
}

EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "level": {"type": "integer", "description": "1~5"},
                    "comment": {"type": "string"},
                    "evidence": {"type": "string", "description": "候选人原话片段"},
                },
                "required": ["dimension", "level", "comment"],
            },
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
    },
    "required": ["dimensions", "strengths", "gaps", "recommendation"],
}

INTERVIEW_SYSTEM_PROMPT = """你是面试官助手，负责按能力模型出题并评估回答。

出题规则：
1. 每道题绑定一个能力维度，题目要能引出具体经历，不问「你能力吗」这类空问题；
2. 题目难度随该维度的期望水平走；
3. 追问要顺着回答里的具体细节，不要重复原题。

评估规则：
1. 只根据候选人的回答内容判断，不根据简历上没写的东西推测；
2. evidence 必须是候选人原话片段；
3. gaps 要具体到维度，不要写「经验不足」这种空话。
"""


class LlmInterviewerBackend:
    """模型出题 + 模型评分。"""

    name = "llm"

    def __init__(
        self, provider: Provider, model: str = "claude-opus-5-5", effort: str = "medium"
    ):
        self.provider = provider
        self.model = model
        self.effort = effort

    def plan(
        self, model: CompetencyModel, resume_text: str, *, max_questions: int = 6
    ) -> list[Question]:
        spec = "\n".join(
            f"- key={c.key} 名称={c.name} 期望水平={c.min_level}/5 权重={c.weight}"
            for c in model.competencies
        )
        resume_block = f"\n候选人简历（已屏蔽敏感字段）：\n{resume_text}" if resume_text else ""
        resp = self.provider.generate(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"能力模型：\n{spec}\n\n请出 {max_questions} 道面试题。{resume_block}",
                        }
                    ],
                }
            ],
            system=INTERVIEW_SYSTEM_PROMPT,
            output_schema=PLAN_SCHEMA,
            effort=self.effort,
        )
        data = resp.parsed or {}
        questions: list[Question] = []
        for raw in data.get("questions", []):
            key = str(raw.get("dimension", ""))
            if model.competency(key) is None:
                continue
            questions.append(
                Question(
                    dimension=key,
                    level=int(raw.get("level", 3) or 3),
                    question=str(raw.get("question", "")),
                    expected_points=[str(p) for p in raw.get("expected_points", [])],
                    follow_up=str(raw.get("follow_up", "")),
                )
            )
        if not questions:
            return RuleInterviewerBackend().plan(model, resume_text, max_questions=max_questions)
        return questions

    def follow_up(self, question: Question, answer: str, model: CompetencyModel) -> str:
        resp = self.provider.generate(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"面试题：{question.question}\n"
                                f"候选人回答：{answer}\n\n"
                                "请写一个具体的追问，一句话；如果回答已经充分，回复空字符串。"
                            ),
                        }
                    ],
                }
            ],
            system=INTERVIEW_SYSTEM_PROMPT,
            effort=self.effort,
        )
        return resp.text.strip()

    def evaluate(
        self, answers: list[Answer], model: CompetencyModel
    ) -> InterviewEvaluation:
        transcript = "\n".join(
            f"[{a.question.dimension}] Q: {a.question.question}\nA: {a.text}" for a in answers
        )
        spec = "\n".join(
            f"- key={c.key} 名称={c.name} 权重={c.weight}" for c in model.competencies
        )
        resp = self.provider.generate(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"能力模型：\n{spec}\n\n面试记录：\n{transcript}\n\n请逐维度评估。",
                        }
                    ],
                }
            ],
            system=INTERVIEW_SYSTEM_PROMPT,
            output_schema=EVAL_SCHEMA,
            effort=self.effort,
        )
        data = resp.parsed or {}
        dims = [
            DimensionScore(
                dimension=str(d.get("dimension", "")),
                level=max(1, min(5, int(d.get("level", 3) or 3))),
                score=max(1, min(5, int(d.get("level", 3) or 3))) / 5.0,
                comment=str(d.get("comment", "")),
                evidence=str(d.get("evidence", ""))[:160],
            )
            for d in data.get("dimensions", [])
        ]
        overall = round(sum(d.score for d in dims) / len(dims) * 100, 2) if dims else 0.0
        return InterviewEvaluation(
            overall=overall,
            dimensions=dims,
            strengths=[str(s) for s in data.get("strengths", [])],
            gaps=[str(s) for s in data.get("gaps", [])],
            recommendation=str(data.get("recommendation", "")),
            comments=["模型评分仅作辅助，最终判断应由人工复核。"],
        )


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


@dataclass
class InterviewConfig:
    max_questions: int = 6
    #: 每题是否允许追问一次
    allow_follow_up: bool = True
    #: 回答短于多少字判定为「信息不足」，追问一次
    short_answer_chars: int = 20


class InterviewEngine:
    """状态机：出题 → 回答 →（可选追问）→ 评分。"""

    def __init__(
        self,
        backend: InterviewerBackend,
        *,
        config: InterviewConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or InterviewConfig()

    def start(
        self,
        role_title: str,
        model: CompetencyModel,
        resume_text: str = "",
    ) -> InterviewSession:
        questions = self.backend.plan(
            model, resume_text, max_questions=self.config.max_questions
        )
        return InterviewSession(role_title=role_title, plan=questions)

    def current_question(self, session: InterviewSession) -> Question | None:
        return session.current

    def submit(
        self, session: InterviewSession, answer_text: str
    ) -> tuple[Question | None, str]:
        """提交一次回答。返回 (已回答的问题, 追问文本)。追问为空表示进入下一题。"""
        question = session.current
        if question is None:
            return None, ""
        turn = 0
        if session.answers and session.answers[-1].question is question:
            turn = session.answers[-1].turn + 1
        session.answers.append(Answer(question=question, text=answer_text, turn=turn))

        follow_up = ""
        if self.config.allow_follow_up and turn == 0:
            follow_up = self.backend.follow_up(question, answer_text, _model_of(session))
        if follow_up:
            return question, follow_up
        session.index += 1
        return question, ""

    def finish(self, session: InterviewSession, model: CompetencyModel) -> InterviewEvaluation:
        session.finished = True
        evaluation = self.backend.evaluate(session.answers, model)
        session.evaluation = evaluation
        return evaluation


def _model_of(session: InterviewSession) -> CompetencyModel:
    """从题目维度反推一个最小能力模型，供追问时使用。"""
    from .model import Competency

    model = CompetencyModel(role_title=session.role_title)
    seen: set[str] = set()
    for q in session.plan:
        if q.dimension in seen:
            continue
        seen.add(q.dimension)
        model.competencies.append(
            Competency(key=q.dimension, name=q.dimension, keywords=[], weight=1.0, min_level=q.level)
        )
    return model
