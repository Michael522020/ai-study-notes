"""领域模型：JD 能力模型、匹配结果、筛查结论、面试状态。

全部用 dataclass，不引 ORM：这套代码的读者要能一眼看懂数据在流转。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardRequirement:
    """硬性门槛。不满足直接出局，不参与加权打分。"""

    key: str
    label: str
    operator: str  # >= | <= | in | not_in | contains | eq
    value: Any
    required: bool = True
    note: str = ""

    def describe(self) -> str:
        return f"{self.label} {self.operator} {self.value}"


@dataclass
class Competency:
    """能力维度。"""

    key: str
    name: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    #: 权重，同一维度内归一化后参与加权
    weight: float = 1.0
    #: 期望水平 1~5
    min_level: int = 3
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "keywords": list(self.keywords),
            "weight": self.weight,
            "min_level": self.min_level,
            "required": self.required,
        }


@dataclass
class CompetencyModel:
    """一份 JD 的结构化表示。"""

    role_title: str = ""
    hard_requirements: list[HardRequirement] = field(default_factory=list)
    competencies: list[Competency] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    #: 抽取方式：rule（规则）| llm（模型）
    backend: str = "rule"

    def competency(self, key: str) -> Competency | None:
        for c in self.competencies:
            if c.key == key:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_title": self.role_title,
            "backend": self.backend,
            "hard_requirements": [
                {
                    "key": h.key,
                    "label": h.label,
                    "operator": h.operator,
                    "value": h.value,
                    "required": h.required,
                }
                for h in self.hard_requirements
            ],
            "competencies": [c.to_dict() for c in self.competencies],
            "soft_skills": list(self.soft_skills),
        }


@dataclass
class Evidence:
    """一条评分证据。`quote` 必须来自原文，不允许模型编。"""

    dimension: str
    quote: str
    level: int
    score: float
    source: str = "resume"  # resume | interview | jd

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "quote": self.quote,
            "level": self.level,
            "score": round(self.score, 4),
            "source": self.source,
        }


@dataclass
class MatchResult:
    """匹配结果：每个维度的分数 + 支撑证据 + 缺口。"""

    scores: dict[str, float] = field(default_factory=dict)
    levels: dict[str, int] = field(default_factory=dict)
    evidences: list[Evidence] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "levels": dict(self.levels),
            "evidences": [e.to_dict() for e in self.evidences],
            "missing": list(self.missing),
            "summary": self.summary,
        }


@dataclass
class FilterOutcome:
    """硬性门槛的判定明细。"""

    key: str
    label: str
    expected: str
    actual: str
    passed: bool
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "required": self.required,
        }


@dataclass
class ScreeningDecision:
    """筛查结论：pass / review / reject。"""

    decision: str
    score: float
    filters: list[FilterOutcome] = field(default_factory=list)
    match: MatchResult = field(default_factory=MatchResult)
    reasons: list[str] = field(default_factory=list)
    candidate_ref: str = ""
    #: 屏蔽掉的敏感字段，供审计
    masked_fields: list[str] = field(default_factory=list)

    @property
    def rejected_by_filter(self) -> bool:
        return any(not f.passed and f.required for f in self.filters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "score": round(self.score, 2),
            "candidate_ref": self.candidate_ref,
            "filters": [f.to_dict() for f in self.filters],
            "match": self.match.to_dict(),
            "reasons": list(self.reasons),
            "masked_fields": list(self.masked_fields),
        }


@dataclass
class Question:
    """面试题。"""

    dimension: str
    level: int
    question: str
    #: 期望考察的要点，用于离线评分
    expected_points: list[str] = field(default_factory=list)
    follow_up: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "level": self.level,
            "question": self.question,
            "expected_points": list(self.expected_points),
            "follow_up": self.follow_up,
        }


@dataclass
class Answer:
    question: Question
    text: str
    #: 追问轮次，0 表示首答
    turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "text": self.text,
            "turn": self.turn,
        }


@dataclass
class DimensionScore:
    dimension: str
    level: int
    score: float
    comment: str = ""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "level": self.level,
            "score": round(self.score, 4),
            "comment": self.comment,
            "evidence": self.evidence,
        }


@dataclass
class InterviewEvaluation:
    overall: float
    dimensions: list[DimensionScore] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommendation: str = ""
    comments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 2),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "strengths": list(self.strengths),
            "gaps": list(self.gaps),
            "recommendation": self.recommendation,
            "comments": list(self.comments),
        }


@dataclass
class InterviewSession:
    """一次模拟面试的完整状态。"""

    role_title: str
    plan: list[Question] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)
    index: int = 0
    finished: bool = False
    evaluation: InterviewEvaluation | None = None

    @property
    def current(self) -> Question | None:
        if self.index < len(self.plan):
            return self.plan[self.index]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_title": self.role_title,
            "plan": [q.to_dict() for q in self.plan],
            "answers": [a.to_dict() for a in self.answers],
            "finished": self.finished,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }
