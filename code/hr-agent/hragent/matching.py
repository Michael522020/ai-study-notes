"""匹配后端：把简历文本与能力模型对齐成「维度分数 + 证据 + 缺口」。

两个后端：
- `RuleMatchBackend`：关键词命中 + 深度词加权，确定性、可离线；
- `LlmMatchBackend`：模型逐维度判定，但**证据必须能在原文里逐字找到**，
  找不到就丢弃这条证据并计入 `ungrounded`。

「证据必须可回指原文」是简历筛查里最重要的一条工程约束：
模型说「候选人有 RAG 经验」不算数，能指出是哪一句话才算数。
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from agentkit.provider import Provider

from .fields import ResumeFields
from .model import Competency, CompetencyModel, Evidence, MatchResult

#: 深度信号词：出现说明不只是「接触过」，而是「主导过」
DEPTH_MARKERS = ("主导", "负责", "搭建", "从0到1", "从 0 到 1", "独立", "推动", "落地", "设计", "实现")

_LINE_SPLIT = re.compile(r"[\n。；;]")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _LINE_SPLIT.split(text)]
    return [p for p in parts if p]


class MatchBackend(Protocol):
    name: str

    def match(
        self, resume_text: str, model: CompetencyModel, fields: ResumeFields
    ) -> MatchResult: ...


def _find_quote(text: str, keywords: list[str]) -> tuple[str, list[str]]:
    """找到包含关键词的原句，返回 (原句, 命中关键词)。"""
    sentences = _split_sentences(text)
    for sentence in sentences:
        low = sentence.lower()
        hits = [kw for kw in keywords if kw in low]
        if hits:
            return (sentence[:160], hits)
    return ("", [])


class RuleMatchBackend:
    """关键词 + 深度词的确定性匹配。"""

    name = "rule"

    def match(
        self, resume_text: str, model: CompetencyModel, fields: ResumeFields
    ) -> MatchResult:
        result = MatchResult()
        low = resume_text.lower()

        for comp in model.competencies:
            hits = [kw for kw in comp.keywords if kw.lower() in low]
            if not hits:
                result.scores[comp.key] = 0.0
                result.levels[comp.key] = 0
                result.missing.append(comp.key)
                continue

            level = min(5, len(hits) + 1)
            quote, _ = _find_quote(resume_text, hits)
            if quote and any(marker in quote for marker in DEPTH_MARKERS):
                level = min(5, level + 1)
            result.levels[comp.key] = level
            result.scores[comp.key] = round(level / 5.0, 4)
            result.evidences.append(
                Evidence(
                    dimension=comp.key,
                    quote=quote,
                    level=level,
                    score=round(level / 5.0, 4),
                )
            )

        result.summary = self._summarize(model, result)
        return result

    def _summarize(self, model: CompetencyModel, result: MatchResult) -> str:
        total = len(model.competencies)
        covered = sum(1 for c in model.competencies if result.scores.get(c.key, 0) > 0)
        strong = [c.name for c in model.competencies if result.levels.get(c.key, 0) >= 4]
        weak = [c.name for c in model.competencies if result.scores.get(c.key, 0) == 0]
        parts = [f"命中 {covered}/{total} 个能力维度"]
        if strong:
            parts.append("强项：" + "、".join(strong))
        if weak:
            parts.append("缺口：" + "、".join(weak))
        return "；".join(parts)


# ---------------------------------------------------------------------------
# 模型后端
# ---------------------------------------------------------------------------

MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "能力 key，必须来自能力模型"},
                    "level": {"type": "integer", "description": "水平 1~5"},
                    "evidence": {
                        "type": "string",
                        "description": "支撑该判定的**原文片段**，必须逐字出现在简历中",
                    },
                    "comment": {"type": "string", "description": "一句话说明"},
                },
                "required": ["key", "level", "evidence"],
            },
        },
        "missing": {"type": "array", "items": {"type": "string"}, "description": "简历中完全没有体现的 key"},
        "summary": {"type": "string", "description": "两句话总结"},
    },
    "required": ["dimensions", "missing", "summary"],
}

MATCH_SYSTEM_PROMPT = """你是简历与岗位能力模型的匹配器。

规则：
1. 逐个能力维度判定水平 1~5：1=听说过，2=参与过，3=能独立完成，4=主导过，5=专家级。
2. evidence 字段必须是简历中的**原文片段**，逐字照抄，不要改写、不要拼接、不要补充。
3. 简历里没有体现的维度，放进 missing，不要凭岗位名称推测。
4. 不要因为年龄、性别、婚育、籍贯、学校层次调整任何判断 —— 这些字段已被屏蔽，不应出现在判断理由中。
"""


class LlmMatchBackend:
    """模型匹配 + 证据回指校验。"""

    name = "llm"

    def __init__(
        self,
        provider: Provider,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        *,
        require_grounding: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.effort = effort
        self.require_grounding = require_grounding
        #: 被判定为「无依据」的证据
        self.ungrounded: list[dict[str, Any]] = []

    def match(
        self, resume_text: str, model: CompetencyModel, fields: ResumeFields
    ) -> MatchResult:
        spec = [
            {
                "key": c.key,
                "name": c.name,
                "description": c.description,
                "min_level": c.min_level,
                "keywords": c.keywords[:8],
            }
            for c in model.competencies
        ]
        prompt = (
            f"能力模型：\n{_dump_spec(spec)}\n\n"
            f"简历原文（已屏蔽敏感字段）：\n{resume_text}"
        )
        resp = self.provider.generate(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            system=MATCH_SYSTEM_PROMPT,
            output_schema=MATCH_SCHEMA,
            effort=self.effort,
        )
        data = resp.parsed or {}
        result = MatchResult(summary=str(data.get("summary", "")))

        for raw in data.get("dimensions", []):
            key = str(raw.get("key", ""))
            if model.competency(key) is None:
                self.ungrounded.append({"reason": "unknown_key", "payload": raw})
                continue
            level = max(0, min(5, int(raw.get("level", 0) or 0)))
            quote = str(raw.get("evidence", ""))
            grounded = bool(quote) and quote.strip() in resume_text
            if not grounded:
                self.ungrounded.append({"reason": "evidence_not_found", "payload": raw})
                if self.require_grounding:
                    # 无依据的证据不进结果：该维度按未命中处理
                    result.scores[key] = 0.0
                    result.levels[key] = 0
                    result.missing.append(key)
                    continue
            result.levels[key] = level
            result.scores[key] = round(level / 5.0, 4)
            result.evidences.append(
                Evidence(
                    dimension=key,
                    quote=quote,
                    level=level,
                    score=round(level / 5.0, 4),
                )
            )
        for key in data.get("missing", []):
            k = str(key)
            if k not in result.missing and model.competency(k) is not None:
                result.missing.append(k)
        return result


def _dump_spec(spec: list[dict[str, Any]]) -> str:
    lines = []
    for s in spec:
        lines.append(
            f"- key={s['key']} 名称={s['name']} 期望水平={s['min_level']}/5 "
            f"关键词={'、'.join(s['keywords'])}"
        )
    return "\n".join(lines)


def compute_weighted_score(
    model: CompetencyModel, match: MatchResult
) -> tuple[float, list[str]]:
    """加权总分（0~100）+ 扣分原因。

    扣分规则：JD 标为 required 的能力缺失，直接按权重扣满并记录原因。
    """
    if not model.competencies:
        return 0.0, ["能力模型为空"]
    total_weight = sum(c.weight for c in model.competencies)
    if total_weight <= 0:
        return 0.0, ["能力模型权重全为 0"]
    raw = sum(c.weight * match.scores.get(c.key, 0.0) for c in model.competencies)
    score = raw / total_weight * 100.0

    penalties: list[str] = []
    for c in model.competencies:
        if c.required and match.scores.get(c.key, 0.0) == 0:
            penalty = 100.0 * c.weight / total_weight
            score -= penalty
            penalties.append(f"缺少必备能力「{c.name}」（扣 {penalty:.1f} 分）")
    return round(max(0.0, score), 2), penalties
