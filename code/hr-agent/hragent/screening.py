"""简历筛查流水线。

四段式，每段都能单独评测：

    屏蔽敏感字段 → 硬性门槛过滤 → 语义匹配打分 → 结论与理由

设计上的两个关键取舍：
1. **硬过滤与打分分离** —— 硬门槛是「一票否决」，不进加权分数；
   把学历也折成几分是错的：学历不达标的简历不该出现在排序里，而不是排在最后。
2. **结论必须带理由与证据** —— 任何一个 pass/reject 都要能回答
   「凭哪一句原文、哪条门槛」。不能回答的结论一律降级为 review，交给人看。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentkit.tracing import Tracer

from .bias import MaskResult, mask_sensitive
from .fields import ResumeFields, extract_fields
from .matching import MatchBackend, compute_weighted_score
from .model import CompetencyModel, FilterOutcome, ScreeningDecision


@dataclass
class ScreeningConfig:
    #: 进入 pass 的分数线
    pass_score: float = 75.0
    #: 进入 review 的分数线，低于则 reject
    review_score: float = 55.0
    #: 是否在评分前屏蔽敏感字段。生产环境必须为 True
    mask_sensitive: bool = True
    #: 是否要求每个命中维度都有原文证据
    require_evidence: bool = True
    #: 有证据的维度占比低于此值就降级为 review
    min_evidence_ratio: float = 0.6


class ScreeningPipeline:
    def __init__(
        self,
        match_backend: MatchBackend,
        *,
        config: ScreeningConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.match_backend = match_backend
        self.config = config or ScreeningConfig()
        self.tracer = tracer

    # -- 硬门槛 -------------------------------------------------------------

    def _evaluate_filters(
        self, model: CompetencyModel, fields: ResumeFields
    ) -> list[FilterOutcome]:
        outcomes: list[FilterOutcome] = []
        for req in model.hard_requirements:
            actual, passed, note = self._check(req.key, req.operator, req.value, fields)
            outcome = FilterOutcome(
                key=req.key,
                label=req.label,
                expected=f"{req.operator} {req.value}",
                actual=actual,
                passed=passed,
                required=req.required,
            )
            if note:
                outcome.expected = f"{outcome.expected}（{note}）"
            outcomes.append(outcome)
        return outcomes

    @staticmethod
    def _check(
        key: str, operator: str, value: Any, fields: ResumeFields
    ) -> tuple[str, bool, str]:
        """返回 (实际值, 是否通过, 备注)。"""
        if key == "education":
            if fields.education_level == 0:
                return ("简历未注明", True, "未注明不判负")
            if operator == ">=":
                return (fields.education_name, fields.education_level >= float(value), "")
            return (fields.education_name, fields.education_level == float(value), "")
        if key == "years":
            if fields.years == 0:
                return ("简历未注明", True, "未注明不判负")
            if operator == ">=":
                return (f"{fields.years:g} 年", fields.years >= float(value), "")
            return (f"{fields.years:g} 年", fields.years <= float(value), "")
        if key == "city":
            if fields.city is None:
                return ("简历未注明", True, "未注明不判负")
            return (fields.city, fields.city == str(value), "")
        if key == "salary_k":
            if fields.salary_k is None:
                return ("简历未注明", True, "未注明不判负")
            limit = float(value)
            if operator == "<=":
                return (f"{fields.salary_k:g}K", fields.salary_k <= limit, "")
            return (f"{fields.salary_k:g}K", fields.salary_k >= limit, "")
        return ("未支持", True, "未知门槛类型，按通过处理")

    # -- 主流程 -------------------------------------------------------------

    def screen(
        self,
        resume_text: str,
        model: CompetencyModel,
        *,
        candidate_ref: str = "",
    ) -> ScreeningDecision:
        mask: MaskResult | None = None
        working_text = resume_text
        if self.config.mask_sensitive:
            mask = mask_sensitive(resume_text)
            working_text = mask.text

        fields = extract_fields(working_text)
        filters = self._evaluate_filters(model, fields)
        failed_required = [f for f in filters if not f.passed and f.required]

        if failed_required:
            reasons = [
                f"硬性门槛未通过：{f.label}（期望 {f.expected}，实际 {f.actual}）"
                for f in failed_required
            ]
            decision = ScreeningDecision(
                decision="reject",
                score=0.0,
                filters=filters,
                reasons=reasons,
                candidate_ref=candidate_ref,
                masked_fields=mask.masked_fields if mask else [],
            )
            self._trace("screen_reject_hard_filter", decision)
            return decision

        match = self.match_backend.match(working_text, model, fields)
        score, penalties = compute_weighted_score(model, match)

        evidence_dims = {e.dimension for e in match.evidences}
        scored_dims = {k for k, v in match.scores.items() if v > 0}
        ratio = len(evidence_dims & scored_dims) / len(scored_dims) if scored_dims else 1.0

        reasons: list[str] = []
        reasons.extend(penalties)
        if match.summary:
            reasons.append(match.summary)

        if score >= self.config.pass_score:
            decision_name = "pass"
        elif score >= self.config.review_score:
            decision_name = "review"
        else:
            decision_name = "reject"

        if self.config.require_evidence and ratio < self.config.min_evidence_ratio:
            if decision_name == "pass":
                decision_name = "review"
            reasons.append(f"证据覆盖不足（{ratio:.0%}），降级为人工复核")

        decision = ScreeningDecision(
            decision=decision_name,
            score=score,
            filters=filters,
            match=match,
            reasons=reasons,
            candidate_ref=candidate_ref,
            masked_fields=mask.masked_fields if mask else [],
        )
        self._trace("screen_done", decision)
        return decision

    def _trace(self, kind: str, decision: ScreeningDecision) -> None:
        if self.tracer is None:
            return
        self.tracer.event(
            kind,
            candidate_ref=decision.candidate_ref,
            decision=decision.decision,
            score=decision.score,
            masked_fields=decision.masked_fields,
            reasons=decision.reasons,
        )
