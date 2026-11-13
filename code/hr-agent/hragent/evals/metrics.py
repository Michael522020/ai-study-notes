"""评测指标。

沿用 agent-lab 的分法：能离线断言的（行为、一致性、覆盖率）离线测；
需要人来判断的（结论是否合理）用**人工标注的期望值**做对照，而不是自证。

关键指标：
1. `hard_filter_accuracy` 硬门槛判对率 —— 判错的后果最严重
2. `decision_accuracy`    结论判对率
3. `evidence_grounding_rate` 证据可回指原文的比例 —— 幻觉证据必须为 0
4. `bias_invariance`      屏蔽前后得分差 —— 必须严格为 0
5. `rank_concordance`     与人工排序的一致性
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

DECISION_RANK = {"pass": 0, "review": 1, "reject": 2}


@dataclass
class ScreeningCaseResult:
    candidate: str
    expected_decision: str
    actual_decision: str
    expected_hard_filter_pass: bool
    actual_hard_filter_pass: bool
    score: float
    expected_score: float | None = None
    score_delta: float | None = None
    evidence_grounded: bool = True
    ungrounded_evidence: list[str] = field(default_factory=list)
    masked_fields: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def decision_ok(self) -> bool:
        return self.expected_decision == self.actual_decision

    @property
    def hard_filter_ok(self) -> bool:
        return self.expected_hard_filter_pass == self.actual_hard_filter_pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "expected_decision": self.expected_decision,
            "actual_decision": self.actual_decision,
            "decision_ok": self.decision_ok,
            "expected_hard_filter_pass": self.expected_hard_filter_pass,
            "actual_hard_filter_pass": self.actual_hard_filter_pass,
            "hard_filter_ok": self.hard_filter_ok,
            "score": round(self.score, 2),
            "evidence_grounded": self.evidence_grounded,
            "ungrounded_evidence": list(self.ungrounded_evidence),
            "masked_fields": list(self.masked_fields),
            "reasons": list(self.reasons),
        }


@dataclass
class InterviewCaseResult:
    questions: int = 0
    answers: int = 0
    dimensions_covered: int = 0
    dimensions_total: int = 0
    all_questions_bound: bool = True
    all_evidence_present: bool = True
    recommendation: str = ""
    followups_triggered: int = 0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "answers": self.answers,
            "dimensions_covered": self.dimensions_covered,
            "dimensions_total": self.dimensions_total,
            "all_questions_bound": self.all_questions_bound,
            "all_evidence_present": self.all_evidence_present,
            "recommendation": self.recommendation,
            "followups_triggered": self.followups_triggered,
            "issues": list(self.issues),
        }


def evaluate_screening_case(
    candidate: str,
    expected: dict[str, Any],
    decision: Any,
    resume_text: str,
) -> ScreeningCaseResult:
    """比对着人工标注检查一次筛查结论。"""
    ungrounded = [
        e.quote for e in decision.match.evidences if e.quote and e.quote not in resume_text
    ]
    result = ScreeningCaseResult(
        candidate=candidate,
        expected_decision=str(expected.get("decision", "")),
        actual_decision=decision.decision,
        expected_hard_filter_pass=bool(expected.get("hard_filter_pass", True)),
        actual_hard_filter_pass=not decision.rejected_by_filter,
        score=decision.score,
        evidence_grounded=not ungrounded,
        ungrounded_evidence=ungrounded,
        masked_fields=list(decision.masked_fields),
        reasons=list(decision.reasons),
    )
    return result


def aggregate_screening(
    results: Sequence[ScreeningCaseResult],
    *,
    expected_rank: Sequence[str],
    actual_rank: Sequence[str],
    bias_delta: float | None,
) -> dict[str, Any]:
    total = len(results)
    eligible_expected = [c for c in expected_rank]
    eligible_actual = [c for c in actual_rank if c in eligible_expected]

    # 排序一致性：只看人工标注里比较过的候选人
    pairs = 0
    concordant = 0
    for i, a in enumerate(eligible_expected):
        for b in eligible_expected[i + 1 :]:
            if a not in eligible_actual or b not in eligible_actual:
                continue
            pairs += 1
            if eligible_actual.index(a) < eligible_actual.index(b):
                concordant += 1

    top_k = min(2, len(eligible_expected))
    top_k_recall = (
        len(set(expected_rank[:top_k]) & set(eligible_actual[:top_k])) / top_k
        if top_k
        else 0.0
    )

    return {
        "cases": total,
        "hard_filter_accuracy": round(
            sum(1 for r in results if r.hard_filter_ok) / total, 4
        ) if total else 0.0,
        "decision_accuracy": round(
            sum(1 for r in results if r.decision_ok) / total, 4
        ) if total else 0.0,
        "evidence_grounding_rate": round(
            sum(1 for r in results if r.evidence_grounded) / total, 4
        ) if total else 0.0,
        "bias_score_delta": None if bias_delta is None else round(bias_delta, 9),
        "bias_invariance_pass": bias_delta is not None and abs(bias_delta) <= 1e-9,
        "rank_concordance": round(concordant / pairs, 4) if pairs else 1.0,
        "top2_recall": round(top_k_recall, 4),
        "expected_rank": list(expected_rank),
        "actual_rank": list(actual_rank),
    }
