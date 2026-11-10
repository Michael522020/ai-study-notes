"""筛查流水线测试。"""

from __future__ import annotations

from hragent.evals.fixtures import EXPECTED, JD_TEXT, RESUMES
from hragent.jd import RuleJdBackend
from hragent.matching import RuleMatchBackend, compute_weighted_score
from hragent.screening import ScreeningConfig, ScreeningPipeline


def _pipeline() -> ScreeningPipeline:
    return ScreeningPipeline(
        RuleMatchBackend(),
        config=ScreeningConfig(pass_score=70.0, review_score=45.0),
    )


def test_jd_model_extracts_expected_dimensions():
    model = RuleJdBackend().build(JD_TEXT)
    keys = {c.key for c in model.competencies}
    for expected in ("customer_dev", "quotation", "python_dev", "llm_app", "agent_dev"):
        assert expected in keys, f"能力模型缺少 {expected}"
    hard = {h.key for h in model.hard_requirements}
    assert "education" in hard
    assert "years" in hard


def test_screening_decisions_match_expectations():
    model = RuleJdBackend().build(JD_TEXT)
    pipeline = _pipeline()
    for cid, text in RESUMES.items():
        decision = pipeline.screen(text, model, candidate_ref=cid)
        expected = EXPECTED[cid]["decision"]
        assert decision.decision == expected, (
            f"{cid}: 期望 {expected}，实际 {decision.decision}（分 {decision.score}，"
            f"理由 {decision.reasons}）"
        )


def test_hard_filter_rejects_short_experience():
    model = RuleJdBackend().build(JD_TEXT)
    decision = _pipeline().screen(RESUMES["R002"], model, candidate_ref="R002")
    assert decision.rejected_by_filter
    assert any("年限" in r for r in decision.reasons)


def test_hard_filter_rejects_education():
    model = RuleJdBackend().build(JD_TEXT)
    decision = _pipeline().screen(RESUMES["R003"], model, candidate_ref="R003")
    assert decision.rejected_by_filter
    assert any("学历" in r for r in decision.reasons)


def test_evidence_is_grounded_in_resume_text():
    model = RuleJdBackend().build(JD_TEXT)
    decision = _pipeline().screen(RESUMES["R004"], model, candidate_ref="R004")
    assert decision.match.evidences
    for ev in decision.match.evidences:
        assert ev.quote in RESUMES["R004"], f"证据不在原文中：{ev.quote}"


def test_masking_keeps_scores_identical():
    """R004 与 R005 只差敏感字段，屏蔽后得分必须完全一致。"""
    model = RuleJdBackend().build(JD_TEXT)
    pipeline = _pipeline()
    d4 = pipeline.screen(RESUMES["R004"], model, candidate_ref="R004")
    d5 = pipeline.screen(RESUMES["R005"], model, candidate_ref="R005")
    assert d4.score == d5.score
    assert d4.decision == d5.decision
    assert "marital" in d5.masked_fields


def test_weighted_score_penalises_missing_required():
    model = RuleJdBackend().build(JD_TEXT)
    from hragent.model import Competency

    model.competencies[0].required = True
    match = RuleMatchBackend().match("完全无关的内容", model, None)
    score, penalties = compute_weighted_score(model, match)
    assert penalties
    assert score < 50


def test_unmasked_screening_is_possible_but_not_default():
    model = RuleJdBackend().build(JD_TEXT)
    pipeline = ScreeningPipeline(
        RuleMatchBackend(),
        config=ScreeningConfig(mask_sensitive=False, pass_score=70.0, review_score=45.0),
    )
    decision = pipeline.screen(RESUMES["R005"], model, candidate_ref="R005")
    assert decision.masked_fields == []
