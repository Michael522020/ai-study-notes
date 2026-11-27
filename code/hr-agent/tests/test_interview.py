"""模拟面试引擎测试。"""

from __future__ import annotations

from hragent.evals.fixtures import INTERVIEW_ANSWERS, JD_TEXT
from hragent.interview import (
    InterviewConfig,
    InterviewEngine,
    RuleInterviewerBackend,
)
from hragent.jd import RuleJdBackend


def _engine() -> InterviewEngine:
    return InterviewEngine(
        RuleInterviewerBackend(),
        config=InterviewConfig(max_questions=5, allow_follow_up=True),
    )


def test_plan_binds_questions_to_model_dimensions():
    model = RuleJdBackend().build(JD_TEXT)
    session = _engine().start(model.role_title, model)
    assert session.plan
    assert len(session.plan) <= 5
    for q in session.plan:
        assert model.competency(q.dimension) is not None
        assert q.question.strip()
        assert q.expected_points


def test_questions_cover_top_dimensions():
    """出题应优先问权重高、且标为必备的维度。"""
    model = RuleJdBackend().build(JD_TEXT)
    session = _engine().start(model.role_title, model)
    asked = {q.dimension for q in session.plan}
    for key in ("customer_dev", "python_dev", "llm_app"):
        assert key in asked


def test_short_answer_triggers_follow_up():
    model = RuleJdBackend().build(JD_TEXT)
    engine = _engine()
    session = engine.start(model.role_title, model)
    question = engine.current_question(session)
    _, follow_up = engine.submit(session, "还行")
    assert follow_up, "过短的回答应触发追问"


def test_full_interview_produces_evaluation_with_evidence():
    model = RuleJdBackend().build(JD_TEXT)
    engine = _engine()
    session = engine.start(model.role_title, model)
    while True:
        q = engine.current_question(session)
        if q is None:
            break
        answer = INTERVIEW_ANSWERS.get(q.dimension, "简短")
        engine.submit(session, answer)
    evaluation = engine.finish(session, model)
    assert evaluation.dimensions
    assert evaluation.recommendation
    assert 0 <= evaluation.overall <= 100
    for d in evaluation.dimensions:
        assert d.evidence, f"维度 {d.dimension} 缺原话引用"


def test_session_finishes_cleanly():
    model = RuleJdBackend().build(JD_TEXT)
    engine = _engine()
    session = engine.start(model.role_title, model)
    planned = len(session.plan)
    while True:
        q = engine.current_question(session)
        if q is None:
            break
        answer = INTERVIEW_ANSWERS.get(q.dimension, "回答")
        _, follow_up = engine.submit(session, answer)
        if follow_up:
            engine.submit(session, answer)  # 回答追问后本题才推进
    assert session.current is None
    assert len(session.answers) >= planned
