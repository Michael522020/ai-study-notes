"""hr-agent 评测执行器。

    python -m hragent.evals.run_eval              # 跑全部评测
    python -m hragent.evals.run_eval --json       # JSON 输出
    python -m hragent.evals.run_eval --report out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agentkit.tracing import MemoryTracer

from ..interview import InterviewConfig, InterviewEngine, RuleInterviewerBackend
from ..jd import RuleJdBackend
from ..matching import RuleMatchBackend
from ..report import render_interview_report, render_screening_report
from ..screening import ScreeningConfig, ScreeningPipeline
from .fixtures import EXPECTED, EXPECTED_RANK, INTERVIEW_ANSWERS, JD_TEXT, RESUMES
from .metrics import (
    DECISION_RANK,
    InterviewCaseResult,
    aggregate_screening,
    evaluate_screening_case,
)


def run_screening() -> tuple[list[Any], dict[str, Any], list[Any]]:
    """跑一遍筛查，返回 (逐个结论, 汇总, 模型)。"""
    model = RuleJdBackend().build(JD_TEXT)
    pipeline = ScreeningPipeline(
        RuleMatchBackend(),
        config=ScreeningConfig(pass_score=70.0, review_score=45.0),
        tracer=MemoryTracer(),
    )
    decisions = []
    for cid, text in RESUMES.items():
        decision = pipeline.screen(text, model, candidate_ref=cid)
        decisions.append(decision)

    results = [
        evaluate_screening_case(cid, EXPECTED[cid], decision, RESUMES[cid])
        for cid, decision in zip(RESUMES, decisions)
    ]

    # 排序：先按结论档位，再按分数
    actual_rank = [
        d.candidate_ref
        for d in sorted(
            decisions,
            key=lambda d: (DECISION_RANK.get(d.decision, 3), -d.score, d.candidate_ref),
        )
    ]

    # 偏差审计：R005（多敏感字段）与 R004（对照）得分必须一致
    score_by_id = {d.candidate_ref: d.score for d in decisions}
    bias_delta = score_by_id.get("R005", 0.0) - score_by_id.get("R004", 0.0)

    summary = aggregate_screening(
        results, expected_rank=EXPECTED_RANK, actual_rank=actual_rank, bias_delta=bias_delta
    )
    return decisions, summary, model


def run_interview(model: Any, candidate: str = "R001") -> tuple[InterviewCaseResult, Any]:
    """对一位候选人跑一轮完整模拟面试。"""
    engine = InterviewEngine(
        RuleInterviewerBackend(),
        config=InterviewConfig(max_questions=6, allow_follow_up=True),
    )
    session = engine.start(model.role_title, model, RESUMES[candidate])
    result = InterviewCaseResult()

    result.questions = len(session.plan)
    result.dimensions_total = len(model.competencies)
    covered = {q.dimension for q in session.plan}
    result.dimensions_covered = len(covered)

    for q in session.plan:
        if model.competency(q.dimension) is None:
            result.all_questions_bound = False
            result.issues.append(f"题目维度 {q.dimension} 不在能力模型内")

    while True:
        question = engine.current_question(session)
        if question is None:
            break
        answer = INTERVIEW_ANSWERS.get(question.dimension, "")
        _, follow_up = engine.submit(session, answer)
        if follow_up:
            result.followups_triggered += 1
            # 追问轮：把同一个维度的补充回答提交一次
            engine.submit(session, answer)

    evaluation = engine.finish(session, model)
    result.answers = len(session.answers)
    result.recommendation = evaluation.recommendation
    for d in evaluation.dimensions:
        if not d.evidence:
            result.all_evidence_present = False
            result.issues.append(f"维度 {d.dimension} 的评分没有引用原话")
    if not result.recommendation:
        result.issues.append("缺少总体建议")
    return result, session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hr-agent 离线评测")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--show-report", action="store_true", help="打印一份筛查报告样例")
    args = parser.parse_args(argv)

    decisions, summary, model = run_screening()
    interview_result, session = run_interview(model)

    payload = {
        "screening": summary,
        "cases": [
            evaluate_screening_case(cid, EXPECTED[cid], d, RESUMES[cid]).to_dict()
            for cid, d in zip(RESUMES, decisions)
        ],
        "interview": interview_result.to_dict(),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("== 简历筛查 ==")
        for case in payload["cases"]:
            mark = "PASS" if case["decision_ok"] and case["hard_filter_ok"] else "FAIL"
            print(
                f"  [{mark}] {case['candidate']}  期望={case['expected_decision']:<7}"
                f"实际={case['actual_decision']:<7} 分={case['score']:<6}"
                f"屏蔽字段={'、'.join(case['masked_fields']) or '无'}"
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\n== 模拟面试 ==")
        print(json.dumps(interview_result.to_dict(), ensure_ascii=False, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报表已写入 {args.report}")

    if args.show_report:
        print("\n== 筛查报告样例 ==")
        print(render_screening_report(decisions[0], model))
        print("\n== 面试报告样例 ==")
        print(render_interview_report(session, model))

    ok = (
        summary["hard_filter_accuracy"] == 1.0
        and summary["evidence_grounding_rate"] == 1.0
        and summary["bias_invariance_pass"]
        and not interview_result.issues
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
