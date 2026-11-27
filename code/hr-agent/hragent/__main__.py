"""hr-agent 命令行入口。

    python -m hragent eval            # 跑离线评测
    python -m hragent screen FILE     # 对一份简历做筛查（真实模型需 API key，规则后端无需）
    python -m hragent interview FILE  # 对一份简历跑模拟面试
    python -m hragent model JD.txt    # 打印 JD 的能力模型
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evals.run_eval import main as eval_main
from .extract import extract_text
from .interview import InterviewConfig, InterviewEngine, RuleInterviewerBackend
from .jd import RuleJdBackend
from .matching import RuleMatchBackend
from .report import render_interview_report, render_screening_report
from .screening import ScreeningConfig, ScreeningPipeline


def _screening_pipeline() -> ScreeningPipeline:
    return ScreeningPipeline(
        RuleMatchBackend(),
        config=ScreeningConfig(pass_score=70.0, review_score=45.0),
    )


def _cmd_screen(args: argparse.Namespace) -> int:
    doc = extract_text(args.file)
    if not doc.ok:
        print(f"抽取失败：{doc.error}")
        return 2
    jd_text = Path(args.jd).read_text(encoding="utf-8") if args.jd else _default_jd()
    model = RuleJdBackend().build(jd_text)
    decision = _screening_pipeline().screen(doc.text, model, candidate_ref=Path(args.file).name)
    if args.json:
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_screening_report(decision, model))
    return 0


def _cmd_interview(args: argparse.Namespace) -> int:
    doc = extract_text(args.file)
    if not doc.ok:
        print(f"抽取失败：{doc.error}")
        return 2
    jd_text = Path(args.jd).read_text(encoding="utf-8") if args.jd else _default_jd()
    model = RuleJdBackend().build(jd_text)
    engine = InterviewEngine(
        RuleInterviewerBackend(), config=InterviewConfig(max_questions=args.questions)
    )
    session = engine.start(model.role_title, model, doc.text)
    print(f"岗位：{model.role_title}｜题目数：{len(session.plan)}")
    for i, q in enumerate(session.plan, 1):
        print(f"  Q{i}[{q.dimension}] {q.question}")
    # 交互式回答；非交互环境用 --auto 直接用占位答案跑完
    answers: list[str] = []
    if args.auto:
        from .evals.fixtures import INTERVIEW_ANSWERS

        answers = [INTERVIEW_ANSWERS.get(q.dimension, "") for q in session.plan]
    else:
        for q in session.plan:
            try:
                answers.append(input(f"A[{q.dimension}]> ").strip())
            except (EOFError, KeyboardInterrupt):
                answers.append("")
                break
    for q, a in zip(session.plan, answers):
        while True:
            _, follow_up = engine.submit(session, a)
            if not follow_up:
                break
            print(f"  追问：{follow_up}")
            try:
                a = input("A[追问]> ").strip()
            except (EOFError, KeyboardInterrupt):
                a = ""
                break
    evaluation = engine.finish(session, model)
    print()
    print(render_interview_report(session, model))
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    from .jd import dump_model

    jd_text = Path(args.file).read_text(encoding="utf-8")
    print(dump_model(RuleJdBackend().build(jd_text)))
    return 0


def _default_jd() -> str:
    from .evals.fixtures import JD_TEXT

    return JD_TEXT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hragent", description="AI 招聘应用实验台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="跑离线评测集")
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("--report")
    p_eval.add_argument("--show-report", action="store_true")

    p_screen = sub.add_parser("screen", help="简历筛查")
    p_screen.add_argument("file")
    p_screen.add_argument("--jd", help="JD 文件路径")
    p_screen.add_argument("--json", action="store_true")

    p_iv = sub.add_parser("interview", help="模拟面试")
    p_iv.add_argument("file")
    p_iv.add_argument("--jd", help="JD 文件路径")
    p_iv.add_argument("--questions", type=int, default=6)
    p_iv.add_argument("--auto", action="store_true", help="用占位答案自动跑完")

    p_model = sub.add_parser("model", help="打印 JD 能力模型")
    p_model.add_argument("file")

    args = parser.parse_args(argv)
    if args.command == "eval":
        argv2 = []
        if args.json:
            argv2.append("--json")
        if args.report:
            argv2 += ["--report", str(args.report)]
        if args.show_report:
            argv2.append("--show-report")
        return eval_main(argv2)
    if args.command == "screen":
        return _cmd_screen(args)
    if args.command == "interview":
        return _cmd_interview(args)
    if args.command == "model":
        return _cmd_model(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
