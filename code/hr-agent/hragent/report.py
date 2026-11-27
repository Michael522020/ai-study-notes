"""报告生成：把结构化结论渲染成人能读的 Markdown。

两条要求：
1. **结论 + 理由 + 证据** 三件套齐全，不能只给分数；
2. 涉及候选人个人的内容要标注**这是 AI 初筛**，且注明屏蔽了哪些字段 ——
   报告本身也要能经得起合规检查。
"""

from __future__ import annotations

from .model import CompetencyModel, InterviewSession, ScreeningDecision

DECISION_LABEL = {
    "pass": "通过（建议进入人工面试）",
    "review": "待人工复核",
    "reject": "不通过",
}


def render_screening_report(
    decision: ScreeningDecision, model: CompetencyModel, *, title: str = "简历筛查报告"
) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- 岗位：{model.role_title}")
    lines.append(f"- 候选人编号：{decision.candidate_ref or '未指定'}")
    lines.append(f"- 结论：**{DECISION_LABEL.get(decision.decision, decision.decision)}**")
    lines.append(f"- 综合分：{decision.score:.1f} / 100")
    if decision.masked_fields:
        lines.append(f"- 已屏蔽敏感字段：{'、'.join(decision.masked_fields)}")
    lines += ["", "> 本报告由 AI 辅助生成，仅用于初筛排序，最终录用判断由人工做出。", ""]

    lines += ["## 硬性门槛"]
    if not decision.filters:
        lines.append("- （无硬性门槛）")
    for f in decision.filters:
        mark = "✅" if f.passed else ("❌" if f.required else "⚠️")
        tag = "" if f.required else "（非必须）"
        lines.append(f"- {mark} {f.label}：期望 {f.expected}，实际 {f.actual}{tag}")
    lines.append("")

    lines += ["## 能力匹配"]
    if not decision.match.scores:
        lines.append("- （无匹配结果）")
    else:
        lines.append("| 能力维度 | 得分 | 证据 |")
        lines.append("|---|---|---|")
        evidence_by_dim = {e.dimension: e for e in decision.match.evidences}
        for comp in model.competencies:
            score = decision.match.scores.get(comp.key, 0.0)
            ev = evidence_by_dim.get(comp.key)
            quote = (ev.quote[:60] + "…") if ev and ev.quote else "—"
            lines.append(f"| {comp.name} | {score * 5:.1f}/5 | {quote} |")
    lines.append("")

    if decision.match.missing:
        names = [
            model.competency(k).name if model.competency(k) else k
            for k in decision.match.missing
        ]
        lines += ["## 缺口维度", "- " + "、".join(names), ""]

    if decision.reasons:
        lines += ["## 结论依据"]
        for r in decision.reasons:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


def render_interview_report(
    session: InterviewSession,
    model: CompetencyModel,
    *,
    title: str = "模拟面试评估报告",
) -> str:
    evaluation = session.evaluation
    lines = [f"# {title}", ""]
    lines.append(f"- 岗位：{session.role_title}")
    lines.append(f"- 题目数：{len(session.plan)}，回答数：{len(session.answers)}")
    if evaluation:
        lines.append(f"- 综合分：{evaluation.overall:.1f} / 100")
        lines.append(f"- 建议：{evaluation.recommendation}")
    lines += ["", "> 本报告由 AI 辅助生成，评分用于自我训练与横向参考，不等同于人工面试结论。", ""]

    if evaluation:
        lines += ["## 逐维度评分", "| 能力维度 | 水平 | 评价 | 候选人原话 |", "|---|---|---|---|"]
        name_by_key = {c.key: c.name for c in model.competencies}
        for d in evaluation.dimensions:
            quote = (d.evidence[:50] + "…") if d.evidence else "—"
            lines.append(
                f"| {name_by_key.get(d.dimension, d.dimension)} | {d.level}/5 | {d.comment} | {quote} |"
            )
        lines.append("")

    if session.plan:
        lines += ["## 问答记录"]
        for i, a in enumerate(session.answers, 1):
            lines.append(f"### Q{i}（{a.question.dimension}）")
            lines.append(f"**问**：{a.question.question}")
            lines.append(f"**答**：{a.text}")
            lines.append("")

    if evaluation:
        if evaluation.strengths:
            lines += ["## 亮点", "- " + "、".join(evaluation.strengths), ""]
        if evaluation.gaps:
            lines += ["## 待补强", "- " + "、".join(evaluation.gaps), ""]
        if evaluation.comments:
            lines += ["## 备注", "- " + "、".join(evaluation.comments), ""]

    return "\n".join(lines)
