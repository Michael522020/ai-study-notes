"""hr-agent：简历筛查与模拟面试的可运行实现。

模块分工：
- `extract`  文本抽取（txt / md / docx / pdf）
- `rules`    词典与正则（能力图谱、敏感字段、简历字段）
- `jd`       JD → 能力模型（规则 / 模型两种后端）
- `fields`   简历 → 结构化字段
- `bias`     敏感字段屏蔽与偏差审计
- `matching` 能力匹配（规则 / 模型两种后端，证据必须可回指原文）
- `screening` 筛查流水线（屏蔽 → 硬过滤 → 打分 → 结论）
- `interview` 模拟面试引擎（出题 → 追问 → 评分）
- `report`   Markdown 报告
"""

from .bias import MaskResult, audit_masking, mask_sensitive
from .fields import ResumeFields, extract_fields
from .interview import (
    InterviewConfig,
    InterviewEngine,
    InterviewerBackend,
    LlmInterviewerBackend,
    RuleInterviewerBackend,
)
from .jd import JdBackend, LlmJdBackend, RuleJdBackend, build_model, dump_model
from .matching import LlmMatchBackend, MatchBackend, RuleMatchBackend, compute_weighted_score
from .model import (
    Answer,
    Competency,
    CompetencyModel,
    DimensionScore,
    Evidence,
    FilterOutcome,
    HardRequirement,
    InterviewEvaluation,
    InterviewSession,
    MatchResult,
    Question,
    ScreeningDecision,
)
from .report import render_interview_report, render_screening_report
from .screening import ScreeningConfig, ScreeningPipeline

__all__ = [
    "Answer",
    "Competency",
    "CompetencyModel",
    "DimensionScore",
    "Evidence",
    "FilterOutcome",
    "HardRequirement",
    "InterviewConfig",
    "InterviewEngine",
    "InterviewEvaluation",
    "InterviewSession",
    "InterviewerBackend",
    "JdBackend",
    "LlmInterviewerBackend",
    "LlmJdBackend",
    "LlmMatchBackend",
    "MaskResult",
    "MatchBackend",
    "MatchResult",
    "Question",
    "ResumeFields",
    "RuleInterviewerBackend",
    "RuleJdBackend",
    "RuleMatchBackend",
    "ScreeningConfig",
    "ScreeningDecision",
    "ScreeningPipeline",
    "audit_masking",
    "build_model",
    "compute_weighted_score",
    "dump_model",
    "extract_fields",
    "mask_sensitive",
    "render_interview_report",
    "render_screening_report",
]
