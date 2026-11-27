"""hr-agent 评测集。"""

from .fixtures import EXPECTED, EXPECTED_RANK, INTERVIEW_ANSWERS, JD_TEXT, RESUMES
from .metrics import (
    InterviewCaseResult,
    ScreeningCaseResult,
    aggregate_screening,
    evaluate_screening_case,
)

__all__ = [
    "EXPECTED",
    "EXPECTED_RANK",
    "INTERVIEW_ANSWERS",
    "InterviewCaseResult",
    "JD_TEXT",
    "RESUMES",
    "ScreeningCaseResult",
    "aggregate_screening",
    "evaluate_screening_case",
]
