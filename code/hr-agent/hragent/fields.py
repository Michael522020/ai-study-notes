"""简历 → 结构化字段。

规则抽取的边界要写清楚：它只能拿到**文本里明写的东西**。
「6年经验」写清楚了才数得到；简历没写年限，就是 0，不是「未知」——
两者在门槛判定上必须区别对待，所以字段值用 0 表示「没写」，
由 `FilterOutcome` 单独记录「期望 vs 实际」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rules import (
    extract_city,
    extract_education,
    extract_salary_expectation_k,
    extract_years,
    find_competency_hits,
    find_soft_skills,
)


@dataclass
class ResumeFields:
    education_name: str = "未知"
    education_level: int = 0
    years: float = 0.0
    city: str | None = None
    salary_k: float | None = None
    skill_hits: dict[str, list[str]] = field(default_factory=dict)
    soft_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "education_name": self.education_name,
            "education_level": self.education_level,
            "years": self.years,
            "city": self.city,
            "salary_k": self.salary_k,
            "skill_hits": dict(self.skill_hits),
            "soft_skills": list(self.soft_skills),
        }


def extract_fields(text: str) -> ResumeFields:
    edu_name, edu_level = extract_education(text)
    return ResumeFields(
        education_name=edu_name,
        education_level=edu_level,
        years=extract_years(text),
        city=extract_city(text),
        salary_k=extract_salary_expectation_k(text),
        skill_hits=find_competency_hits(text),
        soft_skills=find_soft_skills(text),
    )
