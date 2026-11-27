"""JD → 能力模型。

两条后端：
- `RuleJdBackend`：词典 + 正则，确定性、可离线、可作为校验基线；
- `LlmJdBackend`：让模型读 JD 抽结构化能力项，输出走 JSON Schema 强约束。

生产建议：**模型抽、规则校**。模型抽出来的每一条都要过词典比对，
对不上的（幻觉出来的技能）直接丢弃并记一条告警 —— 宁可少抽，不要抽错。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from agentkit.provider import Provider

from .model import Competency, CompetencyModel, HardRequirement
from .rules import (
    COMPETENCY_LEXICON,
    find_competency_hits,
    find_soft_skills,
    extract_salary_range,
)


class JdBackend(Protocol):
    name: str

    def build(self, jd_text: str) -> CompetencyModel: ...


# ---------------------------------------------------------------------------
# 规则后端
# ---------------------------------------------------------------------------


def _extract_role_title(jd_text: str) -> str:
    for line in jd_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(k in line for k in ("岗位职责", "职位描述", "岗位", "职责")):
            continue
        return line[:40]
    return "未命名岗位"


def _extract_education_requirement(jd_text: str) -> HardRequirement | None:
    import re

    m = re.search(r"(博士|硕士|本科|大专|专科|高中)[及]?以上", jd_text)
    if m:
        from .rules import EDUCATION_LEVELS

        name = m.group(1)
        level = next((lv for n, lv in EDUCATION_LEVELS if n == name), 0)
        return HardRequirement(
            key="education",
            label="学历",
            operator=">=",
            value=level,
            required=True,
            note=f"{name}及以上",
        )
    return None


def _extract_years_requirement(jd_text: str) -> HardRequirement | None:
    import re

    # 容忍「N 年以上<任意修饰>经验」的写法，例如「3 年以上外贸或跨境电商相关经验」
    m = re.search(r"(\d+)\s*年[^，。；\n]{0,12}?经验", jd_text)
    if not m:
        m = re.search(r"(\d+)\s*[-~到至]\s*(\d+)\s*年(?:经验|以上)", jd_text)
    if m:
        return HardRequirement(
            key="years",
            label="工作年限",
            operator=">=",
            value=float(m.group(1)),
            required=True,
        )
    return None


def _extract_city_requirement(jd_text: str) -> HardRequirement | None:
    from .rules import extract_city

    city = extract_city(jd_text)
    if city and "远程" not in jd_text:
        return HardRequirement(
            key="city", label="工作城市", operator="eq", value=city, required=False
        )
    return None


def _extract_salary_requirement(jd_text: str) -> HardRequirement | None:
    rng = extract_salary_range(jd_text)
    if rng:
        return HardRequirement(
            key="salary_k",
            label="薪资区间(K)",
            operator="<=",
            value=rng[0],
            required=False,
            note=f"{rng[0]:g}-{rng[1]:g}K",
        )
    return None


class RuleJdBackend:
    """词典驱动的确定性抽取。"""

    name = "rule"

    def build(self, jd_text: str) -> CompetencyModel:
        model = CompetencyModel(
            role_title=_extract_role_title(jd_text),
            backend=self.name,
            soft_skills=find_soft_skills(jd_text),
        )
        for req in (
            _extract_education_requirement(jd_text),
            _extract_years_requirement(jd_text),
            _extract_city_requirement(jd_text),
            _extract_salary_requirement(jd_text),
        ):
            if req is not None:
                model.hard_requirements.append(req)

        hits = find_competency_hits(jd_text)
        for key, matched in hits.items():
            name, keywords, desc, weight, level = COMPETENCY_LEXICON[key]
            model.competencies.append(
                Competency(
                    key=key,
                    name=name,
                    description=desc,
                    keywords=list(keywords),
                    weight=weight,
                    min_level=level,
                )
            )
        return model


# ---------------------------------------------------------------------------
# 模型后端
# ---------------------------------------------------------------------------

JD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role_title": {"type": "string", "description": "岗位名称"},
        "hard_requirements": {
            "type": "array",
            "description": "硬性门槛",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "education | years | city | salary_k | 其他"},
                    "label": {"type": "string", "description": "中文标签"},
                    "operator": {"type": "string", "description": ">= | <= | eq | in"},
                    "value": {"type": "string", "description": "门槛值，统一用字符串"},
                    "required": {"type": "boolean", "description": "不满足是否直接出局"},
                },
                "required": ["key", "label", "operator", "value", "required"],
            },
        },
        "competencies": {
            "type": "array",
            "description": "能力维度",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "能力 key，从给定词表中选"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "weight": {"type": "number", "description": "权重 0.5~1.5"},
                    "min_level": {"type": "integer", "description": "期望水平 1~5"},
                },
                "required": ["key", "name", "keywords", "weight", "min_level"],
            },
        },
        "soft_skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["role_title", "hard_requirements", "competencies", "soft_skills"],
}

JD_SYSTEM_PROMPT = """你是招聘 JD 结构化分析器。把一份 JD 拆成可计算的筛选条件。

规则：
1. 硬性门槛只放真正一票否决的：学历最低档、工作年限下限、必须到岗城市、薪资上限。
   可放宽的写 required=false。
2. 能力维度必须从给定词表里选 key；不要发明新 key。weight 反映对该岗位的重要性（0.5~1.5）。
3. min_level 是期望水平（1 了解 / 3 熟练 / 5 精通），按 JD 描述的深度判断。
4. 只输出结构化结果，不要解释。"""


class LlmJdBackend:
    """模型抽取 + 词表校验。模型抽错 key 的条目会被丢弃并记入 dropped。"""

    name = "llm"

    def __init__(self, provider: Provider, model: str = "claude-opus-5-5", effort: str = "medium"):
        self.provider = provider
        self.model = model
        self.effort = effort
        self.dropped: list[dict[str, Any]] = []

    def build(self, jd_text: str) -> CompetencyModel:
        lexicon_keys = list(COMPETENCY_LEXICON)
        prompt = (
            f"可用能力 key（只能用这些）：{', '.join(lexicon_keys)}\n\n"
            f"JD 原文：\n{jd_text}"
        )
        resp = self.provider.generate(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            system=JD_SYSTEM_PROMPT,
            output_schema=JD_SCHEMA,
            effort=self.effort,
        )
        data = resp.parsed or {}
        model = CompetencyModel(
            role_title=str(data.get("role_title") or "未命名岗位"),
            backend=self.name,
            soft_skills=[str(s) for s in data.get("soft_skills", [])],
        )
        for raw in data.get("hard_requirements", []):
            model.hard_requirements.append(
                HardRequirement(
                    key=str(raw.get("key", "")),
                    label=str(raw.get("label", "")),
                    operator=str(raw.get("operator", "eq")),
                    value=raw.get("value", ""),
                    required=bool(raw.get("required", True)),
                )
            )
        for raw in data.get("competencies", []):
            key = str(raw.get("key", ""))
            if key not in COMPETENCY_LEXICON:
                # 幻觉出来的 key：丢掉，但记账
                self.dropped.append({"reason": "unknown_key", "payload": raw})
                continue
            lexicon_name, lexicon_keywords, desc, default_weight, default_level = COMPETENCY_LEXICON[key]
            model.competencies.append(
                Competency(
                    key=key,
                    name=str(raw.get("name") or lexicon_name),
                    description=str(raw.get("description") or desc),
                    keywords=[str(k) for k in raw.get("keywords", [])] or list(lexicon_keywords),
                    weight=float(raw.get("weight", default_weight) or default_weight),
                    min_level=int(raw.get("min_level", default_level) or default_level),
                )
            )
        if not model.competencies:
            # 模型没抽出能力项时退回规则后端，保证下游有输入
            fallback = RuleJdBackend().build(jd_text)
            fallback.role_title = model.role_title or fallback.role_title
            self.dropped.append({"reason": "empty_competencies", "payload": None})
            return fallback
        return model


def build_model(
    jd_text: str,
    *,
    backend: str = "rule",
    provider: Provider | None = None,
    model: str = "claude-opus-5-5",
) -> CompetencyModel:
    if backend == "llm":
        if provider is None:
            raise ValueError("backend='llm' 需要提供 provider")
        return LlmJdBackend(provider, model=model).build(jd_text)
    return RuleJdBackend().build(jd_text)


def dump_model(model: CompetencyModel) -> str:
    """人类可读的能力模型，调试与报告都用它。"""
    lines = [f"# {model.role_title}（抽取方式：{model.backend}）", "", "## 硬性门槛"]
    for h in model.hard_requirements:
        flag = "必须" if h.required else "优先"
        lines.append(f"- [{flag}] {h.describe()}" + (f"（{h.note}）" if h.note else ""))
    if not model.hard_requirements:
        lines.append("- （无）")
    lines += ["", "## 能力维度"]
    for c in model.competencies:
        lines.append(
            f"- `{c.key}` {c.name}｜权重 {c.weight}｜期望水平 {c.min_level}/5"
            f"｜关键词：{'、'.join(c.keywords[:6])}"
        )
    if model.soft_skills:
        lines += ["", "## 软性要求", "- " + "、".join(model.soft_skills)]
    return "\n".join(lines)
