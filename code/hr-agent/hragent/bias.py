"""敏感字段屏蔽与偏差审计。

招聘场景里最容易被忽略、也最容易出事的一环：
**简历里带着年龄、性别、婚育、籍贯、照片**。模型再"客观"，只要这些字段进了上下文，
就可能被用来推断，从而形成事实上的歧视。

三条防线（缺一不可）：
1. **抽取前屏蔽** —— 评分链路根本看不到这些字段（主防线）；
2. **审计验证** —— 拿屏蔽前后的文本各跑一次，断言分数完全一致（回归防线）；
3. **留痕** —— 记录屏蔽了哪些字段，供合规审计与申诉复核。

法律背景见 06 章「合规与伦理」一节：我国《就业促进法》《个人信息保护法》
对就业阶段的自动化决策有明确约束；欧盟那边是 GDPR 第 22 条与 AI Act。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .rules import MUST_MASK_FIELDS, SENSITIVE_PATTERNS

MASK_TOKEN = "【已屏蔽】"


@dataclass
class MaskResult:
    text: str
    masked_fields: list[str] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_masked(self) -> bool:
        return bool(self.masked_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "masked_fields": list(self.masked_fields),
            "hits": list(self.hits),
        }


def mask_sensitive(
    text: str, *, fields: tuple[str, ...] = MUST_MASK_FIELDS
) -> MaskResult:
    """把敏感字段替换成占位符。

    注意实现细节：按**字段逐条**做替换，而不是一个大正则一把梭。
    逐条的好处是能精确记账「屏蔽了什么」，且新增字段不用改主逻辑。
    """
    result = MaskResult(text=text)
    masked_fields: list[str] = []
    hits: list[dict[str, Any]] = []

    for field_name, pattern in SENSITIVE_PATTERNS:
        if field_name not in fields:
            continue
        regex = re.compile(pattern)

        def _replace(match: re.Match[str]) -> str:
            hits.append({"field": field_name, "span": match.group(0)[:40]})
            return MASK_TOKEN

        new_text, count = regex.subn(_replace, result.text)
        if count:
            result.text = new_text
            if field_name not in masked_fields:
                masked_fields.append(field_name)

    result.masked_fields = masked_fields
    result.hits = hits
    return result


def audit_masking(
    original_text: str,
    masked_text: str,
    scorer: Callable[[str], float],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """偏差审计：屏蔽前后分数必须一致。

    `scorer` 是「给一段简历文本打一个分」的函数。生产里传打分流水线的入口。
    一旦返回的差值超过容差，说明有代码路径绕过了屏蔽 —— 这是必须报警的回归。
    """
    before = scorer(original_text)
    after = scorer(masked_text)
    delta = abs(before - after)
    return {
        "score_before": round(before, 6),
        "score_after": round(after, 6),
        "delta": round(delta, 9),
        "pass": delta <= tolerance,
    }


def mask_summary(mask: MaskResult) -> str:
    if not mask.masked_fields:
        return "未发现需要屏蔽的敏感字段"
    return "已屏蔽字段：" + "、".join(mask.masked_fields)
