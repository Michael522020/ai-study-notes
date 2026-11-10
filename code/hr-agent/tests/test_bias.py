"""敏感字段屏蔽与偏差审计测试。"""

from __future__ import annotations

from hragent.bias import MASK_TOKEN, audit_masking, mask_sensitive

SAMPLE = """张伟 | 性别：男 | 33岁 | 上海 | 本科 | 主导客户开发
婚育状况：已婚，育有一子
籍贯：安徽合肥
民族：汉族
政治面貌：中共党员
身份证号：310101199001011234
负责报价核算与成本核算
"""


def test_mask_removes_sensitive_fields():
    result = mask_sensitive(SAMPLE)
    for field in ("age", "gender", "marital", "ethnicity", "native_place", "political", "id_card"):
        assert field in result.masked_fields, f"{field} 未被识别"
    assert "33岁" not in result.text
    assert "性别：男" not in result.text
    assert "已婚" not in result.text
    assert "安徽合肥" not in result.text
    assert "310101199001011234" not in result.text


def test_mask_keeps_business_content():
    result = mask_sensitive(SAMPLE)
    assert "主导客户开发" in result.text
    assert "报价核算" in result.text
    assert MASK_TOKEN in result.text


def test_mask_without_sensitive_content():
    result = mask_sensitive("负责报价核算，客户开发，Python 脚本")
    assert not result.masked_fields
    assert result.text == "负责报价核算，客户开发，Python 脚本"


def test_audit_reports_zero_delta():
    """屏蔽前后打分必须一致 —— 这条不过就是回归。"""
    def scorer(text: str) -> float:
        # 只看业务内容：屏蔽不该影响分数
        return float("客户开发" in text) + float("报价核算" in text)

    masked = mask_sensitive(SAMPLE)
    report = audit_masking(SAMPLE, masked.text, scorer)
    assert report["pass"]
    assert report["delta"] == 0.0


def test_audit_detects_leak():
    """如果打分函数用到了敏感字段，审计必须报错。"""
    def leaky_scorer(text: str) -> float:
        return 1.0 if "男" in text else 0.0

    masked = mask_sensitive(SAMPLE)
    report = audit_masking(SAMPLE, masked.text, leaky_scorer)
    assert not report["pass"]
