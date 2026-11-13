"""JSON Schema 子集：严格化 + 校验。

为什么自己写而不是装 jsonschema：
1. 本仓库的代码要在目标机器上「pip install 就能跑」，少一个依赖少一份心障；
2. 我们只需要一个**够用且严格**的子集 —— 工具入参校验、结构化输出校验、评测断言共用一套。

两件关键事：
- `strict_schema()` 把 schema 补成 API 要求的严格形态：
  每个 object 必须有 `additionalProperties: false` 和显式 `required`。
  Anthropic 的 strict 工具模式不满足这个形状时会直接拒绝入参。
- `validate()` 返回**错误列表**而不是抛异常：工具入参非法时，
  我们要把错误信息喂回模型让它自修，而不是让整个 Agent 崩掉。
"""

from __future__ import annotations

from typing import Any

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """递归补全严格 schema：additionalProperties=false + required 覆盖全部属性。"""
    out = dict(schema)
    props = out.get("properties")
    if out.get("type") == "object" or isinstance(props, dict):
        out.setdefault("type", "object")
        if isinstance(props, dict):
            out["properties"] = {
                k: strict_schema(v) if isinstance(v, dict) else v for k, v in props.items()
            }
            out["required"] = list(props.keys())
            out["additionalProperties"] = False
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = strict_schema(out["items"])
    return out


def _type_ok(value: Any, expected: str) -> bool:
    types = _TYPE_MAP.get(expected)
    if types is None:
        return True
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in ("number", "integer") and isinstance(value, bool):
        return False  # Python 里 bool 是 int 的子类，JSON 语义上不算数字
    return isinstance(value, types)


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """返回错误信息列表，空列表表示通过。"""
    errors: list[str] = []

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: 期望常量 {schema['const']!r}，实际 {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: 值 {instance!r} 不在枚举 {schema['enum']} 内")

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _type_ok(instance, expected_type):
            errors.append(f"{path}: 期望类型 {expected_type}，实际 {type(instance).__name__}")
            return errors
    elif isinstance(expected_type, list):
        if not any(_type_ok(instance, t) for t in expected_type):
            errors.append(f"{path}: 期望类型 {expected_type}，实际 {type(instance).__name__}")
            return errors

    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = schema[key]
            passed = [not validate(instance, b, path) for b in branches]
            if key == "anyOf" and not any(passed):
                errors.append(f"{path}: 不满足 anyOf 的任何一个分支")
            if key == "oneOf" and sum(passed) != 1:
                errors.append(f"{path}: 恰好一个分支应成立，实际 {sum(passed)} 个成立")

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key}: 缺少必填字段")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(props))
            if extra:
                errors.append(f"{path}: 出现未声明字段 {extra}")
        for key, sub_schema in props.items():
            if key in instance and isinstance(sub_schema, dict):
                errors.extend(validate(instance[key], sub_schema, f"{path}.{key}"))

    if isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(instance):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: 至少需要 {schema['minItems']} 个元素")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: 最多允许 {schema['maxItems']} 个元素")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: 字符串长度至少 {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: 字符串长度最多 {schema['maxLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: 不能小于 {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: 不能大于 {schema['maximum']}")

    return errors
