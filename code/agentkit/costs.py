"""模型定价与成本核算。

口径说明（写进笔记，避免「成本」这个数字被误读）：
- 单价单位是 **美元 / 百万 token**（MTok）；
- cache read = 输入价的 0.1×，cache write（缓存写入）= 输入价的 1.25×；
- 缓存命中只对**前缀**生效，system / tools 顺序为 `tools → system → messages`，
  任何一字节变化都会让其后全部失效；
- 不同模型单价不同，同一份 prompt 在 Opus 5.5 与 Sonnet 5.5 上相差 2×，
  Haiku 5.5 在 100K prompt 以内与 Opus 5.5 相差 40×。

价格数据取自 Anthropic 官方模型表（2026-11 缓存）。价格会变，落地前请回官方页面核对。
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Usage


@dataclass(frozen=True)
class ModelPrice:
    model: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float
    context_window: int
    #: 该价格是官方明示还是按官方惯例推导（缓存读写按 0.1× / 1.25× 推导）
    cache_pricing_source: str = "derived"

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens / 1_000_000 * self.input_per_mtok
            + usage.output_tokens / 1_000_000 * self.output_per_mtok
            + usage.cache_read_input_tokens / 1_000_000 * self.cache_read_per_mtok
            + usage.cache_creation_input_tokens / 1_000_000 * self.cache_write_per_mtok
        )


PRICING: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(
        "claude-opus-5-5", 4.00, 20.00, 0.20, 5.00, 1_000_000, "official"
    ),
    "claude-opus-5": ModelPrice(
        "claude-opus-5", 5.00, 25.00, 0.25, 6.25, 1_000_000, "official"
    ),
    "claude-sonnet-5-5": ModelPrice(
        "claude-sonnet-5-5", 2.00, 10.00, 0.10, 2.50, 1_000_000, "official"
    ),
    "claude-haiku-5-5": ModelPrice(
        "claude-haiku-5-5", 0.10, 0.50, 0.01, 0.125, 200_000, "official"
    ),
    "claude-fable-5-2": ModelPrice(
        "claude-fable-5-2", 8.00, 40.00, 0.20, 10.00, 1_000_000, "official"
    ),
}


def price_of(model: str) -> ModelPrice:
    try:
        return PRICING[model]
    except KeyError as exc:  # pragma: no cover - 配置错误路径
        raise KeyError(
            f"模型 {model!r} 没有登记价格，无法核算成本。已登记：{sorted(PRICING)}"
        ) from exc


def estimate_cost(model: str, usage: Usage) -> float:
    """按 usage 估算美元成本。未知模型直接报错，不做「按最贵算」的保守估计。"""
    return price_of(model).cost(usage)


def format_cost(usd: float) -> str:
    if usd < 0.01:
        return f"${usd:.6f}"
    return f"${usd:.4f}"
