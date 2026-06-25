"""Weighting strategies.

A strategy maps a set of currently-active tickers (plus context such as market
caps on the rebalance date) to target weights that sum to 1. Adding a new
weighting method means adding one subclass; the engine is agnostic to which is
used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from chronovest.config import WeightingMethod


@dataclass
class WeightingContext:
    """Information available to a strategy at a rebalance date."""

    asof: date
    market_caps: pd.Series | None = None
    free_float: pd.Series | None = None
    revenue: pd.Series | None = None
    custom_weights: dict[str, float] | None = None


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        n = len(weights)
        return {t: 1.0 / n for t in weights} if n else {}
    return {t: (w / total if w > 0 else 0.0) for t, w in weights.items()}


class WeightingStrategy(ABC):
    method: WeightingMethod

    @abstractmethod
    def weights(self, tickers: list[str], ctx: WeightingContext) -> dict[str, float]:
        ...


class EqualWeighting(WeightingStrategy):
    method = WeightingMethod.EQUAL

    def weights(self, tickers, ctx):
        n = len(tickers)
        return {t: 1.0 / n for t in tickers} if n else {}


class _SeriesWeighting(WeightingStrategy):
    """Weight proportionally to a per-ticker series, with equal-weight fallback."""

    series_attr: str

    def weights(self, tickers, ctx):
        series = getattr(ctx, self.series_attr)
        if series is None:
            return EqualWeighting().weights(tickers, ctx)
        raw = {}
        for t in tickers:
            val = series.get(t) if hasattr(series, "get") else None
            raw[t] = float(val) if val is not None and pd.notna(val) and val > 0 else 0.0
        if all(v == 0 for v in raw.values()):
            return EqualWeighting().weights(tickers, ctx)
        return _normalize(raw)


class MarketCapWeighting(_SeriesWeighting):
    method = WeightingMethod.MARKET_CAP
    series_attr = "market_caps"


class FreeFloatWeighting(_SeriesWeighting):
    method = WeightingMethod.FREE_FLOAT
    series_attr = "free_float"


class RevenueWeighting(_SeriesWeighting):
    method = WeightingMethod.REVENUE
    series_attr = "revenue"


class CustomWeighting(WeightingStrategy):
    method = WeightingMethod.CUSTOM

    def weights(self, tickers, ctx):
        cw = ctx.custom_weights or {}
        raw = {t: float(cw.get(t, 0.0)) for t in tickers}
        if all(v == 0 for v in raw.values()):
            return EqualWeighting().weights(tickers, ctx)
        return _normalize(raw)


_REGISTRY: dict[WeightingMethod, type[WeightingStrategy]] = {
    WeightingMethod.EQUAL: EqualWeighting,
    WeightingMethod.MARKET_CAP: MarketCapWeighting,
    WeightingMethod.FREE_FLOAT: FreeFloatWeighting,
    WeightingMethod.REVENUE: RevenueWeighting,
    WeightingMethod.CUSTOM: CustomWeighting,
}


def build_weighting(method: WeightingMethod) -> WeightingStrategy:
    return _REGISTRY[method]()
