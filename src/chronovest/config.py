"""Typed configuration objects shared across layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class WeightingMethod(str, Enum):
    MARKET_CAP = "market_cap"
    EQUAL = "equal"
    CUSTOM = "custom"
    FREE_FLOAT = "free_float"
    REVENUE = "revenue"


class RebalanceFrequency(str, Enum):
    NONE = "none"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class Market(str, Enum):
    """Which behaviour the app runs in. The base currency is always BRL.

    LOCAL         -- B3 / Brazilian assets, already in BRL, no FX conversion.
    INTERNATIONAL -- foreign assets converted into BRL, so a Brazilian investor
                     sees the result (including the FX effect) in reais.
    """

    LOCAL = "local"
    INTERNATIONAL = "international"


@dataclass
class BacktestConfig:
    """Everything required to run a single backtest.

    Strategy model (default): the investor never sells. Capital deployed at the
    start, plus every periodic contribution, buys the *currently balanced*
    basket of the sector at that moment (weights recomputed each buy). Existing
    positions are held; winners are left to run. Set ``allow_selling`` to enable
    classic sell-to-target rebalancing instead.

    Attributes:
        sector: Name of the sector universe to invest in.
        start / end: Simulation window.
        initial_capital: Cash deployed at ``start`` in ``base_currency``.
        base_currency: Reporting currency. Defaults to BRL.
        weighting: How each buy is split across active companies.
        reinvest_dividends: If True, dividends buy more of the paying stock
            (DRIP); otherwise they accumulate as idle cash inside the portfolio.
        transaction_cost_bps: Cost charged on traded notional, in basis points.
        contribution_amount: Recurring deposit (aporte) in ``base_currency``.
        contribution_frequency: Cadence of contributions / balanced buys.
        allow_selling: If True, sell-to-target rebalance at ``rebalance``.
        rebalance: Sell-rebalance cadence; only used when ``allow_selling``.
        benchmark: Optional equity index ticker (e.g. ``^BVSP``).
        custom_weights: Required when ``weighting`` is CUSTOM.
        market: Local (B3/BRL) or international (foreign -> BRL).
    """

    sector: str
    start: date
    end: date
    initial_capital: float = 10_000.0
    base_currency: str = "BRL"
    weighting: WeightingMethod = WeightingMethod.MARKET_CAP
    reinvest_dividends: bool = True
    transaction_cost_bps: float = 0.0
    contribution_amount: float = 0.0
    contribution_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    allow_selling: bool = False
    rebalance: RebalanceFrequency = RebalanceFrequency.NONE
    benchmark: str | None = None
    custom_weights: dict[str, float] = field(default_factory=dict)
    market: Market = Market.LOCAL

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.contribution_amount < 0:
            raise ValueError("contribution_amount cannot be negative")
        if self.weighting is WeightingMethod.CUSTOM and not self.custom_weights:
            raise ValueError("custom weighting requires custom_weights")
