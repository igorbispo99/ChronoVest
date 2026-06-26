"""Analytics layer: performance, risk and Brazil-framed metrics."""

from chronovest.analytics.metrics import (
    PerformanceReport,
    annual_returns,
    cagr,
    drawdown_series,
    max_drawdown,
    performance_report,
    rolling_returns,
    sharpe_ratio,
    total_return,
    volatility,
)
from chronovest.analytics.brazil import (
    BrazilReport,
    brazil_report,
    build_benchmarks,
    money_weighted_return,
    real_series,
    sharpe_vs_cdi,
    simulate_dca,
    xirr,
)

__all__ = [
    "PerformanceReport",
    "annual_returns",
    "cagr",
    "drawdown_series",
    "max_drawdown",
    "performance_report",
    "rolling_returns",
    "sharpe_ratio",
    "total_return",
    "volatility",
    "BrazilReport",
    "brazil_report",
    "build_benchmarks",
    "money_weighted_return",
    "real_series",
    "sharpe_vs_cdi",
    "simulate_dca",
    "xirr",
]
