"""Performance and risk metrics computed from an equity curve.

All functions take a pandas Series indexed by date (the portfolio value over
time) and are independent of how that curve was produced, so they apply equally
to the strategy and to a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.dropna().pct_change().dropna()


def _years(equity: pd.Series) -> float:
    idx = equity.dropna().index
    return max((idx[-1] - idx[0]).days / 365.25, 1e-9)


def total_return(equity: pd.Series) -> float:
    e = equity.dropna()
    return float(e.iloc[-1] / e.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    e = equity.dropna()
    return float((e.iloc[-1] / e.iloc[0]) ** (1.0 / _years(e)) - 1.0)


def volatility(equity: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _daily_returns(equity)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else 0.0


def sharpe_ratio(
    equity: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    r = _daily_returns(equity)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    e = equity.dropna()
    return e / e.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def annual_returns(equity: pd.Series) -> pd.Series:
    e = equity.dropna()
    yearly = e.resample("YE").last()
    first = e.iloc[0]
    prev = pd.concat([pd.Series([first], index=[e.index[0]]), yearly]).shift(1)
    out = (yearly / prev.reindex(yearly.index) - 1.0).dropna()
    out.index = out.index.year
    return out


def rolling_returns(equity: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    e = equity.dropna()
    return (e / e.shift(window) - 1.0).dropna()


@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    annual_returns: pd.Series

    def as_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
        }


def performance_report(equity: pd.Series, risk_free: float = 0.0) -> PerformanceReport:
    return PerformanceReport(
        total_return=total_return(equity),
        cagr=cagr(equity),
        volatility=volatility(equity),
        sharpe=sharpe_ratio(equity, risk_free),
        max_drawdown=max_drawdown(equity),
        annual_returns=annual_returns(equity),
    )
