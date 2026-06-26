"""Project a portfolio forward in time with probabilistic bounds.

Given the historical return distribution of a sector (or the portfolio's
time-weighted return), simulate many future paths, optionally continuing the
investor's contribution schedule, and summarise them as median / lower / upper
bands. The bounds are percentiles across simulated paths, not a guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronovest.config import RebalanceFrequency
from chronovest.forecast.base import ForecastResult, Forecaster
from chronovest.forecast.montecarlo import BlockBootstrapForecaster

_FREQ = {
    RebalanceFrequency.MONTHLY: "MS",
    RebalanceFrequency.QUARTERLY: "QS",
    RebalanceFrequency.YEARLY: "YS",
}


def future_business_days(last_date, horizon_years: float) -> pd.DatetimeIndex:
    last = pd.Timestamp(last_date)
    n = max(1, int(round(horizon_years * 252)))
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)


def _contribution_vector(index: pd.DatetimeIndex, amount: float,
                         freq: RebalanceFrequency) -> np.ndarray:
    vec = np.zeros(len(index))
    if amount <= 0 or freq is RebalanceFrequency.NONE:
        return vec
    for b in pd.date_range(index[0], index[-1], freq=_FREQ[freq]):
        pos = index.searchsorted(b, side="left")
        if pos < len(index):
            vec[pos] += amount
    return vec


def forecast_portfolio(
    returns: pd.Series,
    start_value: float,
    last_date,
    horizon_years: float = 5.0,
    forecaster: Forecaster | None = None,
    contribution_amount: float = 0.0,
    contribution_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
    confidence: float = 0.90,
    n_paths: int = 2000,
    seed: int | None = None,
    history: pd.Series | None = None,
) -> ForecastResult:
    forecaster = forecaster or BlockBootstrapForecaster()
    rng = np.random.default_rng(seed)
    index = future_business_days(last_date, horizon_years)
    horizon = len(index)

    sim = forecaster.fit(returns).simulate(horizon, n_paths, rng)
    contrib = _contribution_vector(index, contribution_amount, contribution_frequency)

    values = np.empty((n_paths, horizon))
    prev = np.full(n_paths, float(start_value))
    for t in range(horizon):
        prev = prev * (1.0 + sim[:, t]) + contrib[t]
        values[:, t] = prev

    lo_pct = (1.0 - confidence) / 2.0 * 100.0
    hi_pct = 100.0 - lo_pct
    lower = pd.Series(np.percentile(values, lo_pct, axis=0), index=index)
    median = pd.Series(np.percentile(values, 50.0, axis=0), index=index)
    upper = pd.Series(np.percentile(values, hi_pct, axis=0), index=index)

    total_invested = float(start_value + contrib.sum())
    terminal = {
        "median": float(median.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "upper": float(upper.iloc[-1]),
        "total_invested": total_invested,
        "prob_above_invested": float(np.mean(values[:, -1] > total_invested)),
        "prob_above_start": float(np.mean(values[:, -1] > start_value)),
    }
    return ForecastResult(
        history=history if history is not None else pd.Series(dtype=float),
        median=median, lower=lower, upper=upper,
        confidence=confidence, horizon_years=horizon_years,
        terminal=terminal, paths=values,
    )


def forecast_from_result(
    result,
    horizon_years: float = 5.0,
    forecaster: Forecaster | None = None,
    confidence: float = 0.90,
    n_paths: int = 2000,
    seed: int | None = None,
    continue_contributions: bool = True,
) -> ForecastResult:
    """Forecast a BacktestResult forward using its time-weighted return history."""
    returns = result.twr.dropna().pct_change().dropna()
    cfg = result.config
    return forecast_portfolio(
        returns=returns,
        start_value=float(result.equity.iloc[-1]),
        last_date=result.equity.index[-1],
        horizon_years=horizon_years,
        forecaster=forecaster,
        contribution_amount=cfg.contribution_amount if continue_contributions else 0.0,
        contribution_frequency=cfg.contribution_frequency,
        confidence=confidence,
        n_paths=n_paths,
        seed=seed,
        history=result.equity,
    )
