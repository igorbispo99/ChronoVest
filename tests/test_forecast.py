from datetime import date

import numpy as np
import pandas as pd
import pytest

from chronovest.config import RebalanceFrequency
from chronovest.forecast import (
    BlockBootstrapForecaster,
    GbmForecaster,
    forecast_portfolio,
)


def _returns(mean=0.0005, vol=0.01, n=750, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(rng.normal(mean, vol, n), index=idx)


@pytest.mark.parametrize("fc", [GbmForecaster(), BlockBootstrapForecaster()])
def test_bounds_are_ordered(fc):
    f = forecast_portfolio(_returns(), start_value=10_000, last_date=date(2021, 1, 1),
                           horizon_years=3, forecaster=fc, n_paths=500, seed=42)
    assert (f.lower <= f.median + 1e-6).all()
    assert (f.median <= f.upper + 1e-6).all()


def test_seed_is_reproducible():
    kw = dict(returns=_returns(), start_value=10_000, last_date=date(2021, 1, 1),
              horizon_years=2, n_paths=400, seed=7)
    a = forecast_portfolio(forecaster=GbmForecaster(), **kw)
    b = forecast_portfolio(forecaster=GbmForecaster(), **kw)
    assert a.median.iloc[-1] == pytest.approx(b.median.iloc[-1])


def test_positive_drift_median_grows():
    f = forecast_portfolio(_returns(mean=0.001), start_value=10_000,
                           last_date=date(2021, 1, 1), horizon_years=3,
                           forecaster=GbmForecaster(), n_paths=800, seed=3)
    assert f.median.iloc[-1] > 10_000


def test_longer_horizon_widens_interval():
    base = dict(returns=_returns(), start_value=10_000, last_date=date(2021, 1, 1),
                forecaster=GbmForecaster(), n_paths=800, seed=5)
    short = forecast_portfolio(horizon_years=1, **base)
    long = forecast_portfolio(horizon_years=5, **base)
    sw = short.upper.iloc[-1] - short.lower.iloc[-1]
    lw = long.upper.iloc[-1] - long.lower.iloc[-1]
    assert lw > sw


def test_contributions_raise_terminal_median():
    base = dict(returns=_returns(), start_value=10_000, last_date=date(2021, 1, 1),
                horizon_years=4, forecaster=GbmForecaster(), n_paths=800, seed=9)
    without = forecast_portfolio(contribution_amount=0, **base)
    with_ = forecast_portfolio(contribution_amount=500,
                               contribution_frequency=RebalanceFrequency.MONTHLY, **base)
    assert with_.terminal["median"] > without.terminal["median"]
    assert with_.terminal["total_invested"] > without.terminal["total_invested"]


def test_terminal_probabilities_in_range():
    f = forecast_portfolio(_returns(), start_value=10_000, last_date=date(2021, 1, 1),
                           horizon_years=2, n_paths=600, seed=11)
    assert 0.0 <= f.terminal["prob_above_invested"] <= 1.0
    assert 0.0 <= f.terminal["prob_above_start"] <= 1.0
