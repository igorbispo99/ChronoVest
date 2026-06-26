import numpy as np
import pandas as pd

from chronovest.analytics import metrics


def _curve(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_total_return():
    e = _curve([100, 110, 121])
    assert metrics.total_return(e) == pytest.approx(0.21)


def test_cagr_doubling_over_one_year():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
    e = pd.Series([100.0, 200.0], index=idx)
    assert metrics.cagr(e) == pytest.approx(1.0, rel=1e-2)


def test_max_drawdown():
    e = _curve([100, 120, 60, 90])
    assert metrics.max_drawdown(e) == pytest.approx(-0.5)


def test_volatility_constant_growth_is_zero():
    e = _curve([100 * 1.01 ** i for i in range(50)])
    assert metrics.volatility(e) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_positive_for_uptrend():
    e = _curve([100 * 1.001 ** i for i in range(100)])
    assert metrics.sharpe_ratio(e) > 0


def test_annual_returns_index_is_year():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    e = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    ar = metrics.annual_returns(e)
    assert list(ar.index) == [2020, 2021]


import pytest  # noqa: E402
