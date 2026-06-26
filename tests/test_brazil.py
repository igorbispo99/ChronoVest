from datetime import date

import numpy as np
import pandas as pd
import pytest

from chronovest.analytics.brazil import (
    money_weighted_return,
    simulate_dca,
    xirr,
)
from chronovest.config import BacktestConfig, WeightingMethod
from chronovest.data.brazil_indicators import StaticIndicators
from chronovest.engine.portfolio import PortfolioEngine
from chronovest.universe.sector import Membership, Sector

from conftest import SyntheticProvider


def test_static_cdi_factor_grows():
    idx = pd.bdate_range("2020-01-01", periods=252)
    f = StaticIndicators(cdi_daily_rate=0.0004).cdi_factor(idx)
    assert f.iloc[0] == pytest.approx(1.0)
    assert f.iloc[-1] > 1.0


def test_simulate_dca_single_flow_buy_and_hold():
    idx = pd.bdate_range("2020-01-01", periods=10)
    price = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    flows = pd.Series(0.0, index=idx)
    flows.iloc[0] = 1000.0
    eq = simulate_dca(price, flows)
    assert eq.iloc[0] == pytest.approx(1000.0)
    assert eq.iloc[-1] == pytest.approx(2000.0)


def test_simulate_dca_accumulates_contributions():
    idx = pd.bdate_range("2020-01-01", periods=5)
    price = pd.Series(100.0, index=idx)
    flows = pd.Series([1000, 0, 1000, 0, 0], index=idx, dtype=float)
    eq = simulate_dca(price, flows)
    assert eq.iloc[-1] == pytest.approx(2000.0)


def test_xirr_simple_doubling_one_year():
    cf = pd.Series([-1000.0, 2000.0],
                   index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    assert xirr(cf) == pytest.approx(1.0, rel=1e-2)


def test_money_weighted_return_positive_for_growth():
    flows = pd.Series([10_000.0],
                      index=pd.to_datetime(["2020-01-01"]))
    r = money_weighted_return(flows, 15_000.0,
                              final_date=pd.Timestamp("2021-01-01"))
    assert r == pytest.approx(0.5, rel=1e-2)


def test_brazil_report_pct_of_cdi_above_100_when_beating_cdi():
    from chronovest.analytics.brazil import brazil_report
    sector = Sector("G", "custom", [Membership("AAA")], currency="BRL")
    provider = SyntheticProvider({"AAA": {"p0": 100.0, "rate": 0.002}})
    cfg = BacktestConfig(sector="G", start=date(2020, 1, 1), end=date(2021, 12, 31),
                         base_currency="BRL", weighting=WeightingMethod.EQUAL)
    res = PortfolioEngine(provider, sector).run(cfg)
    rep = brazil_report(res, StaticIndicators(cdi_daily_rate=0.00001))
    assert rep.nominal_total_return > rep.cdi_total_return
    assert rep.pct_of_cdi > 100
    assert rep.real_total_return < rep.nominal_total_return
