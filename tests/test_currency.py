from datetime import date

import numpy as np
import pandas as pd
import pytest

from chronovest.config import BacktestConfig, RebalanceFrequency, WeightingMethod
from chronovest.data.currency import (
    IdentityConverter,
    StaticConverter,
    infer_currency,
)
from chronovest.engine.portfolio import PortfolioEngine
from chronovest.universe.sector import Membership, Sector

from conftest import SyntheticProvider


def test_infer_currency_from_suffix():
    assert infer_currency("ITUB4.SA") == "BRL"
    assert infer_currency("UHR.SW") == "CHF"
    assert infer_currency("7762.T") == "JPY"
    assert infer_currency("MOV") == "USD"
    assert infer_currency("^BVSP") == "BRL"


def test_identity_converter_is_one():
    idx = pd.bdate_range("2020-01-01", periods=5)
    assert (IdentityConverter().rate("USD", "BRL", idx) == 1.0).all()


def test_static_converter_inverse():
    idx = pd.bdate_range("2020-01-01", periods=3)
    fx = StaticConverter({("USD", "BRL"): 5.0})
    assert float(fx.rate("USD", "BRL", idx).iloc[0]) == pytest.approx(5.0)
    assert float(fx.rate("BRL", "USD", idx).iloc[0]) == pytest.approx(0.2)
    assert float(fx.rate("BRL", "BRL", idx).iloc[0]) == pytest.approx(1.0)


def test_fx_appreciation_boosts_base_currency_return():
    sector = Sector("X", "custom", [Membership("AAA", currency="USD")], currency="USD")
    span = (date(2020, 1, 1), date(2020, 12, 31))
    idx = pd.bdate_range(span[0], span[1])
    rate = pd.Series(np.linspace(5.0, 10.0, len(idx)), index=idx)
    fx = StaticConverter({("USD", "BRL"): rate})
    provider = SyntheticProvider({"AAA": {"p0": 100.0, "rate": 0.0}})

    common = dict(sector="X", start=span[0], end=span[1], initial_capital=10_000,
                  weighting=WeightingMethod.EQUAL, rebalance=RebalanceFrequency.NONE)
    usd = PortfolioEngine(provider, sector, fx=fx).run(
        BacktestConfig(base_currency="USD", **common))
    brl = PortfolioEngine(provider, sector, fx=fx).run(
        BacktestConfig(base_currency="BRL", **common))

    assert usd.equity.iloc[-1] == pytest.approx(10_000, rel=1e-6)
    assert brl.equity.iloc[-1] == pytest.approx(20_000, rel=1e-3)


def test_mixed_currency_weights_use_converted_caps():
    sector = Sector(
        "M", "custom",
        [Membership("USDBIG", currency="USD"), Membership("BRLSMALL", currency="BRL")],
    )
    provider = SyntheticProvider({
        "USDBIG": {"p0": 100.0, "rate": 0.0, "shares": 1_000_000},
        "BRLSMALL": {"p0": 100.0, "rate": 0.0, "shares": 1_000_000},
    })
    span = (date(2020, 1, 1), date(2020, 6, 30))
    fx = StaticConverter({("USD", "BRL"): 5.0})
    cfg = BacktestConfig(sector="M", start=span[0], end=span[1],
                         base_currency="BRL", weighting=WeightingMethod.MARKET_CAP,
                         rebalance=RebalanceFrequency.NONE)
    res = PortfolioEngine(provider, sector, fx=fx).run(cfg)
    w = res.target_weights.iloc[0]
    assert w["USDBIG"] == pytest.approx(5 / 6, rel=1e-6)
    assert w["BRLSMALL"] == pytest.approx(1 / 6, rel=1e-6)


def test_missing_fx_warns_for_mixed_currency():
    sector = Sector("W", "custom", [Membership("AAA", currency="CHF")], currency="CHF")
    provider = SyntheticProvider({"AAA": {"p0": 100.0, "rate": 0.0}})
    cfg = BacktestConfig(sector="W", start=date(2020, 1, 1), end=date(2020, 6, 30),
                         base_currency="USD", weighting=WeightingMethod.EQUAL,
                         rebalance=RebalanceFrequency.NONE)
    res = PortfolioEngine(provider, sector, fx=None).run(cfg)
    assert any("fx" in w.lower() for w in res.warnings)
