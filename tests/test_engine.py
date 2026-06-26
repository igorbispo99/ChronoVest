from datetime import date

import pytest

from chronovest.config import BacktestConfig, RebalanceFrequency, WeightingMethod
from chronovest.engine.portfolio import PortfolioEngine
from chronovest.universe.sector import Membership, Sector

from conftest import SyntheticProvider


def _run(sector, provider, span, **overrides):
    start, end = span
    cfg = BacktestConfig(sector=sector.name, start=start, end=end,
                         initial_capital=10_000, base_currency="BRL", **overrides)
    return PortfolioEngine(provider, sector).run(cfg)


def test_buy_and_hold_grows_and_conserves_initial(two_stock_sector, span):
    sector, provider = two_stock_sector
    res = _run(sector, provider, span, weighting=WeightingMethod.EQUAL)
    assert res.equity.iloc[0] == pytest.approx(10_000, rel=1e-6)
    assert res.equity.iloc[-1] > res.equity.iloc[0]


def test_no_sell_strategy_never_sells(two_stock_sector, span):
    sector, provider = two_stock_sector
    res = _run(sector, provider, span, weighting=WeightingMethod.MARKET_CAP,
               contribution_amount=500, contribution_frequency=RebalanceFrequency.MONTHLY)
    if not res.trades.empty:
        assert not res.trades["action"].str.contains("sell").any()


def test_contributions_increase_total_invested(two_stock_sector, span):
    sector, provider = two_stock_sector
    res = _run(sector, provider, span, weighting=WeightingMethod.EQUAL,
               contribution_amount=1000, contribution_frequency=RebalanceFrequency.MONTHLY)
    assert res.total_contributed > 0
    assert res.total_invested == pytest.approx(10_000 + res.total_contributed)


def test_contributions_sum_to_total_pnl(two_stock_sector, span):
    sector, provider = two_stock_sector
    res = _run(sector, provider, span, weighting=WeightingMethod.MARKET_CAP,
               contribution_amount=1000, contribution_frequency=RebalanceFrequency.MONTHLY,
               transaction_cost_bps=0)
    gain = res.equity.iloc[-1] - res.total_invested
    assert res.contributions.sum() == pytest.approx(gain, rel=1e-6)


def test_twr_ignores_contributions_flat_market(span):
    sector = Sector("F", "custom", [Membership("AAA")], currency="BRL")
    provider = SyntheticProvider({"AAA": {"p0": 100.0, "rate": 0.0}})
    res = _run(sector, provider, span, weighting=WeightingMethod.EQUAL,
               contribution_amount=1000, contribution_frequency=RebalanceFrequency.MONTHLY)
    assert res.equity.iloc[-1] > res.equity.iloc[0]
    assert res.twr.iloc[-1] == pytest.approx(10_000, rel=1e-3)


def test_market_cap_weighting_favours_larger_company(two_stock_sector, span):
    sector, provider = two_stock_sector
    res = _run(sector, provider, span, weighting=WeightingMethod.MARKET_CAP)
    w = res.target_weights.iloc[0]
    assert w["AAA"] > w["BBB"]


def test_transaction_costs_reduce_value(two_stock_sector, span):
    sector, provider = two_stock_sector
    free = _run(sector, provider, span, contribution_amount=1000,
                contribution_frequency=RebalanceFrequency.MONTHLY, transaction_cost_bps=0)
    costly = _run(sector, provider, span, contribution_amount=1000,
                  contribution_frequency=RebalanceFrequency.MONTHLY, transaction_cost_bps=50)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_dividends_reinvested_beats_cash(span):
    sector = Sector("D", "custom", [Membership("AAA")], currency="BRL")
    specs = {"AAA": {"p0": 100.0, "rate": 0.0005,
                     "dividends": [{"date": "2020-06-15", "amount": 5.0},
                                   {"date": "2021-06-15", "amount": 5.0}]}}
    provider = SyntheticProvider(specs)
    on = _run(sector, provider, span, weighting=WeightingMethod.EQUAL,
              reinvest_dividends=True)
    off = _run(sector, provider, span, weighting=WeightingMethod.EQUAL,
               reinvest_dividends=False)
    assert on.equity.iloc[-1] > off.equity.iloc[-1]


def test_delisting_is_liquidated(span):
    sector = Sector("X", "custom",
                    [Membership("AAA"), Membership("BBB", end=date(2020, 6, 30))],
                    currency="BRL")
    specs = {
        "AAA": {"p0": 100.0, "rate": 0.0003, "shares": 1_000_000},
        "BBB": {"p0": 100.0, "rate": 0.0003, "shares": 1_000_000, "last": "2020-06-30"},
    }
    provider = SyntheticProvider(specs)
    res = _run(sector, provider, span, weighting=WeightingMethod.EQUAL)
    assert res.holdings_value.iloc[-1]["BBB"] == pytest.approx(0.0)
    assert res.equity.iloc[-1] > 0


def test_allow_selling_rebalances(two_stock_sector, span):
    sector, provider = two_stock_sector
    res = _run(sector, provider, span, weighting=WeightingMethod.MARKET_CAP,
               allow_selling=True, rebalance=RebalanceFrequency.MONTHLY)
    assert res.trades["action"].str.contains("rebalance").any()


def test_missing_ticker_is_skipped_with_warning(span):
    sector = Sector("M", "custom",
                    [Membership("AAA"), Membership("DEAD")], currency="BRL")
    provider = SyntheticProvider({"AAA": {"p0": 100.0, "rate": 0.0003}})
    res = _run(sector, provider, span, weighting=WeightingMethod.EQUAL)
    assert res.equity.iloc[-1] > 0
    assert any("DEAD" in w for w in res.warnings)
    assert "DEAD" not in res.holdings_value.columns
