from datetime import date

import pytest

from chronovest.config import BacktestConfig, Market, RebalanceFrequency, WeightingMethod
from chronovest.data.brazil_indicators import StaticIndicators
from chronovest.data.currency import StaticConverter
from chronovest.runner import run_backtest
from chronovest.universe.sector import Membership, Sector

from conftest import SyntheticProvider

SPAN = (date(2018, 1, 1), date(2022, 12, 31))


def _provider(extra=None):
    specs = {
        "AAA": {"p0": 100.0, "rate": 0.0004, "shares": 2_000_000},
        "BBB": {"p0": 50.0, "rate": 0.0003, "shares": 1_000_000},
        "^BVSP": {"p0": 100_000.0, "rate": 0.0002},
        "^GSPC": {"p0": 3000.0, "rate": 0.0003},
    }
    specs.update(extra or {})
    return SyntheticProvider(specs)


def test_local_mode_has_cdi_and_ibovespa():
    sector = Sector("Local", "custom",
                    [Membership("AAA"), Membership("BBB")],
                    currency="BRL", market="local")
    cfg = BacktestConfig(sector="Local", start=SPAN[0], end=SPAN[1],
                         base_currency="BRL", market=Market.LOCAL,
                         weighting=WeightingMethod.MARKET_CAP,
                         contribution_amount=500,
                         contribution_frequency=RebalanceFrequency.MONTHLY,
                         benchmark="^BVSP")
    full = run_backtest(_provider(), sector, cfg, fx=None,
                        indicators=StaticIndicators())
    assert "CDI" in full.benchmarks
    assert "Ibovespa" in full.benchmarks
    assert full.result.equity.iloc[-1] > 0
    assert full.report.pct_of_cdi == full.report.pct_of_cdi  # not NaN


def test_international_mode_adds_native_index_in_brl():
    sector = Sector("Intl", "custom",
                    [Membership("AAA", currency="USD")],
                    currency="USD", market="international")
    cfg = BacktestConfig(sector="Intl", start=SPAN[0], end=SPAN[1],
                         base_currency="BRL", market=Market.INTERNATIONAL,
                         weighting=WeightingMethod.MARKET_CAP,
                         contribution_amount=500,
                         contribution_frequency=RebalanceFrequency.MONTHLY,
                         benchmark="^BVSP")
    fx = StaticConverter({("USD", "BRL"): 5.0})
    full = run_backtest(_provider(), sector, cfg, fx=fx,
                        indicators=StaticIndicators(),
                        native_indices=[("S&P 500", "^GSPC")])
    assert {"CDI", "Ibovespa", "S&P 500"} <= set(full.benchmarks)
    assert full.result.equity.iloc[-1] > 0
