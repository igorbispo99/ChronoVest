from datetime import date
from pathlib import Path

import pytest

from chronovest.config import BacktestConfig, RebalanceFrequency, WeightingMethod
from chronovest.data.currency import StaticConverter, infer_currency
from chronovest.engine.portfolio import PortfolioEngine
from chronovest.universe.sector import load_sector

from conftest import SyntheticProvider

SECTOR_DIR = Path(__file__).resolve().parents[1] / "data" / "sectors"
SECTORS = sorted(SECTOR_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", SECTORS, ids=lambda p: p.stem)
def test_sector_loads(path):
    sector = load_sector(path)
    assert sector.members
    for m in sector.members:
        ccy = sector.currency_of(m.ticker) or infer_currency(m.ticker)
        assert ccy


@pytest.mark.parametrize("path", SECTORS, ids=lambda p: p.stem)
def test_sector_runs_end_to_end_offline(path):
    sector = load_sector(path)
    specs = {m.ticker: {"p0": 100.0, "rate": 0.0003, "shares": 1_000_000}
             for m in sector.members}
    provider = SyntheticProvider(specs)
    fx = StaticConverter({("USD", "BRL"): 5.0, ("CHF", "BRL"): 6.0,
                          ("JPY", "BRL"): 0.03, ("CHF", "USD"): 1.1,
                          ("JPY", "USD"): 0.007})
    cfg = BacktestConfig(
        sector=sector.name, start=date(2018, 1, 1), end=date(2023, 12, 31),
        initial_capital=10_000, base_currency=sector.currency or "USD",
        weighting=WeightingMethod.MARKET_CAP, rebalance=RebalanceFrequency.YEARLY,
    )
    res = PortfolioEngine(provider, sector, fx=fx).run(cfg)
    assert res.equity.iloc[-1] > 0
    assert len(res.target_weights) >= 1
