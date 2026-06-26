"""International mode: foreign sector converted to BRL, vs CDI/IPCA/Ibovespa + native index.

    python examples/run_international.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from chronovest.config import BacktestConfig, Market, RebalanceFrequency, WeightingMethod
from chronovest.data.brazil_indicators import BcbIndicators
from chronovest.data.cache import CachedProvider
from chronovest.data.currency import YFinanceConverter
from chronovest.data.yfinance_provider import YFinanceProvider
from chronovest.runner import run_backtest
from chronovest.universe.sector import load_sector
from chronovest.viz.charts import export_excel

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sector = load_sector(ROOT / "data" / "sectors" / "intl_technology.yaml")
    provider = CachedProvider(YFinanceProvider(), cache_dir=ROOT / ".cache")
    config = BacktestConfig(
        sector=sector.name, start=date(2016, 1, 1), end=date(2024, 12, 31),
        initial_capital=10_000, base_currency="BRL", market=Market.INTERNATIONAL,
        weighting=WeightingMethod.MARKET_CAP,
        contribution_amount=1_000, contribution_frequency=RebalanceFrequency.MONTHLY,
        reinvest_dividends=True, transaction_cost_bps=10, benchmark="^BVSP",
    )
    full = run_backtest(provider, sector, config, fx=YFinanceConverter(),
                        indicators=BcbIndicators())
    r, rep = full.result, full.report
    print(f"Final value : R$ {r.equity.iloc[-1]:,.0f}  (invested R$ {r.total_invested:,.0f})")
    print(f"Nominal CAGR: {rep.nominal_cagr:6.1%}   Real CAGR (IPCA): {rep.real_cagr:6.1%}")
    print(f"% do CDI    : {rep.pct_of_cdi:6.0f}%   Sharpe vs CDI: {rep.sharpe_vs_cdi:5.2f}")
    print(f"XIRR        : {rep.money_weighted_return:6.1%}   Max drawdown: {rep.max_drawdown:6.1%}")
    print("Benchmarks  :", ", ".join(f"{k}=R${v.iloc[-1]:,.0f}" for k, v in full.benchmarks.items()))
    for w in r.warnings:
        print("WARN:", w)
    export_excel(r, ROOT / "international_backtest.xlsx")


if __name__ == "__main__":
    main()
