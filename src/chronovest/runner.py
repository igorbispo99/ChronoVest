"""High-level orchestration tying the two modes together.

The base currency is always BRL. The mode decides:
  * LOCAL         -- B3 assets, no FX conversion, benchmarks: CDI, Ibovespa
                     (+ IPCA for real returns).
  * INTERNATIONAL -- foreign assets converted to BRL (the FX effect is part of
                     the return), benchmarks: CDI, Ibovespa, and the native
                     foreign index converted to BRL.

`run_backtest` returns the raw engine result, comparable benchmark curves (built
with the portfolio's own contribution schedule), and the Brazilian report.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chronovest.analytics.brazil import BrazilReport, brazil_report, build_benchmarks
from chronovest.config import BacktestConfig, Market
from chronovest.data.base import DataProvider
from chronovest.data.brazil_indicators import BrazilIndicators
from chronovest.data.currency import CurrencyConverter, infer_currency
from chronovest.engine.portfolio import BacktestResult, PortfolioEngine
from chronovest.universe.sector import Sector

IBOVESPA = ("Ibovespa", "^BVSP")
NATIVE_INDICES = {
    "International Technology": [("Nasdaq", "^IXIC"), ("S&P 500", "^GSPC")],
    "International AI": [("Nasdaq", "^IXIC")],
    "International Banking": [("S&P 500", "^GSPC")],
    "Global Watchmaking": [("S&P 500", "^GSPC")],
}


@dataclass
class FullResult:
    result: BacktestResult
    benchmarks: dict[str, pd.Series]
    report: BrazilReport


def run_backtest(
    provider: DataProvider,
    sector: Sector,
    config: BacktestConfig,
    fx: CurrencyConverter | None,
    indicators: BrazilIndicators,
    native_indices: list[tuple[str, str]] | None = None,
) -> FullResult:
    result = PortfolioEngine(provider, sector, fx=fx).run(config)

    price_curves: dict[str, pd.Series] = {}
    _add_index_curve(price_curves, provider, fx, config, *IBOVESPA)
    if config.market is Market.INTERNATIONAL:
        indices = native_indices
        if indices is None:
            indices = NATIVE_INDICES.get(sector.name, [("S&P 500", "^GSPC")])
        for name, ticker in indices:
            _add_index_curve(price_curves, provider, fx, config, name, ticker)

    cost_rate = config.transaction_cost_bps / 1e4
    benchmarks = build_benchmarks(result, indicators, price_curves, cost_rate)
    report = brazil_report(result, indicators)
    return FullResult(result=result, benchmarks=benchmarks, report=report)


def _add_index_curve(curves, provider, fx, config, name, ticker) -> None:
    raw = provider.get_prices([ticker], config.start, config.end).get(ticker)
    if raw is None or raw.empty:
        return
    series = raw["adj_close"].dropna()
    if series.empty:
        return
    ccy = infer_currency(ticker, default=config.base_currency)
    if fx is not None and ccy != config.base_currency:
        rate = fx.rate(ccy, config.base_currency, series.index)
        series = series * rate.reindex(series.index).ffill().bfill()
    curves[name] = series
