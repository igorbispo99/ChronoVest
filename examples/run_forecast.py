"""Forecast a sector forward with probabilistic bounds.

    python examples/run_forecast.py

Projecao por Monte Carlo a partir da distribuicao historica de retornos.
Bandas sao percentis dos cenarios simulados, nao uma garantia. Nao e
recomendacao de investimento.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from chronovest.config import BacktestConfig, Market, RebalanceFrequency, WeightingMethod
from chronovest.data.brazil_indicators import BcbIndicators
from chronovest.data.cache import CachedProvider
from chronovest.data.yfinance_provider import YFinanceProvider
from chronovest.forecast import BlockBootstrapForecaster, forecast_from_result
from chronovest.runner import run_backtest
from chronovest.universe.sector import load_sector

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sector = load_sector(ROOT / "data" / "sectors" / "brazil_banks.yaml")
    provider = CachedProvider(YFinanceProvider(), cache_dir=ROOT / ".cache")
    config = BacktestConfig(
        sector=sector.name, start=date(2016, 1, 1), end=date(2024, 12, 31),
        initial_capital=10_000, base_currency="BRL", market=Market.LOCAL,
        weighting=WeightingMethod.MARKET_CAP,
        contribution_amount=1_000, contribution_frequency=RebalanceFrequency.MONTHLY,
        benchmark="^BVSP",
    )
    full = run_backtest(provider, sector, config, fx=None, indicators=BcbIndicators())

    fc = forecast_from_result(full.result, horizon_years=5,
                              forecaster=BlockBootstrapForecaster(),
                              confidence=0.90, n_paths=3000, seed=42)
    t = fc.terminal
    print(f"Current value : R$ {full.result.equity.iloc[-1]:,.0f}")
    print(f"5y median     : R$ {t['median']:,.0f}")
    print(f"5y {fc.band_label()}: R$ {t['lower']:,.0f}  ..  R$ {t['upper']:,.0f}")
    print(f"P(end > invested R$ {t['total_invested']:,.0f}): {t['prob_above_invested']:.0%}")


if __name__ == "__main__":
    main()
