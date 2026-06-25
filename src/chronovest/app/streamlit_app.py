"""Streamlit front-end for ChronoVest.

Two modes, both reported in BRL with Brazilian benchmarks (CDI, IPCA, Ibovespa):
  * Local (B3)     -- Brazilian assets, no currency conversion.
  * International  -- foreign assets converted to BRL; the native index is also
                      shown (converted to BRL) for reference.

Run with:
    streamlit run src/chronovest/app/streamlit_app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from chronovest.config import (
    BacktestConfig,
    Market,
    RebalanceFrequency,
    WeightingMethod,
)
from chronovest.data.brazil_indicators import BcbIndicators
from chronovest.data.cache import CachedProvider
from chronovest.data.currency import YFinanceConverter
from chronovest.data.yfinance_provider import YFinanceProvider
from chronovest.forecast import (
    BlockBootstrapForecaster,
    GbmForecaster,
    forecast_from_result,
)
from chronovest.runner import run_backtest
from chronovest.universe.sector import load_sector
from chronovest.viz import (
    allocation_area,
    forecast_chart,
    benchmark_comparison_chart,
    contribution_bar,
    dividend_bar,
    drawdown_chart,
    invested_vs_value_chart,
    real_vs_nominal_chart,
)

ROOT = Path(__file__).resolve().parents[3]
SECTOR_DIR = ROOT / "data" / "sectors"

st.set_page_config(page_title="ChronoVest", layout="wide")
st.title("ChronoVest")
st.caption("Backtesting setorial - sempre em BRL, comparado a CDI, IPCA e Ibovespa")


def _sectors_for(market: Market):
    out = []
    for p in sorted(SECTOR_DIR.glob("*.yaml")):
        try:
            s = load_sector(p)
        except Exception:
            continue
        if (s.market or "local") == market.value:
            out.append((p, s.name))
    return out


@st.cache_data(show_spinner=True)
def _run(sector_path: str, cfg_kwargs: dict):
    sector = load_sector(sector_path)
    cfg_kwargs = dict(cfg_kwargs)
    cfg_kwargs["sector"] = sector.name
    config = BacktestConfig(**cfg_kwargs)
    provider = CachedProvider(YFinanceProvider(), cache_dir=ROOT / ".cache")
    fx = YFinanceConverter() if config.market is Market.INTERNATIONAL else None
    full = run_backtest(provider, sector, config, fx, BcbIndicators())
    return full


with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Mode", list(Market), format_func=lambda m: (
        "Local (B3)" if m is Market.LOCAL else "International"))
    sector_options = _sectors_for(mode)
    if not sector_options:
        st.error("No sectors for this mode")
        st.stop()
    sector_path = st.selectbox("Sector", sector_options, format_func=lambda o: o[1])[0]

    col1, col2 = st.columns(2)
    start = col1.date_input("Start", date(2015, 1, 1))
    end = col2.date_input("End", date(2024, 12, 31))
    capital = st.number_input("Initial capital (BRL)", 1000, 100_000_000, 10_000,
                              step=1000)
    st.subheader("Monthly contribution (aporte)")
    contribution = st.number_input("Amount (BRL)", 0, 1_000_000, 500, step=100)
    contrib_freq = st.selectbox("Frequency", [RebalanceFrequency.MONTHLY,
                                              RebalanceFrequency.QUARTERLY,
                                              RebalanceFrequency.YEARLY],
                                format_func=lambda m: m.value)
    weighting = st.selectbox("Weighting", list(WeightingMethod),
                             format_func=lambda m: m.value)
    reinvest = st.checkbox("Reinvest dividends (DRIP)", value=True)
    cost_bps = st.number_input("Transaction cost (bps)", 0.0, 200.0, 10.0)
    run = st.button("Run backtest", type="primary")

if run:
    cfg = dict(
        start=start, end=end, initial_capital=float(capital), base_currency="BRL",
        weighting=weighting, contribution_amount=float(contribution),
        contribution_frequency=contrib_freq, reinvest_dividends=reinvest,
        transaction_cost_bps=float(cost_bps), benchmark="^BVSP", market=mode,
    )
    try:
        full = _run(str(sector_path), cfg)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

    result, benchmarks, rep = full.result, full.benchmarks, full.report
    for w in result.warnings:
        st.warning(w)

    st.subheader("Performance (Brazilian frame)")
    c = st.columns(4)
    c[0].metric("Final value", f"R$ {result.equity.iloc[-1]:,.0f}")
    c[1].metric("Invested", f"R$ {result.total_invested:,.0f}")
    c[2].metric("Nominal CAGR", f"{rep.nominal_cagr:.1%}")
    c[3].metric("Real CAGR (IPCA)", f"{rep.real_cagr:.1%}")
    c = st.columns(4)
    c[0].metric("% do CDI", f"{rep.pct_of_cdi:.0f}%")
    c[1].metric("Sharpe vs CDI", f"{rep.sharpe_vs_cdi:.2f}")
    c[2].metric("Money-weighted (XIRR)", f"{rep.money_weighted_return:.1%}")
    c[3].metric("Max drawdown", f"{rep.max_drawdown:.1%}")

    st.plotly_chart(benchmark_comparison_chart(result, benchmarks),
                    width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(invested_vs_value_chart(result), width="stretch")
    b.plotly_chart(drawdown_chart(result), width="stretch")

    ipca = BcbIndicators().ipca_factor(result.equity.index)
    st.plotly_chart(real_vs_nominal_chart(result, ipca), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(allocation_area(result), width="stretch")
    b.plotly_chart(dividend_bar(result), width="stretch")
    st.plotly_chart(contribution_bar(result), width="stretch")

    st.download_button("Download equity CSV",
                       result.equity.rename("equity").to_csv().encode(),
                       file_name="equity.csv")

    st.divider()
    st.subheader("Projecao / Forecast")
    st.caption(
        "Projecao probabilistica por simulacao de Monte Carlo a partir da "
        "distribuicao historica de retornos. Bandas = percentis dos cenarios "
        "simulados, nao garantia. Nao e recomendacao de investimento."
    )
    fc1, fc2, fc3, fc4 = st.columns(4)
    horizon = fc1.slider("Horizon (years)", 1, 10, 5)
    conf = fc2.select_slider("Confidence", options=[0.5, 0.8, 0.9, 0.95], value=0.9)
    model = fc3.selectbox("Model", ["Block bootstrap", "GBM (lognormal)"])
    keep_aporte = fc4.checkbox("Continue aportes", value=True)
    forecaster = (BlockBootstrapForecaster() if model.startswith("Block")
                  else GbmForecaster())
    fc = forecast_from_result(result, horizon_years=horizon, forecaster=forecaster,
                              confidence=conf, n_paths=2000, seed=42,
                              continue_contributions=keep_aporte)
    st.plotly_chart(forecast_chart(fc, f"{sector_path.stem} - {horizon}y projection"),
                    width="stretch")
    t = fc.terminal
    m = st.columns(4)
    m[0].metric(f"Median in {horizon}y", f"R$ {t['median']:,.0f}")
    m[1].metric("Lower bound", f"R$ {t['lower']:,.0f}")
    m[2].metric("Upper bound", f"R$ {t['upper']:,.0f}")
    m[3].metric("P(> invested)", f"{t['prob_above_invested']:.0%}")
