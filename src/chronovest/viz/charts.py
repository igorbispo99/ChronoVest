"""Plotly figures and spreadsheet export for a BacktestResult."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from chronovest.analytics.metrics import drawdown_series
from chronovest.engine.portfolio import BacktestResult


def equity_chart(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=result.equity.index, y=result.equity.values,
                   name="Portfolio", mode="lines")
    )
    if result.benchmark is not None:
        fig.add_trace(
            go.Scatter(x=result.benchmark.index, y=result.benchmark.values,
                       name="Benchmark", mode="lines", line=dict(dash="dash"))
        )
    fig.update_layout(title="Portfolio value over time",
                      xaxis_title="Date", yaxis_title=result.config.base_currency)
    return fig


def drawdown_chart(result: BacktestResult) -> go.Figure:
    dd = drawdown_series(result.equity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                             name="Drawdown"))
    fig.update_layout(title="Drawdown", xaxis_title="Date",
                      yaxis_title="Drawdown", yaxis_tickformat=".0%")
    return fig


def allocation_area(result: BacktestResult, top: int = 12) -> go.Figure:
    hv = result.holdings_value.copy()
    totals = hv.sum().sort_values(ascending=False)
    keep = list(totals.head(top).index)
    data = hv[keep].copy()
    if len(totals) > top:
        data["Other"] = hv[[c for c in hv.columns if c not in keep]].sum(axis=1)
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col,
                                 stackgroup="one", mode="lines"))
    fig.update_layout(title="Company allocation over time",
                      xaxis_title="Date", yaxis_title=result.config.base_currency)
    return fig


def contribution_bar(result: BacktestResult, top: int = 15) -> go.Figure:
    c = result.contributions.sort_values()
    c = pd.concat([c.head(top), c.tail(top)]).drop_duplicates()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h"))
    fig.update_layout(title="Contribution to P&L by company",
                      xaxis_title=result.config.base_currency)
    return fig


def dividend_bar(result: BacktestResult) -> go.Figure:
    annual = result.dividend_income.resample("YE").sum()
    annual.index = annual.index.year
    fig = go.Figure(go.Bar(x=annual.index.astype(str), y=annual.values))
    fig.update_layout(title="Dividend income per year",
                      xaxis_title="Year", yaxis_title=result.config.base_currency)
    return fig


def export_excel(result: BacktestResult, path: str | Path) -> Path:
    """Write equity, holdings, dividends, contributions and trades to .xlsx."""
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        result.equity.rename("equity").to_frame().to_excel(xl, sheet_name="equity")
        result.holdings_value.to_excel(xl, sheet_name="holdings_value")
        result.dividend_income.rename("dividends").to_frame().to_excel(
            xl, sheet_name="dividends"
        )
        result.contributions.rename("pnl").to_frame().to_excel(
            xl, sheet_name="contributions"
        )
        if not result.trades.empty:
            result.trades.to_excel(xl, sheet_name="trades", index=False)
    return path


def benchmark_comparison_chart(result: BacktestResult,
                               benchmarks: dict | None = None) -> go.Figure:
    """Portfolio value vs comparable benchmark curves (CDI, Ibovespa, ...)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="Portfolio", mode="lines",
                             line=dict(width=3)))
    for name, curve in (benchmarks or {}).items():
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=name,
                                 mode="lines", line=dict(dash="dash")))
    fig.update_layout(title="Portfolio vs benchmarks (same contributions)",
                      xaxis_title="Date", yaxis_title=result.config.base_currency)
    return fig


def real_vs_nominal_chart(result: BacktestResult, ipca_factor) -> go.Figure:
    """Nominal equity vs IPCA-deflated (real) equity, in start-date reais."""
    nominal = result.equity
    real = nominal / ipca_factor.reindex(nominal.index).ffill()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nominal.index, y=nominal.values, name="Nominal",
                             mode="lines"))
    fig.add_trace(go.Scatter(x=real.index, y=real.values, name="Real (IPCA)",
                             mode="lines"))
    fig.update_layout(title="Nominal vs real value (IPCA-deflated)",
                      xaxis_title="Date", yaxis_title=result.config.base_currency)
    return fig


def invested_vs_value_chart(result: BacktestResult) -> go.Figure:
    """Cumulative money contributed vs portfolio value over time."""
    invested = result.flows.cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=invested.index, y=invested.values, name="Invested",
                             mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="Value", mode="lines"))
    fig.update_layout(title="Money invested vs portfolio value",
                      xaxis_title="Date", yaxis_title=result.config.base_currency)
    return fig


def forecast_chart(forecast, title: str = "Sector forecast") -> go.Figure:
    """Fan chart: historical value, median projection and a shaded bound band."""
    fig = go.Figure()
    hist = getattr(forecast, "history", None)
    if hist is not None and len(hist) > 0:
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History",
                                 mode="lines", line=dict(color="#2b6cb0")))
    fig.add_trace(go.Scatter(x=forecast.upper.index, y=forecast.upper.values,
                             name="Upper bound", mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast.lower.index, y=forecast.lower.values,
                             name=forecast.band_label(), mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(49,130,189,0.2)"))
    fig.add_trace(go.Scatter(x=forecast.median.index, y=forecast.median.values,
                             name="Median projection", mode="lines",
                             line=dict(color="#e8590c", dash="dash")))
    fig.update_layout(title=title, xaxis_title="Date",
                      yaxis_title="Value")
    return fig
