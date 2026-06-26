"""Visualization layer: plotly figures built from a BacktestResult."""

from chronovest.viz.charts import (
    allocation_area,
    benchmark_comparison_chart,
    contribution_bar,
    dividend_bar,
    drawdown_chart,
    equity_chart,
    export_excel,
    forecast_chart,
    invested_vs_value_chart,
    real_vs_nominal_chart,
)

__all__ = [
    "equity_chart",
    "drawdown_chart",
    "allocation_area",
    "contribution_bar",
    "dividend_bar",
    "benchmark_comparison_chart",
    "real_vs_nominal_chart",
    "invested_vs_value_chart",
    "forecast_chart",
    "export_excel",
]
