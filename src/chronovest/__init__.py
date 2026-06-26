"""ChronoVest: historical sector portfolio backtesting engine.

Layered architecture:
    data      -> ingestion and caching of prices, dividends, market caps
    universe  -> point-in-time sector membership
    engine    -> weighting strategies and the backtest portfolio engine
    analytics -> performance and risk metrics
    viz       -> plotly visualizations
    app       -> streamlit front-end

The layers communicate through small, explicit interfaces so any one of them
(e.g. the data provider) can be swapped without touching the others.
"""

from chronovest.config import BacktestConfig, RebalanceFrequency, WeightingMethod

__all__ = ["BacktestConfig", "RebalanceFrequency", "WeightingMethod"]
__version__ = "0.1.0"
