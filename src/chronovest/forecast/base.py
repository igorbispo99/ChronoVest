"""Forecasting interfaces and result type.

A forecast here is *probabilistic*: equity prices are close to a random walk, so
a single-number "prediction" is not meaningful. Instead a Forecaster simulates
many plausible future return paths from the historical return distribution, and
the bounds are percentiles across those paths (a fan chart). This is a projection
of historical statistics, not a guarantee, and is not financial advice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    history: pd.Series
    median: pd.Series
    lower: pd.Series
    upper: pd.Series
    confidence: float
    horizon_years: float
    terminal: dict = field(default_factory=dict)
    paths: np.ndarray | None = None

    def band_label(self) -> str:
        lo = (1 - self.confidence) / 2 * 100
        hi = (1 + self.confidence) / 2 * 100
        return f"{lo:.0f}-{hi:.0f}% interval"


class Forecaster(ABC):
    """Estimates a return model from history and simulates future daily returns."""

    @abstractmethod
    def fit(self, returns: pd.Series) -> "Forecaster":
        """Learn from a series of historical daily simple returns."""

    @abstractmethod
    def simulate(self, horizon: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """Return an array of shape (n_paths, horizon) of simulated daily returns."""
