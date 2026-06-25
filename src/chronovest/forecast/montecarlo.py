"""Monte Carlo forecasters.

GbmForecaster        -- parametric geometric Brownian motion: daily log returns
                        are Normal(mu, sigma) estimated from history. Smooth,
                        thin-tailed.
BlockBootstrapForecaster -- non-parametric: future paths are built by resampling
                        contiguous blocks of historical log returns, which keeps
                        autocorrelation, volatility clustering and fat tails. More
                        realistic for financial series; the default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronovest.forecast.base import Forecaster


def _log_returns(returns: pd.Series) -> np.ndarray:
    r = returns.dropna().to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    return np.log1p(np.clip(r, -0.99, None))


class GbmForecaster(Forecaster):
    def __init__(self) -> None:
        self.mu = 0.0
        self.sigma = 0.0

    def fit(self, returns):
        lr = _log_returns(returns)
        if lr.size < 2:
            raise ValueError("need at least 2 returns to fit")
        self.mu = float(np.mean(lr))
        self.sigma = float(np.std(lr, ddof=1))
        return self

    def simulate(self, horizon, n_paths, rng):
        shocks = rng.normal(self.mu, self.sigma, size=(n_paths, horizon))
        return np.expm1(shocks)


class BlockBootstrapForecaster(Forecaster):
    def __init__(self, block: int = 21) -> None:
        self.block = block
        self._lr = np.empty(0)

    def fit(self, returns):
        self._lr = _log_returns(returns)
        if self._lr.size < self.block:
            self.block = max(1, self._lr.size)
        if self._lr.size < 2:
            raise ValueError("need at least 2 returns to fit")
        return self

    def simulate(self, horizon, n_paths, rng):
        n = self._lr.size
        n_blocks = int(np.ceil(horizon / self.block))
        starts = rng.integers(0, max(1, n - self.block + 1),
                              size=(n_paths, n_blocks))
        offsets = np.arange(self.block)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
        idx = np.clip(idx[:, :horizon], 0, n - 1)
        return np.expm1(self._lr[idx])
