"""On-disk parquet cache wrapper for any DataProvider.

Wraps a provider so repeated backtests do not re-download the same series.
Cache key is (provider class, ticker, kind). Range requests are served from
cache when the cached range covers them; otherwise the wrapped provider is
queried and the result stored.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

from chronovest.data.base import DataProvider, PriceFrame


class CachedProvider(DataProvider):
    def __init__(self, inner: DataProvider, cache_dir: str | Path = ".cache") -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, ticker: str) -> Path:
        key = f"{type(self.inner).__name__}:{kind}:{ticker}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        safe = ticker.replace("/", "_").replace("^", "_")
        return self.cache_dir / f"{kind}_{safe}_{digest}.parquet"

    def get_prices(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, PriceFrame]:
        out: dict[str, PriceFrame] = {}
        missing: list[str] = []
        for t in tickers:
            p = self._path("prices", t)
            if p.exists():
                df = pd.read_parquet(p)
                if df.index.min().date() <= start and df.index.max().date() >= end:
                    out[t] = df.loc[str(start):str(end)]
                    continue
            missing.append(t)
        if missing:
            fetched = self.inner.get_prices(missing, start, end)
            for t, df in fetched.items():
                if not df.empty:
                    df.to_parquet(self._path("prices", t))
                out[t] = df
        return out

    def get_market_caps(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        return self.inner.get_market_caps(tickers, start, end)

    def get_meta(self, tickers: list[str]):
        return self.inner.get_meta(tickers)
