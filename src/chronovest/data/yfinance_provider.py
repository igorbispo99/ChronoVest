"""yfinance-backed DataProvider.

Fails soft: a ticker that 404s, is delisted, or returns nothing yields an empty
frame instead of raising, so one bad symbol never breaks a whole sector backtest
(the engine reports skipped tickers as a warning).

Free-data caveats (documented intentionally):
  * Historical market capitalization is not directly available. It is
    approximated as ``shares_outstanding(t) x price(t)`` using yfinance's
    historical share count when available, else the latest known share count
    held constant. Treat market-cap weights as best-effort, not point-in-time.
  * Sector membership is not reconstructed here; it comes from the universe
    layer. Include defunct tickers there (with end dates) to avoid survivorship
    bias.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from chronovest.data.base import DataProvider, PriceFrame
from chronovest.data.models import SecurityMeta

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class YFinanceProvider(DataProvider):
    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("yfinance is required: pip install yfinance") from exc

    def get_prices(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, PriceFrame]:
        import yfinance as yf

        out: dict[str, PriceFrame] = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(
                    start=str(start), end=str(end), auto_adjust=False, actions=True
                )
            except Exception:  # delisted / 404 / network variance
                hist = None
            if hist is None or hist.empty:
                out[t] = _empty_price_frame()
                continue
            try:
                hist.index = pd.to_datetime(hist.index).tz_localize(None)
                frame = pd.DataFrame(index=hist.index)
                frame["price"] = hist["Close"].astype(float)
                frame["adj_close"] = hist.get("Adj Close", hist["Close"]).astype(float)
                frame["dividend"] = hist.get("Dividends", 0.0).astype(float).fillna(0.0)
                out[t] = frame
            except Exception:  # pragma: no cover - malformed payload
                out[t] = _empty_price_frame()
        return out

    def get_market_caps(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        import yfinance as yf

        prices = self.get_prices(tickers, start, end)
        cols = {}
        for t in tickers:
            px = prices.get(t)
            if px is None or px.empty:
                continue
            try:
                shares = self._shares_series(yf.Ticker(t), px.index)
                cols[t] = px["price"] * shares
            except Exception:  # pragma: no cover - keep one bad ticker from failing
                continue
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols)

    @staticmethod
    def _shares_series(ticker, index: pd.DatetimeIndex) -> pd.Series:
        shares = None
        try:
            full = ticker.get_shares_full(start=str(index.min().date()))
            if full is not None and len(full):
                full.index = pd.to_datetime(full.index).tz_localize(None)
                shares = full.reindex(index, method="ffill")
        except Exception:  # pragma: no cover - network/parse variance
            shares = None
        if shares is None or shares.isna().all():
            static = None
            try:
                static = ticker.info.get("sharesOutstanding")
            except Exception:  # pragma: no cover
                static = None
            shares = pd.Series(static if static else np.nan, index=index)
        return shares.astype(float)

    def get_meta(self, tickers: list[str]) -> dict[str, SecurityMeta]:
        import yfinance as yf

        out: dict[str, SecurityMeta] = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
            except Exception:  # pragma: no cover
                info = {}
            out[t] = SecurityMeta(
                ticker=t,
                name=info.get("shortName"),
                currency=info.get("currency", "USD"),
                exchange=info.get("exchange"),
                sector=info.get("sector"),
                industry=info.get("industry"),
            )
        return out


def _empty_price_frame() -> PriceFrame:
    return pd.DataFrame(columns=["price", "adj_close", "dividend"]).astype(float)
