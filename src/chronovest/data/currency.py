"""Currency conversion and ticker-to-currency inference.

Securities trade in their local currency (B3 in BRL, SIX in CHF, Tokyo in JPY,
NYSE in USD). To value a multi-currency portfolio the engine converts every
price, dividend and market cap into the configured base currency using a daily
FX series. ``CurrencyConverter`` is the seam: the yfinance implementation pulls
rates from Yahoo, while tests use a deterministic static converter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

SUFFIX_CURRENCY = {
    ".SA": "BRL",
    ".SW": "CHF",
    ".T": "JPY",
    ".HK": "HKD",
    ".L": "GBP",
    ".PA": "EUR",
    ".AS": "EUR",
    ".DE": "EUR",
    ".F": "EUR",
    ".MI": "EUR",
    ".MC": "EUR",
    ".NS": "INR",
    ".BO": "INR",
    ".TO": "CAD",
    ".AX": "AUD",
    ".MX": "MXN",
}

INDEX_CURRENCY = {
    "^BVSP": "BRL",
    "^GSPC": "USD",
    "^IXIC": "USD",
    "^DJI": "USD",
    "^N225": "JPY",
    "^STOXX": "EUR",
    "^FTSE": "GBP",
}


def infer_currency(ticker: str, default: str = "USD") -> str:
    """Best-effort currency from a ticker symbol's exchange suffix."""
    if ticker in INDEX_CURRENCY:
        return INDEX_CURRENCY[ticker]
    for suffix, ccy in SUFFIX_CURRENCY.items():
        if ticker.endswith(suffix):
            return ccy
    return default


class CurrencyConverter(ABC):
    @abstractmethod
    def rate(self, frm: str, to: str, index: pd.DatetimeIndex) -> pd.Series:
        """Units of ``to`` per 1 unit of ``frm``, aligned to ``index``."""


class IdentityConverter(CurrencyConverter):
    """No conversion: every rate is 1. Used when a portfolio is single-currency."""

    def rate(self, frm, to, index):
        return pd.Series(1.0, index=index)


class StaticConverter(CurrencyConverter):
    """Deterministic converter for tests and offline use.

    ``rates`` maps ``(frm, to)`` to a constant or a pandas Series. Missing pairs
    fall back to the inverse if available, else 1.0.
    """

    def __init__(self, rates: dict[tuple[str, str], object] | None = None):
        self.rates = rates or {}

    def rate(self, frm, to, index):
        if frm == to:
            return pd.Series(1.0, index=index)
        if (frm, to) in self.rates:
            return self._series(self.rates[(frm, to)], index)
        if (to, frm) in self.rates:
            return 1.0 / self._series(self.rates[(to, frm)], index)
        return pd.Series(1.0, index=index)

    @staticmethod
    def _series(value, index):
        if isinstance(value, pd.Series):
            return value.reindex(index).ffill().bfill()
        return pd.Series(float(value), index=index)


class YFinanceConverter(CurrencyConverter):
    """FX rates from Yahoo Finance pairs like ``USDBRL=X``.

    Uses a browser-impersonating session (when curl_cffi is available) to avoid
    Yahoo's HTTP 406 rejections.
    """

    def __init__(self):
        self._cache: dict[tuple[str, str], pd.Series] = {}
        from chronovest.data._yf_session import make_impersonated_session

        self._session = make_impersonated_session()

    def rate(self, frm, to, index):
        if frm == to:
            return pd.Series(1.0, index=index)
        series = self._fetch(frm, to)
        if series is None:
            inv = self._fetch(to, frm)
            series = 1.0 / inv if inv is not None else None
        if series is None:
            return pd.Series(1.0, index=index)
        return series.reindex(index.union(series.index)).ffill().reindex(index).bfill()

    def _fetch(self, frm, to) -> pd.Series | None:
        key = (frm, to)
        if key in self._cache:
            return self._cache[key]
        import yfinance as yf

        try:
            pair = f"{frm}{to}=X"
            ticker = (yf.Ticker(pair, session=self._session)
                      if self._session is not None else yf.Ticker(pair))
            hist = ticker.history(period="max", auto_adjust=True)
        except Exception:  # pragma: no cover - network variance / 406
            hist = None
        if hist is None or hist.empty:
            self._cache[key] = None
            return None
        s = hist["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        self._cache[key] = s
        return s
