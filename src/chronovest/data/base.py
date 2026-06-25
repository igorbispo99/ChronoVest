"""Provider interface. Implement this to plug in a new data source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from chronovest.data.models import SecurityMeta

PriceFrame = pd.DataFrame
"""A DataFrame indexed by a DatetimeIndex with columns from PRICE_COLUMNS."""


class DataProvider(ABC):
    """Abstract source of historical market data.

    Implementations must return data already aligned to calendar dates. The
    engine treats missing rows for a ticker as 'not listed / not trading' on
    that date, which is how IPOs and delistings are handled downstream.
    """

    @abstractmethod
    def get_prices(
        self, tickers: list[str], start: date, end: date
    ) -> dict[str, PriceFrame]:
        """Return ticker -> price frame (columns: price, adj_close, dividend)."""

    @abstractmethod
    def get_market_caps(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Return a wide frame: index=date, columns=tickers, values=market cap.

        Values may be NaN when capitalization is unknown for a date/ticker.
        """

    def get_meta(self, tickers: list[str]) -> dict[str, SecurityMeta]:
        """Return descriptive metadata. Default: minimal stubs."""
        return {t: SecurityMeta(ticker=t) for t in tickers}
