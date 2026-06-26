"""Synthetic data provider so the engine is tested without any network."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from chronovest.data.base import DataProvider
from chronovest.universe.sector import Membership, Sector


class SyntheticProvider(DataProvider):
    """Deterministic prices: each ticker grows at a fixed daily rate.

    Optional dividends and a listing window let tests exercise reinvestment,
    IPOs and delistings.
    """

    def __init__(self, specs: dict[str, dict]):
        self.specs = specs

    def _frame(self, spec, start, end):
        idx = pd.bdate_range(start, end)
        first = spec.get("first")
        last = spec.get("last")
        price = pd.Series(np.nan, index=idx, dtype=float)
        rate = spec.get("rate", 0.0)
        p0 = spec.get("p0", 100.0)
        live = idx
        if first:
            live = live[live >= pd.Timestamp(first)]
        if last:
            live = live[live <= pd.Timestamp(last)]
        steps = np.arange(len(live))
        price.loc[live] = p0 * (1 + rate) ** steps
        div = pd.Series(0.0, index=idx)
        for d in spec.get("dividends", []):
            ts = pd.Timestamp(d["date"])
            if ts in div.index:
                div.loc[ts] = d["amount"]
        return pd.DataFrame(
            {"price": price, "adj_close": price.ffill(), "dividend": div}
        ).dropna(subset=["price"], how="all")

    def get_prices(self, tickers, start, end):
        return {t: self._frame(self.specs[t], start, end)
                for t in tickers if t in self.specs}

    def get_market_caps(self, tickers, start, end):
        prices = self.get_prices(tickers, start, end)
        cols = {}
        for t, df in prices.items():
            shares = self.specs[t].get("shares", 1_000_000)
            cols[t] = df["price"] * shares
        return pd.DataFrame(cols)


@pytest.fixture
def two_stock_sector():
    sector = Sector(
        name="Test",
        classification="custom",
        members=[Membership("AAA"), Membership("BBB")],
    )
    provider = SyntheticProvider({
        "AAA": {"p0": 100.0, "rate": 0.001, "shares": 3_000_000},
        "BBB": {"p0": 50.0, "rate": 0.0005, "shares": 1_000_000},
    })
    return sector, provider


@pytest.fixture
def span():
    return date(2020, 1, 1), date(2021, 12, 31)
