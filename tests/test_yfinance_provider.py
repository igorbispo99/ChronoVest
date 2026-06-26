"""Tests for YFinanceProvider resilience and version requirements."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import packaging.version
import pytest
import yfinance

from chronovest.data.yfinance_provider import YFinanceProvider

MIN_YFINANCE = "0.2.54"
START = date(2026, 1, 1)
END = date(2026, 3, 31)


def test_yfinance_version_meets_minimum():
    installed = packaging.version.Version(yfinance.__version__)
    required = packaging.version.Version(MIN_YFINANCE)
    assert installed >= required, (
        f"yfinance {installed} is too old; upgrade to >={MIN_YFINANCE} to avoid "
        "HTTP 406 errors from the updated Yahoo Finance API."
    )


def _http_406():
    return HTTPError(url=None, code=406, msg="Not Acceptable", hdrs=None, fp=None)


@patch("yfinance.Ticker")
def test_provider_returns_empty_frame_on_406(mock_ticker_cls):
    ticker_obj = MagicMock()
    ticker_obj.history.side_effect = _http_406()
    mock_ticker_cls.return_value = ticker_obj

    provider = YFinanceProvider()
    result = provider.get_prices(["AAPL"], START, END)

    assert "AAPL" in result
    assert result["AAPL"].empty


@patch("yfinance.Ticker")
def test_provider_skips_bad_ticker_and_returns_valid_ones(mock_ticker_cls):
    """A 406 on one ticker must not abort the whole batch."""
    import pandas as pd

    good_hist = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Adj Close": [100.0, 101.0],
            "Dividends": [0.0, 0.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    good_hist.index = good_hist.index.tz_localize("UTC")

    def side_effect(symbol):
        obj = MagicMock()
        if symbol == "BAD":
            obj.history.side_effect = _http_406()
        else:
            obj.history.return_value = good_hist
        return obj

    mock_ticker_cls.side_effect = side_effect

    provider = YFinanceProvider()
    result = provider.get_prices(["BAD", "GOOD"], START, END)

    assert result["BAD"].empty
    assert not result["GOOD"].empty
    assert list(result["GOOD"].columns) == ["price", "adj_close", "dividend"]
