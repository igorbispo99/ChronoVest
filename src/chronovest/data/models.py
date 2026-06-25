"""Data models exchanged between the data layer and the rest of the system."""

from __future__ import annotations

from dataclasses import dataclass


PRICE_COLUMNS = ("price", "adj_close", "dividend")
"""Required columns of a per-ticker price frame.

price      -- split-adjusted close (share counts are continuous across splits)
adj_close  -- split and dividend adjusted close (total-return proxy)
dividend   -- cash dividend per (split-adjusted) share, on the ex-date, else 0
"""


@dataclass(frozen=True)
class SecurityMeta:
    """Static descriptive metadata for a security."""

    ticker: str
    name: str | None = None
    currency: str = "USD"
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
