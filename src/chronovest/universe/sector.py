"""Sector definitions with point-in-time membership.

A sector is a named set of companies, each carrying an optional active window
``[start, end]`` and an optional trading currency. The window lets the engine
answer the core question at any rebalance date: *which companies were in this
sector then?* Defunct companies (merged, delisted, bankrupt) belong in the
definition with an ``end`` date so the backtest is not survivorship-biased.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Membership:
    ticker: str
    start: date | None = None
    end: date | None = None
    currency: str | None = None
    note: str | None = None

    def active_on(self, on: date) -> bool:
        if isinstance(on, datetime):
            on = on.date()
        if self.start is not None and on < self.start:
            return False
        if self.end is not None and on > self.end:
            return False
        return True


@dataclass
class Sector:
    name: str
    classification: str
    members: list[Membership]
    currency: str | None = None
    market: str | None = None
    description: str | None = None

    def active_tickers(self, on: date) -> list[str]:
        """Tickers belonging to the sector on ``on``."""
        return [m.ticker for m in self.members if m.active_on(on)]

    def all_tickers(self) -> list[str]:
        return [m.ticker for m in self.members]

    def currency_of(self, ticker: str) -> str | None:
        """Explicit currency for a ticker: member override, else sector default."""
        for m in self.members:
            if m.ticker == ticker and m.currency:
                return m.currency
        return self.currency


def _parse_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def load_sector(path: str | Path) -> Sector:
    """Load a sector definition from YAML.

    Expected schema::

        name: Brazilian Banks
        classification: custom
        currency: BRL          # default trading currency for members
        description: ...
        members:
          - ticker: ITUB4.SA
            start: 2005-01-01   # optional
            end: null           # optional
            currency: BRL       # optional per-member override
            note: ...           # optional
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    members = [
        Membership(
            ticker=m["ticker"],
            start=_parse_date(m.get("start")),
            end=_parse_date(m.get("end")),
            currency=m.get("currency"),
            note=m.get("note"),
        )
        for m in data.get("members", [])
    ]
    return Sector(
        name=data["name"],
        classification=data.get("classification", "custom"),
        members=members,
        currency=data.get("currency"),
        market=data.get("market"),
        description=data.get("description"),
    )
