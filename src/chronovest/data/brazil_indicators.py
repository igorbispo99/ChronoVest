"""Brazilian macro indicators: CDI and IPCA from the Banco Central (BCB).

Both come from the BCB SGS open API (no key required):
  * CDI  - series 12  (daily rate, percent per day)
  * IPCA - series 433 (monthly inflation, percent per month)

Each provider exposes accumulation-factor series rebased to 1.0 at the start of
the requested index, so "R$1 invested in the CDI" or "an IPCA-deflator" is just
``initial * factor``. ``StaticIndicators`` gives deterministic factors for tests
and offline use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

SGS_CDI = 12
SGS_IPCA = 433
_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"


class BrazilIndicators(ABC):
    @abstractmethod
    def cdi_factor(self, index: pd.DatetimeIndex) -> pd.Series:
        """Accumulated CDI growth factor, rebased to 1.0 at ``index[0]``."""

    @abstractmethod
    def ipca_factor(self, index: pd.DatetimeIndex) -> pd.Series:
        """Accumulated IPCA (price level) factor, rebased to 1.0 at ``index[0]``."""

    @abstractmethod
    def cdi_daily(self, index: pd.DatetimeIndex) -> pd.Series:
        """Daily CDI return (fraction), aligned to ``index`` (0 when unknown)."""


def _rebase(factor: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = factor.reindex(factor.index.union(index)).ffill().reindex(index).bfill()
    base = aligned.iloc[0]
    return aligned / base if base else aligned


class BcbIndicators(BrazilIndicators):
    def __init__(self) -> None:
        self._cache: dict[int, pd.Series] = {}

    def _fetch(self, code: int, start: date, end: date) -> pd.Series:
        if code in self._cache:
            return self._cache[code]
        import json
        import urllib.request

        url = (f"{_BASE.format(code=code)}?formato=json"
               f"&dataInicial={start.strftime('%d/%m/%Y')}"
               f"&dataFinal={end.strftime('%d/%m/%Y')}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        idx = pd.to_datetime([r["data"] for r in rows], format="%d/%m/%Y")
        vals = pd.Series([float(r["valor"]) for r in rows], index=idx).sort_index()
        self._cache[code] = vals
        return vals

    def cdi_daily(self, index):
        start, end = index[0].date(), index[-1].date()
        pct = self._fetch(SGS_CDI, start, end) / 100.0
        return pct.reindex(index).fillna(0.0)

    def cdi_factor(self, index):
        start, end = index[0].date(), index[-1].date()
        pct = self._fetch(SGS_CDI, start, end) / 100.0
        return _rebase((1.0 + pct).cumprod(), index)

    def ipca_factor(self, index):
        start, end = index[0].date(), index[-1].date()
        pct = self._fetch(SGS_IPCA, start, end) / 100.0
        return _rebase((1.0 + pct).cumprod(), index)


class StaticIndicators(BrazilIndicators):
    """Deterministic factors: constant daily CDI and constant monthly IPCA."""

    def __init__(self, cdi_daily_rate: float = 0.0004, ipca_monthly_rate: float = 0.004):
        self.cdi_daily_rate = cdi_daily_rate
        self.ipca_monthly_rate = ipca_monthly_rate

    def cdi_daily(self, index):
        return pd.Series(self.cdi_daily_rate, index=index)

    def cdi_factor(self, index):
        steps = pd.Series(range(len(index)), index=index)
        return (1.0 + self.cdi_daily_rate) ** steps

    def ipca_factor(self, index):
        daily = (1.0 + self.ipca_monthly_rate) ** (1.0 / 21.0) - 1.0
        steps = pd.Series(range(len(index)), index=index)
        return (1.0 + daily) ** steps
