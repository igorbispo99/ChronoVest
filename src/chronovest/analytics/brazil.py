"""Brazil-framed analytics: CDI, IPCA and money-weighted returns.

Everything here keeps the Brazilian investor's perspective: nominal returns are
compared against the CDI (the local risk-free), deflated by the IPCA to give
real returns, and benchmark curves are built with the *same* contribution
schedule as the portfolio so the comparison is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chronovest.analytics.metrics import (
    TRADING_DAYS,
    cagr,
    max_drawdown,
    total_return,
)
from chronovest.data.brazil_indicators import BrazilIndicators


def simulate_dca(price: pd.Series, flows: pd.Series, cost_rate: float = 0.0) -> pd.Series:
    """Value over time of investing ``flows`` into a single asset, never selling.

    ``price`` is the asset's value per unit (a stock index, or a CDI factor);
    ``flows`` is external cash in by date (initial at the first date, then each
    contribution). Units only ever accumulate.
    """
    price = price.dropna()
    if price.empty:
        return pd.Series(dtype=float)
    f = flows.reindex(price.index).fillna(0.0)
    units = 0.0
    out = pd.Series(index=price.index, dtype=float)
    for day, p in price.items():
        if f.loc[day] > 0 and p > 0:
            units += f.loc[day] * (1.0 - cost_rate) / p
        out.loc[day] = units * p
    return out


def real_series(equity: pd.Series, ipca_factor: pd.Series) -> pd.Series:
    """Deflate a nominal curve to real terms (reais of the start date)."""
    return equity / ipca_factor.reindex(equity.index).ffill()


def sharpe_vs_cdi(returns_index: pd.Series, cdi_daily: pd.Series,
                  periods: int = TRADING_DAYS) -> float:
    r = returns_index.dropna().pct_change().dropna()
    if len(r) < 2:
        return 0.0
    excess = r - cdi_daily.reindex(r.index).fillna(0.0)
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def xirr(cashflows: pd.Series, guess: float = 0.1) -> float:
    """Annualised money-weighted return. ``cashflows``: negative in, positive out."""
    cf = cashflows.dropna()
    if cf.empty or (cf > 0).sum() == 0 or (cf < 0).sum() == 0:
        return 0.0
    t0 = cf.index[0]
    years = np.array([(d - t0).days / 365.25 for d in cf.index])
    amounts = cf.values

    def npv(rate):
        return np.sum(amounts / (1.0 + rate) ** years)

    low, high = -0.9999, 10.0
    f_low, f_high = npv(low), npv(high)
    if np.sign(f_low) == np.sign(f_high):
        return float("nan")
    for _ in range(200):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-6:
            return float(mid)
        if np.sign(f_mid) == np.sign(f_low):
            low, f_low = mid, f_mid
        else:
            high = mid
    return float((low + high) / 2.0)


def money_weighted_return(flows: pd.Series, final_value: float,
                          final_date=None) -> float:
    """XIRR from the investor's external flows and the final portfolio value."""
    cf = -flows[flows > 0].copy()
    end = final_date or flows.index[-1]
    cf.loc[end] = cf.get(end, 0.0) + final_value
    return xirr(cf.sort_index())


@dataclass
class BrazilReport:
    nominal_total_return: float
    nominal_cagr: float
    real_total_return: float
    real_cagr: float
    cdi_total_return: float
    pct_of_cdi: float
    ipca_total: float
    sharpe_vs_cdi: float
    max_drawdown: float
    money_weighted_return: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def brazil_report(result, indicators: BrazilIndicators) -> BrazilReport:
    """Build the Brazilian performance report from a BacktestResult.

    Return/risk metrics use the time-weighted index (``result.twr``) so periodic
    contributions do not distort them; the money-weighted return uses actual
    flows.
    """
    twr = result.twr.dropna()
    index = twr.index
    cdi_f = indicators.cdi_factor(index)
    ipca_f = indicators.ipca_factor(index)

    nom_total = total_return(twr)
    nom_cagr = cagr(twr)
    ipca_total = float(ipca_f.iloc[-1] / ipca_f.iloc[0] - 1.0)
    cdi_total = float(cdi_f.iloc[-1] / cdi_f.iloc[0] - 1.0)
    real_total = (1.0 + nom_total) / (1.0 + ipca_total) - 1.0
    real_idx = real_series(twr, ipca_f)
    real_cagr = cagr(real_idx)
    pct_cdi = (nom_total / cdi_total * 100.0) if abs(cdi_total) > 1e-9 else float("nan")

    return BrazilReport(
        nominal_total_return=nom_total,
        nominal_cagr=nom_cagr,
        real_total_return=real_total,
        real_cagr=real_cagr,
        cdi_total_return=cdi_total,
        pct_of_cdi=pct_cdi,
        ipca_total=ipca_total,
        sharpe_vs_cdi=sharpe_vs_cdi(twr, indicators.cdi_daily(index)),
        max_drawdown=max_drawdown(result.equity),
        money_weighted_return=money_weighted_return(
            result.flows, float(result.equity.iloc[-1])),
    )


def build_benchmarks(result, indicators: BrazilIndicators,
                     price_curves: dict[str, pd.Series] | None = None,
                     cost_rate: float = 0.0) -> dict[str, pd.Series]:
    """Comparable benchmark equity curves using the portfolio's own flows.

    Always includes CDI. ``price_curves`` adds equity indices (already in BRL),
    e.g. Ibovespa or an international index, each simulated with the same
    contribution schedule.
    """
    index = result.equity.index
    flows = result.flows
    out: dict[str, pd.Series] = {}
    out["CDI"] = simulate_dca(indicators.cdi_factor(index), flows)
    for name, price in (price_curves or {}).items():
        curve = simulate_dca(price.reindex(index).ffill(), flows, cost_rate)
        if not curve.empty:
            out[name] = curve
    return out
