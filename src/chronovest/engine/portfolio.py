"""The backtest portfolio engine.

Default strategy (no selling): capital deployed at the start, plus every
periodic contribution (aporte), buys the *currently balanced* basket of the
sector at that moment - the engine asks "which companies are in the sector now
and how big is each", derives weights, and buys only. Existing positions are
held; winners run. Dividends are reinvested into the paying stock (DRIP) or kept
as idle cash. Delisted names are liquidated by the corporate action and the
proceeds are redeployed at the next buy. Set ``allow_selling`` for classic
sell-to-target rebalancing instead.

Returns are reported two ways so that contributions do not distort them:
``equity`` is the actual portfolio value (money in), while ``twr`` is the
time-weighted return index (the strategy's intrinsic performance, flows removed).

Multi-currency: every price, dividend and market cap is converted into
``config.base_currency`` via the CurrencyConverter, so the whole simulation runs
in one currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from chronovest.config import BacktestConfig, RebalanceFrequency, WeightingMethod
from chronovest.data.base import DataProvider
from chronovest.data.currency import CurrencyConverter, infer_currency
from chronovest.engine.weighting import (
    WeightingContext,
    WeightingStrategy,
    build_weighting,
)
from chronovest.universe.sector import Sector

_FREQ = {
    RebalanceFrequency.MONTHLY: "MS",
    RebalanceFrequency.QUARTERLY: "QS",
    RebalanceFrequency.YEARLY: "YS",
}


@dataclass
class BacktestResult:
    """Outputs of a backtest, consumed by the analytics and viz layers."""

    config: BacktestConfig
    equity: pd.Series
    twr: pd.Series
    flows: pd.Series
    holdings_value: pd.DataFrame
    target_weights: pd.DataFrame
    dividend_income: pd.Series
    contributions: pd.Series
    trades: pd.DataFrame
    total_contributed: float = 0.0
    benchmark: pd.Series | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total_invested(self) -> float:
        return float(self.config.initial_capital + self.total_contributed)


class PortfolioEngine:
    def __init__(
        self,
        provider: DataProvider,
        sector: Sector,
        weighting: WeightingStrategy | None = None,
        fx: CurrencyConverter | None = None,
    ) -> None:
        self.provider = provider
        self.sector = sector
        self.fx = fx
        self._explicit_weighting = weighting

    def run(self, config: BacktestConfig) -> BacktestResult:
        weighting = self._explicit_weighting or build_weighting(config.weighting)
        warnings: list[str] = []
        self._cash = float(config.initial_capital)
        self._reserve = 0.0
        self._contributed = 0.0
        self._base = config.base_currency
        tickers = self.sector.all_tickers()

        currencies = {t: self._ccy(t) for t in tickers}
        if self.fx is None and any(c != self._base for c in currencies.values()):
            warnings.append(
                "no fx converter supplied; non-base currencies are assumed already "
                f"in {self._base} (values may be wrong for mixed-currency sectors)"
            )

        price, div, value_price = self._price_panels(tickers, config, currencies)
        if price.empty:
            raise ValueError(
                "no market data returned for any ticker in the sector. This is "
                "usually an outdated yfinance hitting Yahoo's HTTP 406; update it "
                "with: pip install -U yfinance curl_cffi"
            )
        missing = [t for t in self.sector.all_tickers() if t not in price.columns]
        if missing:
            warnings.append(
                "no price data (skipped; delisted or unavailable): "
                + ", ".join(missing)
            )
        tickers = list(price.columns)
        caps = self._caps_panel(tickers, config, weighting, currencies)

        calendar = price.index
        first = calendar[0]
        contrib_dates = (
            self._schedule(calendar, config.contribution_frequency, skip_first=True)
            if config.contribution_amount > 0 else set()
        )
        buy_dates = {first} | contrib_dates
        sell_dates = (
            self._schedule(calendar, config.rebalance, skip_first=False)
            if config.allow_selling else set()
        )
        last_valid = {t: price[t].last_valid_index() for t in tickers}

        shares = {t: 0.0 for t in tickers}
        last_price = {t: np.nan for t in tickers}
        buys = {t: 0.0 for t in tickers}
        sells = {t: 0.0 for t in tickers}
        divs_recv = {t: 0.0 for t in tickers}

        equity = pd.Series(index=calendar, dtype=float)
        twr = pd.Series(index=calendar, dtype=float)
        flows = pd.Series(0.0, index=calendar, dtype=float)
        div_income = pd.Series(0.0, index=calendar, dtype=float)
        holdings_value = pd.DataFrame(0.0, index=calendar, columns=tickers)
        weights_log: dict[date, dict[str, float]] = {}
        trades: list[dict] = []
        cost_rate = config.transaction_cost_bps / 1e4

        flows.loc[first] = config.initial_capital
        prev_total: float | None = None
        prev_twr = float(config.initial_capital)

        for day in calendar:
            px = value_price.loc[day]
            for t in tickers:
                if not np.isnan(px[t]):
                    last_price[t] = px[t]

            for t in tickers:
                lv = last_valid[t]
                if shares[t] > 0 and lv is not None and day > lv:
                    proceeds = shares[t] * last_price[t]
                    if proceeds > 0:
                        sells[t] += proceeds
                        trades.append({"date": day, "ticker": t,
                                       "action": "delist_liquidate", "value": proceeds})
                        self._cash += proceeds
                        shares[t] = 0.0

            today_div = 0.0
            for t in tickers:
                d = div.loc[day, t] if t in div.columns else 0.0
                if shares[t] > 0 and d and not np.isnan(d):
                    income = shares[t] * d
                    today_div += income
                    divs_recv[t] += income
                    if config.reinvest_dividends and not np.isnan(px[t]) and px[t] > 0:
                        shares[t] += income / px[t]
                        buys[t] += income
                    else:
                        self._reserve += income
            div_income.loc[day] = today_div

            flow_today = 0.0
            if day in contrib_dates:
                self._cash += config.contribution_amount
                self._contributed += config.contribution_amount
                flow_today += config.contribution_amount
                flows.loc[day] += config.contribution_amount

            if config.allow_selling and day in sell_dates:
                self._full_rebalance(day, tickers, shares, last_price, caps, config,
                                     weighting, cost_rate, buys, sells, trades,
                                     weights_log)
            elif day in buy_dates and self._cash > 1e-9:
                self._deploy_cash(day, tickers, shares, last_price, caps, config,
                                  weighting, cost_rate, buys, trades, weights_log)

            total = self._cash + self._reserve + sum(
                shares[t] * last_price[t]
                for t in tickers if not np.isnan(last_price[t])
            )
            equity.loc[day] = total
            if prev_total is None or prev_total <= 0:
                cur_twr = float(config.initial_capital)
            else:
                r = (total - flow_today) / prev_total - 1.0
                cur_twr = prev_twr * (1.0 + r)
            twr.loc[day] = cur_twr
            prev_total, prev_twr = total, cur_twr
            for t in tickers:
                lp = last_price[t]
                holdings_value.loc[day, t] = shares[t] * lp if not np.isnan(lp) else 0.0

        contributions = pd.Series(
            {t: holdings_value.iloc[-1][t] + sells[t] + divs_recv[t] - buys[t]
             for t in tickers}
        ).sort_values(ascending=False)

        benchmark = self._benchmark(config)
        if caps is None and config.weighting in (
            WeightingMethod.MARKET_CAP, WeightingMethod.FREE_FLOAT
        ):
            warnings.append("market caps unavailable; fell back to equal weighting")

        return BacktestResult(
            config=config,
            equity=equity,
            twr=twr,
            flows=flows,
            holdings_value=holdings_value,
            target_weights=pd.DataFrame(weights_log).T.sort_index(),
            dividend_income=div_income,
            contributions=contributions,
            trades=pd.DataFrame(trades),
            total_contributed=self._contributed,
            benchmark=benchmark,
            warnings=warnings,
        )

    def _weights_at(self, day, active, caps, config, weighting):
        cap_row = None
        if caps is not None:
            cap_row = caps.reindex(caps.index.union([pd.Timestamp(day)])).ffill().loc[
                pd.Timestamp(day)]
        ctx = WeightingContext(
            asof=day,
            market_caps=cap_row[active] if cap_row is not None else None,
            free_float=cap_row[active] if cap_row is not None else None,
            revenue=None,
            custom_weights=config.custom_weights,
        )
        return weighting.weights(active, ctx)

    def _active(self, day, tickers, last_price):
        return [t for t in self.sector.active_tickers(day)
                if t in tickers and not np.isnan(last_price[t]) and last_price[t] > 0]

    def _deploy_cash(self, day, tickers, shares, last_price, caps, config, weighting,
                     cost_rate, buys, trades, weights_log):
        active = self._active(day, tickers, last_price)
        if not active:
            return
        weights = self._weights_at(day, active, caps, config, weighting)
        weights_log[day] = weights
        cash = self._cash
        cost = cash * cost_rate
        invest = cash - cost
        for t in active:
            basis = weights.get(t, 0.0) * cash
            sh = weights.get(t, 0.0) * invest / last_price[t]
            if sh > 0:
                shares[t] += sh
                buys[t] += basis
                trades.append({"date": day, "ticker": t, "action": "buy",
                               "value": basis})
        self._cash = 0.0

    def _full_rebalance(self, day, tickers, shares, last_price, caps, config, weighting,
                        cost_rate, buys, sells, trades, weights_log):
        active = self._active(day, tickers, last_price)
        if not active:
            return
        total = self._cash + self._reserve + sum(
            shares[t] * last_price[t]
            for t in tickers if not np.isnan(last_price[t]))
        weights = self._weights_at(day, active, caps, config, weighting)
        weights_log[day] = weights
        current = {t: shares[t] * last_price[t] if not np.isnan(last_price[t]) else 0.0
                   for t in tickers}
        target = {t: weights.get(t, 0.0) * total for t in tickers}
        turnover = sum(abs(target[t] - current[t]) for t in tickers)
        cost = turnover * cost_rate
        invest_total = total - cost
        self._cash = 0.0
        self._reserve = 0.0
        for t in tickers:
            tv = weights.get(t, 0.0) * invest_total
            delta = tv - current[t]
            if delta > 1e-9:
                buys[t] += delta
                trades.append({"date": day, "ticker": t, "action": "rebalance_buy",
                               "value": delta})
            elif delta < -1e-9:
                sells[t] += -delta
                trades.append({"date": day, "ticker": t, "action": "rebalance_sell",
                               "value": -delta})
            shares[t] = tv / last_price[t] if last_price[t] > 0 else 0.0

    def _ccy(self, ticker: str) -> str:
        explicit = self.sector.currency_of(ticker)
        return explicit or infer_currency(ticker, default=self._base)

    def _convert(self, series, ccy, index):
        if self.fx is None or ccy == self._base:
            return series
        rate = self.fx.rate(ccy, self._base, index)
        return series * rate.reindex(index).ffill().bfill()

    def _price_panels(self, tickers, config, currencies):
        raw = self.provider.get_prices(tickers, config.start, config.end)
        price_cols, div_cols = {}, {}
        for t, df in raw.items():
            if df is None or df.empty:
                continue
            df = df.loc[str(config.start):str(config.end)]
            ccy = currencies.get(t, self._base)
            price_cols[t] = self._convert(df["price"], ccy, df.index)
            raw_div = df.get("dividend", pd.Series(0.0, index=df.index))
            div_cols[t] = self._convert(raw_div, ccy, df.index)
        if not price_cols:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        price = pd.DataFrame(price_cols).sort_index()
        div = pd.DataFrame(div_cols).reindex(price.index).fillna(0.0)
        return price, div, price.ffill()

    def _caps_panel(self, tickers, config, weighting, currencies):
        if weighting.method not in (WeightingMethod.MARKET_CAP,
                                    WeightingMethod.FREE_FLOAT):
            return None
        caps = self.provider.get_market_caps(tickers, config.start, config.end)
        if caps is None or caps.empty:
            return None
        caps.index = pd.to_datetime(caps.index)
        caps = caps.sort_index()
        for t in caps.columns:
            caps[t] = self._convert(caps[t], currencies.get(t, self._base), caps.index)
        return caps

    def _schedule(self, calendar, freq, skip_first) -> set:
        cal = pd.DatetimeIndex(calendar)
        dates = set()
        if freq is RebalanceFrequency.NONE:
            return {cal[0]} if not skip_first else dates
        if not skip_first:
            dates.add(cal[0])
        for b in pd.date_range(cal[0], cal[-1], freq=_FREQ[freq]):
            pos = cal.searchsorted(b, side="left")
            if pos < len(cal):
                dates.add(cal[pos])
        if skip_first:
            dates.discard(cal[0])
        return dates

    def _benchmark(self, config):
        if not config.benchmark:
            return None
        raw = self.provider.get_prices([config.benchmark], config.start, config.end)
        df = raw.get(config.benchmark)
        if df is None or df.empty:
            return None
        ccy = infer_currency(config.benchmark, default=self._base)
        series = self._convert(df["adj_close"].dropna(), ccy, df.index).dropna()
        if series.empty:
            return None
        return config.initial_capital * series / series.iloc[0]

    _cash: float = 0.0
    _reserve: float = 0.0
    _contributed: float = 0.0
    _base: str = "BRL"
