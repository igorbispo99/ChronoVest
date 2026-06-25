# ChronoVest

Historical sector portfolio backtesting, framed for a Brazilian investor.
Everything is reported in **BRL** and compared against **CDI, IPCA and Ibovespa**.

Answers questions like: *"What if I had put R$10,000 into Brazilian banks in 2016
and added R$1,000 every month, always buying the currently-balanced basket? How
would that compare to leaving the money in the CDI, or in Ibovespa?"*

## Strategy model (no selling)

The investor **never sells**. Capital deployed at the start, plus every periodic
contribution (aporte), buys the *currently balanced* basket of the sector at that
moment: the engine asks "which companies are in the sector now and how big is
each", derives weights, and **buys only**. Existing positions are held and
winners are left to run. Dividends are reinvested into the paying stock (DRIP) or
kept as idle cash. Names that delist are liquidated by the corporate action and
the proceeds are redeployed at the next buy.

Classic sell-to-target rebalancing is still available via `allow_selling=True`.

## Two modes (base currency always BRL)

| Mode | Universe | Currency handling | Benchmarks |
|------|----------|-------------------|------------|
| **Local** | B3 / Brazilian assets | none (already BRL) | CDI, IPCA, Ibovespa |
| **International** | foreign assets | converted to BRL (FX is part of the return) | CDI, IPCA, Ibovespa **+ native index** (S&P 500 / Nasdaq, in BRL) |

The two modes never mix currencies inside one portfolio: a sector is either
local (BRL-native) or international (single foreign market converted to BRL).

## Returns done right with contributions

With monthly aportes a naive "final / initial" return is meaningless, so the
engine reports:

- **equity** — actual portfolio value (money in).
- **TWR** — time-weighted return index; the strategy's intrinsic performance with
  cash flows removed. CAGR / Sharpe use this.
- **Money-weighted return (XIRR)** — the investor's actual annualised return given
  when each aporte landed.
- **% do CDI**, **real return (deflated by IPCA)**, **Sharpe vs CDI**.

Benchmark curves (CDI, Ibovespa, native index) are simulated with the *same*
contribution schedule, so the comparison is apples-to-apples.

## Architecture

```
config.py                       BacktestConfig + Market enum
        │
data/    DataProvider ──────────┤ yfinance adapter + parquet cache
         CurrencyConverter ─────┤ FX (yfinance / static / identity)
         BrazilIndicators ──────┤ CDI + IPCA from the Banco Central (BCB SGS API)
universe/ Sector ───────────────┤ point-in-time membership, currency, market tag
engine/  WeightingStrategy ─────┤ market-cap / equal / custom / free-float / revenue
         PortfolioEngine ───────┤ no-sell buying, aportes, dividends, costs, FX, TWR
analytics/ metrics + brazil ────┤ CAGR/vol/Sharpe/drawdown; CDI, IPCA, XIRR, DCA
viz/     charts ────────────────┤ plotly (benchmarks, real vs nominal, allocation)
forecast/ Forecaster ───────────┤ Monte Carlo projection (GBM / block bootstrap)
runner.py ──────────────────────┤ ties mode + benchmarks + report together
app/     streamlit_app ─────────┘ mode toggle, aporte inputs, Brazilian dashboard
```

## Sectors (`data/sectors/`)

Local (B3, BRL):
`brazil_banks`, `brazil_mining_steel`, `brazil_utilities_energy`,
`br_technology` (incl. Sinqia SQIA3, delisted 2023-11-01), `br_ai` (proxy basket).

International (foreign → BRL):
`intl_technology`, `intl_ai`, `intl_banking`, `watchmaking` (CHF/JPY/USD).

Listing/delisting dates are encoded as membership `start`/`end` windows; IPOs are
handled automatically because the engine only buys names that have prices.

## Install

```bash
pip install -r requirements.txt
# or: pip install -e ".[app,dev]"
```

## Run

```bash
python examples/run_local_br.py        # B3 sector, BRL, monthly aporte
python examples/run_international.py    # US tech -> BRL, vs S&P 500 in BRL
streamlit run src/chronovest/app/streamlit_app.py
```

## Programmatic use

```python
from datetime import date
from chronovest.config import BacktestConfig, Market, RebalanceFrequency, WeightingMethod
from chronovest.data.yfinance_provider import YFinanceProvider
from chronovest.data.brazil_indicators import BcbIndicators
from chronovest.runner import run_backtest
from chronovest.universe.sector import load_sector

sector = load_sector("data/sectors/brazil_banks.yaml")
config = BacktestConfig(
    sector=sector.name, start=date(2016, 1, 1), end=date(2024, 12, 31),
    initial_capital=10_000, base_currency="BRL", market=Market.LOCAL,
    weighting=WeightingMethod.MARKET_CAP,
    contribution_amount=1_000, contribution_frequency=RebalanceFrequency.MONTHLY,
    reinvest_dividends=True, transaction_cost_bps=15, benchmark="^BVSP",
)
full = run_backtest(YFinanceProvider(), sector, config, fx=None, indicators=BcbIndicators())
print(full.report.as_dict())          # nominal/real CAGR, % do CDI, XIRR, ...
```

For international mode pass `market=Market.INTERNATIONAL` and
`fx=YFinanceConverter()`.

## Tests

```bash
python -m pytest    # 60 tests, fully offline (synthetic data, static FX & indicators)
```

## Prediction / forecast mode

Equity prices are close to a random walk, so the forecast is **probabilistic**, not
a single number. A `Forecaster` learns the historical return distribution and
simulates thousands of future paths; the **lower and upper bounds are percentiles
across those paths** (a fan chart). Future aportes can be continued along every
path, so the projection answers "where might this portfolio be in N years, and how
wide is the uncertainty?".

Two engines:
- **Block bootstrap** (default) — non-parametric; resamples contiguous blocks of
  historical returns, preserving volatility clustering and fat tails.
- **GBM** — parametric geometric Brownian motion (lognormal returns).

```python
from chronovest.forecast import forecast_from_result, BlockBootstrapForecaster

fc = forecast_from_result(full.result, horizon_years=5,
                          forecaster=BlockBootstrapForecaster(),
                          confidence=0.90, n_paths=3000, seed=42)
print(fc.terminal)   # median, lower, upper, prob_above_invested, ...
```

`python examples/run_forecast.py` prints a 5-year median and 90% band; the Streamlit
app shows the fan chart under "Projecao / Forecast".

This is a projection of historical statistics, not a guarantee of future returns,
and is not financial advice.

## Data caveats (free sources)

- **CDI / IPCA** come from the BCB SGS open API (series 12 and 433); the app
  fetches them at runtime.
- **Market cap** is approximated as `shares_outstanding x price`.
- **Survivorship bias**: encode delisted names in the sector YAML with `end`
  dates; the engine liquidates them correctly.
- **% do CDI** and CAGR are computed on the time-weighted index, so a portfolio
  can show a positive XIRR yet a sub-CDI TWR (or vice-versa) when contributions
  are timed favourably/unfavourably — both numbers are shown on purpose.
- **`br_ai`** is a thematic proxy: Brazil has no pure-play listed AI companies.

## Roadmap

Taxes (15%/20% on Brazilian equities), free-float factors, revenue weighting from
fundamentals, and a point-in-time paid data provider each land in a single layer
behind the existing interfaces.
