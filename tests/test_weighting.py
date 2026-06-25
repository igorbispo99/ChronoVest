from datetime import date

import pandas as pd
import pytest

from chronovest.config import WeightingMethod
from chronovest.engine.weighting import WeightingContext, build_weighting


def _ctx(**kw):
    return WeightingContext(asof=date(2020, 1, 1), **kw)


def test_equal_weighting_sums_to_one():
    w = build_weighting(WeightingMethod.EQUAL).weights(["A", "B", "C"], _ctx())
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1 / 3) for v in w.values())


def test_market_cap_weighting_proportional():
    caps = pd.Series({"A": 300.0, "B": 100.0})
    w = build_weighting(WeightingMethod.MARKET_CAP).weights(["A", "B"], _ctx(market_caps=caps))
    assert w["A"] == pytest.approx(0.75)
    assert w["B"] == pytest.approx(0.25)


def test_market_cap_falls_back_to_equal_when_missing():
    w = build_weighting(WeightingMethod.MARKET_CAP).weights(["A", "B"], _ctx())
    assert w["A"] == pytest.approx(0.5)


def test_custom_weighting_normalizes():
    w = build_weighting(WeightingMethod.CUSTOM).weights(
        ["A", "B"], _ctx(custom_weights={"A": 2, "B": 2})
    )
    assert w["A"] == pytest.approx(0.5)


def test_empty_universe():
    assert build_weighting(WeightingMethod.EQUAL).weights([], _ctx()) == {}
