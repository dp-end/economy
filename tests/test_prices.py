
import numpy as np
import pandas as pd
import pytest

from src.data.prices import get_stock_history, get_index_history
from src.data.universe import load_universe
import src.data.yahoo_prices as yp


def test_stock_clean():
    df = get_stock_history("ASELS")
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert (df["Close"] > 0).all()
    assert df.index.max() < pd.Timestamp("2026-11-09")


def test_index_calendar_matches_stocks():
    xu = get_index_history()
    assert len(xu) == 1258
    for t in load_universe()["ticker"]:
        assert get_stock_history(t).index.equals(xu.index), t


def test_unknown_ticker():
    with pytest.raises(ValueError):
        get_stock_history("XXXXX")


def test_no_fake_jumps_and_splits_adjusted():
    for t in load_universe()["ticker"]:
        r = get_stock_history(t)["Close"].pct_change().abs()
        assert r.max() < 0.11, (t, r.idxmax(), r.max())  # BIST günlük limit %10
    assert abs(get_stock_history("CCOLA")["Close"].pct_change().loc["2024-08-01"]) < 0.10


def test_special_days():
    idx = get_index_history().index
    assert pd.Timestamp("2023-02-08") not in idx   # borsa kapalı, Yahoo'da sahte satır
    assert pd.Timestamp("2024-04-09") in idx       # arife, yarım gün işlem
    assert pd.Timestamp("2024-12-25") in idx       # normal işlem günü


def test_sasa_volume_complete():
    v = get_stock_history("SASA")["Volume"]
    assert v.notna().all() and (v > 0).all()


def test_high_low_bracket_close():
    df = get_stock_history("TURSG")
    assert (df["High"] >= df["Close"]).all() and (df["Low"] <= df["Close"]).all()


def test_agrees_with_yahoo_on_most_days():
    a = get_stock_history("ASELS")["Close"]
    y = yp.get_stock_history("ASELS")["Close"]
    both = a.index.intersection(y.index)
    corr = a.loc[both].pct_change().corr(y.loc[both].pct_change())
    assert corr > 0.98