
import pandas as pd
import pytest

from src.data.yahoo_prices import get_stock_history, get_index_history, market_calendar


def test_stock_clean():
    df = get_stock_history("ASELS")
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index.tz is None
    assert (df["Close"] > 0).all()
    assert df.index.max() < pd.Timestamp("2026-11-09")


def test_index_loads():
    assert len(get_index_history("XU100")) > 1000


def test_unknown_ticker():
    with pytest.raises(ValueError):
        get_stock_history("XXXXX")


def test_ccola_split_adjusted():
    df = get_stock_history("CCOLA")
    r = df["Close"].pct_change().abs().drop(pd.Timestamp("2023-02-15"), errors="ignore")
    assert r.max() < 0.15
    assert abs(df["Close"].pct_change().loc["2024-08-01"]) < 0.10


def test_placeholder_removed():
    assert pd.Timestamp("2024-04-09") not in market_calendar()
    assert pd.Timestamp("2024-04-09") not in get_index_history().index


def test_stocks_keep_real_trading_days():
    df = get_stock_history("THYAO")
    assert df.index.isin(market_calendar()).all()
    assert pd.Timestamp("2024-12-25") in df.index  # gerçek işlem günü


def test_gubrf_no_fake_jump():
    r = get_stock_history("GUBRF")["Close"].pct_change().abs()
    r = r.drop(pd.Timestamp("2023-02-15"), errors="ignore")
    assert r.max() < 0.12