import numpy as np
import pandas as pd

from src.data.prices import get_stock_history
from src.indicators.technical import (
    add_all, sma, rsi, bollinger, kama, supertrend, atr, ichimoku,
)


def test_sma_manual():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert sma(s, 3).iloc[-1] == 5.0


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1.0, 60))
    assert rsi(up).dropna().iloc[-1] == 100.0
    df = get_stock_history("ASELS")
    r = rsi(df["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_bollinger_order():
    b = bollinger(get_stock_history("ASELS")["Close"]).dropna()
    assert (b["bb_lower"] < b["bb_mid"]).all() and (b["bb_mid"] < b["bb_upper"]).all()
    assert (b["bb_bandwidth"] > 0).all()


def test_kama_constant_series():
    s = pd.Series([10.0] * 50)
    assert np.allclose(kama(s).dropna(), 10.0)


def test_atr_positive():
    assert (atr(get_stock_history("ASELS")).dropna() > 0).all()


def test_supertrend_direction_values():
    st = supertrend(get_stock_history("ASELS"))
    assert set(st["st_dir"].dropna().unique()) <= {1.0, -1.0}
    assert set(st["st_flip"].unique()) <= {-1.0, 0.0, 1.0}


def test_ichimoku_warmup():
    ich = ichimoku(get_stock_history("ASELS"))
    assert ich["cloud_state"].iloc[:77].isna().all()   # 52 + 26 - 1 gün ısınma
    assert ich["cloud_state"].iloc[77:].notna().all()


def test_no_lookahead():
    """t gününe kadar kesilmiş veriyle hesaplanan değer, tam veriyle hesaplananla aynı olmalı."""
    df = get_stock_history("ASELS")
    full = add_all(df)
    for cut in (300, 600, 900, 1200):
        part = add_all(df.iloc[: cut + 1])
        a = full.iloc[cut].to_numpy(dtype=float)
        b = part.iloc[-1].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), f"look-ahead: {cut}"