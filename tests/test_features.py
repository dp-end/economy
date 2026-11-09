
import numpy as np
import pandas as pd

from src.data.prices import get_stock_history, get_index_history
from src.indicators.support_resistance import sr_features
from src.indicators.relative import sector_lag, relative_features
from src.indicators.features import full_features


def test_sector_lag_manual():
    idx = pd.date_range("2024-01-01", periods=3)
    panel = pd.DataFrame({"X": [100, 110, 121.0], "Y": [100, 100, 100.0], "Z": [100, 120, 144.0]}, index=idx)
    lag = sector_lag(panel, {"G": ["X", "Y", "Z"]}, 1)
    # 2. gün getirileri: X=0.10, Y=0.0, Z=0.20
    assert np.isclose(lag.loc[idx[1], "Y"], -0.15)   # 0.0 - median(0.10, 0.20)
    assert np.isclose(lag.loc[idx[1], "X"], 0.0)     # 0.10 - median(0.0, 0.20)


def test_rel_xu_manual():
    xu = get_index_history()["Close"]
    st = get_stock_history("THYAO")["Close"]
    f = relative_features("THYAO")
    expect = (st.iloc[100] / st.iloc[95] - 1) - (xu.iloc[100] / xu.iloc[95] - 1)
    assert np.isclose(f.iloc[100]["rel_xu_5"], expect)


def test_sr_sanity():
    df = get_stock_history("ASELS")
    sr = sr_features(df)
    s = sr["support"].dropna()
    r = sr["resistance"].dropna()
    assert (s < df.loc[s.index, "Close"]).all()
    assert (r > df.loc[r.index, "Close"]).all()
    assert (sr["support_touches"].dropna() >= 1).all()
    assert len(s) > 500


def test_sr_no_lookahead():
    df = get_stock_history("ASELS")
    full = sr_features(df)
    for cut in (400, 700, 1000):
        part = sr_features(df.iloc[: cut + 1])
        a = full.iloc[cut].to_numpy(dtype=float)
        b = part.iloc[-1].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), f"look-ahead: {cut}"


def test_full_features():
    f = full_features("ASELS")
    assert len(f) == 1258
    for c in ["rsi_14", "support", "rel_xu_5", "sector_lag_20", "cloud_state"]:
        assert c in f.columns