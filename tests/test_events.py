
import numpy as np
import pandas as pd

from src.data.prices import get_stock_history
from src.indicators.features import full_features
from src.events.technical_events import (
    detect_events, dedupe, quake_flags, event_table, all_events, DIRECTION, K2_APPLICABLE, combined, states,
)


def _frame(**over):
    n = 6
    base = dict(Close=100.0, bb_lower=90.0, bb_upper=110.0, rsi_14=50.0, st_dir=1.0,
                kama_slope=1.0, cloud_state=1.0, resistance=np.nan, support=np.nan)
    base.update(over)
    return pd.DataFrame(base, index=pd.date_range("2024-01-01", periods=n))


def test_rsi_exit_detected_once():
    f = _frame(rsi_14=[25, 28, 31, 35, 50, 50])
    ev = detect_events(f)
    assert ev["rsi_oversold_exit"].tolist() == [False, False, True, False, False, False]
    assert ev.sum().sum() == 1


def test_dedupe_gap():
    m = pd.Series([True, True, True, False, False, False, False, False, False, False, True])
    assert dedupe(m, gap=5).to_numpy().nonzero()[0].tolist() == [0, 10]


def test_events_no_lookahead():
    f = full_features("THYAO")
    full = detect_events(f)
    for cut in (400, 700, 1000):
        part = detect_events(f.iloc[: cut + 1])
        assert (part.iloc[-1] == full.iloc[cut]).all()
        for col in full.columns:
            assert dedupe(part[col]).iloc[-1] == dedupe(full[col]).iloc[cut]


def test_forward_return_manual():
    t = event_table("THYAO")
    row = t[t["fwd_5"].notna()].iloc[0]
    c = get_stock_history("THYAO")["Close"]
    pos = c.index.get_loc(row.name)
    assert np.isclose(row["fwd_5"], c.iloc[pos + 5] / c.iloc[pos] - 1)


def test_quake_flags():
    q = quake_flags(get_stock_history("ASELS").index)
    assert q.loc["2023-02-03"] and q.loc["2023-02-20"]
    assert not q.loc["2023-01-02"] and not q.loc["2023-02-27"] and not q.loc["2024-06-03"]


def test_all_events_coverage():
    ev = all_events()
    assert ev["ticker"].nunique() == 30
    counts = ev["event"].value_counts()
    for name in DIRECTION:
        assert counts.get(name, 0) >= 30, name
    assert set(ev["direction"]) == {1, -1}

# ---------- Adım 2: olay anındaki gösterge durumları ve K1/K2 ----------
def test_states_no_lookahead():
    f = full_features("ASELS")
    full = states(f)
    for cut in (400, 700, 1000):
        part = states(f.iloc[: cut + 1]).iloc[-1]
        row = full.iloc[cut]
        for c in full.columns:
            assert (part[c] == row[c]) or (pd.isna(part[c]) and pd.isna(row[c])), (cut, c)


def test_rsi_state_uses_last_three_days():
    f = pd.DataFrame({"rsi_14": [50, 34, 40, 45, 50, 66, 60], "kama": 100.0, "st_dir": 1.0, "cloud_state": 1.0,
                      "vol_z_20": 0.0}, index=pd.date_range("2024-01-01", periods=7))
    assert states(f)["rsi_state"].tolist() == ["bilinmiyor", "bilinmiyor", "dip", "dip", "notr", "tepe", "tepe"]


def test_combined_conditions_mirror():
    st = pd.DataFrame({"rsi_state": ["dip", "tepe", "dip", "notr"], "vol_recent": [True, True, False, True],
                       "kama_turn": [0.5, -0.5, 0.5, -1.0], "st_state": ["asagi", "yukari", "yukari", "asagi"],
                       "cloud_pos": ["altinda", "ustunde", "altinda", "altinda"]})
    bull, bear = combined(1, st), combined(-1, st)
    assert bull["conf_score"].tolist() == [3, 1, 2, 1] and bear["conf_score"].tolist() == [1, 3, 0, 2]
    assert bull["k1_confirmed"].tolist() == [True, False, True, False]
    assert bull["k1_all3"].tolist() == [True, False, False, False]
    assert bull["k2_against_trend"].tolist() == [True, False, False, True]
    assert bear["k2_against_trend"].tolist() == [False, True, False, False]


def test_k2_impossible_for_supertrend_and_cloud_events():
    ev = all_events()
    other = ev[~ev["event"].isin(K2_APPLICABLE)]
    assert set(other["event"]) == {"supertrend_up", "supertrend_down", "cloud_break_up", "cloud_break_down"}
    assert not other["k2_against_trend"].any()
