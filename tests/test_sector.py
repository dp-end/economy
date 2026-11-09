
import numpy as np
import pandas as pd

from src.data.prices import get_stock_history
from src.events.technical_events import dedupe
from src.events.sector_catchup import (
    quake_flags_wide, lag_triggers, sector_events, _context, CONFIGS, EVENT_GAP,
)


def test_quake_flags_wide():
    q = quake_flags_wide(get_stock_history("ASELS").index)
    assert q.loc["2023-01-20"] and q.loc["2023-02-15"] and q.loc["2023-03-20"]
    assert not q.loc["2023-01-02"] and not q.loc["2023-04-10"] and not q.loc["2024-06-03"]


def test_lag_triggers_manual():
    idx = pd.date_range("2024-01-01", periods=4)
    panel = pd.DataFrame({"X": [100.0] * 4, "Y": [100, 100, 90, 90.0], "Z": [100.0] * 4}, index=idx)
    first, lag = lag_triggers(panel, {"G": ["X", "Y", "Z"]}, 1, 0.04, "laggard")
    assert first["Y"].tolist() == [False, False, True, False]
    assert not first["X"].any() and not first["Z"].any()
    first_l, _ = lag_triggers(panel, {"G": ["X", "Y", "Z"]}, 1, 0.04, "leader")
    assert first_l["X"].tolist() == [False, False, True, False]


def test_triggers_no_lookahead():
    c = _context()
    for n, thr in CONFIGS.values():
        full, _ = lag_triggers(c["panel"], c["groups"], n, thr, "laggard")
        for cut in (400, 700, 1000):
            part, _ = lag_triggers(c["panel"].iloc[: cut + 1], c["groups"], n, thr, "laggard")
            assert (part.iloc[-1] == full.iloc[cut]).all()
            for t in ("ASELS", "SASA", "TURSG"):
                assert dedupe(part[t], EVENT_GAP).iloc[-1] == dedupe(full[t], EVENT_GAP).iloc[cut]


def test_forward_gap_manual():
    ev = sector_events()
    row = ev[ev["gap_10"].notna()].iloc[0]
    t, d = row["ticker"], row.name
    c = _context()
    members = c["groups"][c["group_of"][t]]

    def f10(x):
        s = get_stock_history(x)["Close"]
        pos = s.index.get_loc(d)
        return s.iloc[pos + 10] / s.iloc[pos] - 1

    expect = f10(t) - np.median([f10(m) for m in members if m != t])
    assert np.isclose(row["gap_10"], expect)


def test_coverage():
    ev = sector_events()
    c = _context()
    assert set(ev["ticker"]) <= set(c["core"])
    assert "ECILC" not in set(ev["ticker"]) and "MPARK" not in set(ev["ticker"])   # Sağlık: 2 hisse
    for cfg in CONFIGS:
        for side in ("laggard", "leader"):
            assert ((ev["config"] == cfg) & (ev["side"] == side)).sum() >= 30