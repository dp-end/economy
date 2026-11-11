
import numpy as np
import pandas as pd

from src.data.prices import get_index_history, get_stock_history
from src.events.earnings_reaction import (GROUPS, WINDOWS, add_terciles, car, earnings_events,
                                          quarterly_spread, spread_tests)


def test_car_hand_computed():
    close = pd.Series([100, 110, 121, 121, 133.1])
    xu = pd.Series([100, 100, 110, 110, 110.0])
    assert np.isclose(car(close, xu, 2, 0, 0), 0.10 - 0.10)          # gün 2: hisse +%10, endeks +%10
    assert np.isclose(car(close, xu, 1, 0, 1), 0.21 - 0.10)          # gün 1-2
    assert np.isclose(car(close, xu, 2, -1, -1), 0.10 - 0.0)         # olaydan önceki gün
    assert np.isnan(car(close, xu, 0, 0, 0))                         # önceki kapanış yok
    assert np.isnan(car(close, xu, 3, 0, 4))                         # pencere verinin dışına taşıyor


def test_terciles_within_quarter():
    q1, q2 = pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")
    df = pd.DataFrame({"ticker": [f"T{i}" for i in range(18)] * 2,
                       "period_end": [q1] * 18 + [q2] * 18,
                       "ni_yoy": list(range(18)) + [x * 100 for x in range(17)] + [np.nan]})
    out = add_terciles(df)
    a = out[out["period_end"] == q1]
    assert a["tercile"].value_counts().to_dict() == {"alt": 6, "orta": 6, "ust": 6}
    assert a.sort_values("ni_yoy")["tercile"].tolist() == ["alt"] * 6 + ["orta"] * 6 + ["ust"] * 6
    b = out[out["period_end"] == q2]
    assert b["tercile"].isna().sum() == 1                             # ni_yoy boş -> sıralama yok
    # ölçek önemsiz: ikinci çeyrek 100 kat büyük ama sıralama kendi içinde
    assert b.sort_values("ni_yoy")["tercile"].iloc[0] == "alt" and b.sort_values("ni_yoy")["tercile"].iloc[16] == "ust"
    few = add_terciles(df[df["ticker"].isin(["T0", "T1", "T2"])])
    assert few["tercile"].isna().all()                                # çeyrekte çok az hisse


def test_real_event_matches_prices():
    ev = earnings_events()
    r = ev[(ev["ticker"] == "ASELS") & (ev["period_end"] == "2026-09-30")].iloc[0]
    assert r["event_day"] == pd.Timestamp("2026-10-28")               # 27 Ekim 18:34'te yayınlandı
    c, x = get_stock_history("ASELS")["Close"], get_index_history("XU100")["Close"]
    hand = (c["2026-10-28"] / c["2026-10-27"] - 1) - (x["2026-10-28"] / x["2026-10-27"] - 1)
    assert np.isclose(r["d0"], hand)
    pre = (c["2026-10-27"] / c["2026-10-20"] - 1) - (x["2026-10-27"] / x["2026-10-20"] - 1)
    assert np.isclose(r["pre5"], pre)                                 # ön pencere olay gününü içermez


def test_real_table_and_spread():
    ev = earnings_events()
    assert len(ev) == 545 and not ev.duplicated(["ticker", "period_end"]).any()
    assert set(ev["tercile"].dropna()) == set(GROUPS)
    has = ev["event_day"].notna()
    assert ev.loc[has, "d0"].notna().all() and ev.loc[~has, list(WINDOWS)].isna().all().all()
    assert ev.loc[has, "d0"].abs().max() < 0.25                       # tek günde hisse ±%10, endeks sınırlı
    sp = quarterly_spread(ev)
    assert 15 <= len(sp) <= 19 and list(sp.columns) == list(WINDOWS)
    t = spread_tests(sp)
    assert ((t["p"] >= 0) & (t["p"] <= 1)).all() and (t["q_bh"] >= t["p"] - 1e-12).all()