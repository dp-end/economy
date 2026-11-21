
import numpy as np
import pandas as pd

import src.events.earnings_reaction as er
from src.data.pit_fundamentals import panel
from src.data.prices import get_index_history, get_stock_history
from src.events.earnings_reaction import (GROUPS, MIN_STOCKS, WINDOWS, car, earnings_events, quarterly_spread,
                                          spread_tests, terciles_expanding, terciles_expost, terciles_prior)


def test_car_hand_computed():
    close = pd.Series([100, 110, 121, 121, 133.1])
    xu = pd.Series([100, 100, 110, 110, 110.0])
    assert np.isclose(car(close, xu, 2, 0, 0), 0.10 - 0.10)          # gün 2: hisse +%10, endeks +%10
    assert np.isclose(car(close, xu, 1, 0, 1), 0.21 - 0.10)          # gün 1-2
    assert np.isclose(car(close, xu, 2, -1, -1), 0.10 - 0.0)         # olaydan önceki gün
    assert np.isnan(car(close, xu, 0, 0, 0))                         # önceki kapanış yok
    assert np.isnan(car(close, xu, 3, 0, 4))                         # pencere verinin dışına taşıyor


def test_terciles_expost_within_quarter():
    """Eski yöntem (yalnızca karşılaştırma için): çeyreğin bütün hisseleri arasında sıralama."""
    q1, q2 = pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")
    df = pd.DataFrame({"ticker": [f"T{i}" for i in range(18)] * 2,
                       "period_end": [q1] * 18 + [q2] * 18,
                       "ni_yoy": list(range(18)) + [x * 100 for x in range(17)] + [np.nan]})
    out = terciles_expost(df)
    a = out[out["period_end"] == q1]
    assert a["tercile"].value_counts().to_dict() == {"alt": 6, "orta": 6, "ust": 6}
    assert a.sort_values("ni_yoy")["tercile"].tolist() == ["alt"] * 6 + ["orta"] * 6 + ["ust"] * 6
    b = out[out["period_end"] == q2]
    assert b["tercile"].isna().sum() == 1                             # ni_yoy boş -> sıralama yok
    assert b.sort_values("ni_yoy")["tercile"].iloc[0] == "alt" and b.sort_values("ni_yoy")["tercile"].iloc[16] == "ust"
    few = terciles_expost(df[df["ticker"].isin(["T0", "T1", "T2"])])
    assert few["tercile"].isna().all()                                # çeyrekte çok az hisse


def _season(n=18, q="2024-06-30", start="2024-07-20"):
    """Yapay bilanço sezonu: T0..T(n-1) sırayla birer gün arayla açıklar."""
    days = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "period_end": pd.Timestamp(q),
                         "event_day": days, "ni_yoy": np.linspace(-1, 1, n)})


def test_terciles_prior_uses_only_previous_quarter_known_values(monkeypatch):
    """A: eşikler bir önceki çeyrekten ve yalnızca olay gününe kadar bilinen değerlerden gelir."""
    df = _season()
    prev_q = pd.Timestamp("2024-03-31")
    # önceki çeyrek: 20 değer 0..19; 18'i 1 Temmuz'da, 2'si (çok büyük) 1 Eylül'de bilinir
    vals = list(range(18)) + [1000, 2000]
    known = [pd.Timestamp("2024-07-01")] * 18 + [pd.Timestamp("2024-09-01")] * 2
    pool = pd.DataFrame({"period_end": prev_q, "value": vals, "known": known})
    monkeypatch.setattr(er, "_pool", lambda col, tickers: pool)
    out = terciles_prior(df.assign(ni_yoy=[0.0, 5.0, 6.0, 11.0, 12.0, 17.0] * 3))
    lo, hi = pd.Series(range(18)).quantile([1 / 3, 2 / 3])          # geç bilinen 1000 ve 2000 havuza girmez
    assert np.allclose(out["tercile_lo"], lo) and np.allclose(out["tercile_hi"], hi)
    assert (out["tercile_n"] == 18).all() and (out["tercile_ref"] == prev_q).all()
    assert out["tercile"].tolist()[:6] == ["alt", "alt", "orta", "orta", "ust", "ust"]   # eşikler 5,67 ve 11,33
    # aynı çeyreğin başka hissesi ne açıklarsa açıklasın sınıf değişmez (geleceğe bakış yok)
    other = df.assign(ni_yoy=[0.0, 5.0, 6.0, 11.0, 12.0, 17.0] * 3)
    other.loc[other.index[-1], "ni_yoy"] = 1e6
    assert terciles_prior(other)["tercile"].iloc[0] == out["tercile"].iloc[0]
    small = pool.iloc[:MIN_STOCKS - 1]
    monkeypatch.setattr(er, "_pool", lambda col, tickers: small)
    assert terciles_prior(df)["tercile"].isna().all()                # havuz < 15 -> sınıf yok


def test_terciles_expanding_ranks_only_reported_companies():
    """B: aynı çeyrekte o güne kadar açıklayanlar arasında; ilk 14 açıklayan sıralanamaz."""
    out = terciles_expanding(_season())
    assert out["tercile"].iloc[:MIN_STOCKS - 1].isna().all() and out["tercile"].iloc[MIN_STOCKS - 1:].notna().all()
    assert out["tercile"].iloc[MIN_STOCKS - 1] == "ust"               # artan sırada: 15. açıklayan en yüksek değer


def test_real_terciles_have_no_lookahead():
    """Gerçek veride her sıralanmış olayın eşikleri, önceki çeyrekte olay gününe kadar bilinen değerlerden yeniden
    hesaplanabilir; aynı çeyreğin hiçbir değeri kullanılmaz."""
    ev = earnings_events()
    ranked = ev[ev["tercile"].notna()]
    assert len(ranked) > 300
    pans = {t: panel(t) for t in ev["ticker"].unique()}
    for _, r in ranked.sample(40, random_state=0).iterrows():
        ref = r["period_end"] - pd.offsets.QuarterEnd(1)
        assert r["tercile_ref"] == ref
        vals = [p.loc[ref, "ni_yoy"] for p in pans.values() if ref in p.index and pd.notna(p.loc[ref, "ni_yoy"])
                and p.loc[ref, "ni_yoy__known"] <= r["event_day"]]
        lo, hi = pd.Series(vals).quantile([1 / 3, 2 / 3])
        assert len(vals) == r["tercile_n"] and np.isclose(lo, r["tercile_lo"]) and np.isclose(hi, r["tercile_hi"])


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
    assert len(sp) >= 12 and list(sp.columns) == list(WINDOWS)
    # TMS 29 geçişi çevresinde (2022/4 - 2024/1) karşılaştırılabilir önceki çeyrek yok: bu çeyrekler test dışı
    assert not sp.index.to_series().between("2022-12-31", "2024-03-31").any()
    t = spread_tests(sp)
    assert ((t["p"] >= 0) & (t["p"] <= 1)).all() and (t["q_bh"] >= t["p"] - 1e-12).all()