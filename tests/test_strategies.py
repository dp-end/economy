
import numpy as np
import pandas as pd

from src.backtest.engine import load_returns
from src.backtest.strategies import (TOP_K, WARMUP, evaluate, generate_strategy_variants, macro_scale,
                                     panels, pick_top, weekly)


def test_weekly_holds_last_decision_of_week():
    idx = pd.bdate_range("2024-01-01", periods=10)                    # Pzt 1 Ocak - Cum 12 Ocak
    w = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=idx)
    out = weekly(w)["A"].tolist()
    assert out == [0, 0, 0, 0, 4, 4, 4, 4, 4, 9]                      # Cuma kararı sonraki Cumaya kadar taşınır


def test_pick_top():
    idx = pd.bdate_range("2024-01-01", periods=1)
    score = pd.DataFrame([[5.0, 4.0, 3.0, 2.0, np.nan]], index=idx, columns=list("ABCDE"))
    ok = pd.DataFrame([[True, False, True, True, True]], index=idx, columns=list("ABCDE"))
    w = pick_top(ok, score, k=2)
    assert w.iloc[0].to_dict() == {"A": 0.5, "B": 0.0, "C": 0.5, "D": 0.0, "E": 0.0}   # B uygun değil, E skorsuz
    one = pick_top(ok & (score > 4.5), score, k=2)
    assert one.iloc[0].sum() == 0.5                                   # tek hisse: yarısı nakit
    vol = pd.DataFrame([[1.0, 1.0, 3.0, 1.0, 1.0]], index=idx, columns=list("ABCDE"))
    wv = pick_top(ok, score, k=2, weight_by=vol)
    assert np.isclose(wv.loc[idx[0], "A"], 0.25) and np.isclose(wv.loc[idx[0], "C"], 0.75)


def test_weights_valid_and_weekly():
    ws = generate_strategy_variants()
    assert set(ws) == {"A_teknik", "B_sektor", "C_temel", "D_makro", "D_makro_oynaklik", "D_makro_faiz"}
    for name, w in ws.items():
        assert (w >= 0).all().all() and (w.sum(axis=1) <= 1 + 1e-9).all(), name
        assert ((w > 0).sum(axis=1) <= TOP_K).all(), name
        changed = (w.diff().abs().sum(axis=1) > 0)
        week = w.index.to_series().dt.to_period("W")
        assert (week[changed] != week.shift(-1)[changed]).all(), name    # sadece haftanın son gününde değişir
    assert (ws["D_makro"] <= ws["C_temel"] + 1e-12).all().all()          # makro sadece azaltır
    assert set(np.unique(macro_scale(ws["A_teknik"].index))) <= {0.5, 1.0}


def test_filters_respected():
    p, ws = panels(), generate_strategy_variants()
    week = p["close"].index.to_series().dt.to_period("W")
    reb = (week != week.shift(-1)).to_numpy()                            # dengeleme günleri
    held_b = (ws["B_sektor"] > 0)[reb]
    assert not (held_b & p["shock"][reb]).any().any()                    # şoklu hisse alınmaz
    held_c = (ws["C_temel"] > 0)[reb]
    assert not (held_c & p["weak_earnings"][reb]).any().any()            # bilançosu alt üçte birde olan alınmaz
    held_a = (ws["A_teknik"] > 0)[reb]
    assert not (held_a & (p["st_dir"][reb] <= 0)).any().any()


def test_no_lookahead_in_weights():
    """Bugünden sonraki fiyatlar bugünkü kararı değiştirmemeli: gelecekteki bilanço bugünkü filtreye girmez."""
    p = panels()
    ev_day, t = pd.Timestamp("2026-10-28"), "ASELS"                      # ASELS 2026/9 bilançosu bu gün görünür oldu
    before = p["weak_earnings"].loc[:ev_day - pd.Timedelta(days=1), t]
    from src.events.earnings_reaction import earnings_events
    ev = earnings_events()
    prev = ev[(ev["ticker"] == t) & (ev["event_day"] < ev_day)].sort_values("event_day").iloc[-1]
    assert before.iloc[-1] == (prev["tercile"] == "alt")                # 27 Ekim'de hâlâ önceki çeyrek geçerli


def test_evaluate_shape_and_costs():
    res = evaluate()
    assert len(res) == 6 * 3 and len(load_returns()) - WARMUP == 1008
    free, high = evaluate(cost=0.0), evaluate(cost=0.003)
    a = ("A_teknik", "tum")
    assert free.loc[a, "toplam_getiri"] > res.loc[a, "toplam_getiri"] > high.loc[a, "toplam_getiri"]
    assert (res["islem_sayisi"] > 30).all() and (res["uyari"] == "").all()