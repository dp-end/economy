
import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (MIN_TRADES, benchmarks, calculate_performance, load_returns,
                                 max_drawdown, run_backtest, split_halves)

IDX = pd.bdate_range("2024-01-01", periods=4)
RET = pd.DataFrame({"A": [0.0, 0.10, -0.05, 0.02], "B": [0.0, 0.00, 0.10, 0.00]}, index=IDX)


def test_hand_computed_with_shift_and_cost():
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]}, index=IDX)
    bt = run_backtest(w, RET, cost=0.01)
    # gün 0 kararı gün 1'de taşınır: A +%10; gün 1 kararı gün 2: A -%5; gün 2 kararı gün 3: B %0
    assert np.allclose(bt["gross"], [0.0, 0.10, -0.05, 0.0])
    assert np.allclose(bt["turnover"], [0.0, 1.0, 0.0, 2.0])          # giriş 1, A'dan B'ye geçiş 2
    assert np.allclose(bt["net"], [0.0, 0.09, -0.05, -0.02])
    assert bt["entries"].tolist() == [0, 1, 0, 1]


def test_no_shift_is_lookahead():
    w = (RET > 0).astype(float)                                       # bugünün getirisini bilen sinyal
    cheat = run_backtest(w, RET, cost=0, shift=0)["gross"].sum()
    honest = run_backtest(w, RET, cost=0, shift=1)["gross"].sum()
    assert np.isclose(cheat, 0.22) and honest < cheat                 # hile: sadece artı günleri toplar


def test_weights_validation():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame({"A": [0.8] * 4, "B": [0.8] * 4}, index=IDX), RET)
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame({"A": [-0.5] * 4, "B": [0.5] * 4}, index=IDX), RET)


def test_performance_metrics():
    assert np.isclose(max_drawdown(pd.Series([0.10, -0.50, 0.20])), -0.50)
    w = pd.DataFrame({"A": [1.0] * 4, "B": [0.0] * 4}, index=IDX)
    bt = run_backtest(w, RET, cost=0)
    p = calculate_performance(bt, bench=RET["B"])
    assert np.isclose(p["toplam_getiri"], 1.10 * 0.95 * 1.02 - 1)
    assert np.isclose(p["kiyas_getiri"], 0.10) and np.isclose(p["fark"], p["toplam_getiri"] - 0.10)
    assert p["islem_sayisi"] == 1 and p["uyari"] == "az sinyal" and MIN_TRADES > 1
    a, b = split_halves(bt)
    assert len(a) == 2 and len(b) == 2


def test_real_data_lookahead_and_costs():
    ret = load_returns()
    bm = benchmarks(ret)
    assert ret.shape == (1258, 30) and list(bm.columns) == ["xu100", "esit_agirlik"]
    cheat = (ret > 0).astype(float)
    cheat = cheat.div(cheat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    wrong = calculate_performance(run_backtest(cheat, ret, shift=0))
    right = calculate_performance(run_backtest(cheat, ret, shift=1))
    free = calculate_performance(run_backtest(cheat, ret, cost=0.0))
    high = calculate_performance(run_backtest(cheat, ret, cost=0.005))
    assert wrong["yillik_getiri"] > 10 and wrong["yillik_getiri"] > 20 * abs(right["yillik_getiri"])   # gerçek dışı
    assert free["toplam_getiri"] > right["toplam_getiri"] > high["toplam_getiri"]
    hold = pd.DataFrame(1 / 30, index=ret.index, columns=ret.columns)
    bh = run_backtest(hold, ret)
    assert np.isclose(bh["turnover"].sum(), 1.0) and bh["entries"].sum() == 30   # al-tut maliyeti bir kez öder


# ---------- Adım 5: sektör nötr kıyas, işlem günlüğü, walk-forward ----------
def test_sector_neutral_equals_strategy_when_holding_the_whole_sector():
    """Tüm üyeleri çekirdek olan bir sektörü eşit ağırlıkla tutan strateji = sektör nötr kıyas (sektör içi seçim yok)."""
    from src.backtest.engine import sector_neutral
    from src.data.universe import load_universe
    u = load_universe()
    members = u[u["peer_group"] == "Metals & Mining"]["ticker"].tolist()
    assert len(members) == 3 and u.set_index("ticker").loc[members, "is_core"].all()
    ret = load_returns().iloc[-300:]
    w = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    w[members] = 1 / 3
    bt = run_backtest(w, ret, cost=0.0)
    assert np.allclose(sector_neutral(w, ret).iloc[2:], bt["gross"].iloc[2:])


def test_trade_log_by_hand():
    from src.backtest.engine import trade_log, trade_stats
    idx = pd.bdate_range("2025-01-06", periods=8)
    ret = pd.DataFrame({"AAA": [0, 0.10, -0.05, 0.02, 0, 0, 0.03, 0], "BBB": 0.0}, index=idx)
    w = pd.DataFrame(0.0, index=idx, columns=ret.columns)
    w.loc[idx[0]:idx[2], "AAA"] = 0.5                     # karar 0-2. gün kapanışı -> taşınan 1-3. gün
    w.loc[idx[5]:idx[6], "AAA"] = 0.5                     # taşınan 6-7. gün; veri sonunda açık
    tr = trade_log(w, ret, cost=0.01)
    assert len(tr) == 2 and tr["days"].tolist() == [3, 2] and tr["open"].tolist() == [False, True]
    g = 1.10 * 0.95 * 1.02 - 1
    assert np.isclose(tr.loc[0, "gross"], g) and np.isclose(tr.loc[0, "net"], (1 + g) * 0.99 ** 2 - 1)
    st = trade_stats(tr)
    assert st["islem_kapanan"] == 1 and st["islem_kazanma_orani"] == 1.0


def test_walk_forward_selects_with_past_data_only():
    from src.backtest.strategies import TRAIN, WARMUP, generate_strategy_variants, walk_forward
    v = generate_strategy_variants()
    sub = {k: v[k] for k in ("A_teknik", "C_temel")}
    wf, summ = walk_forward(sub)
    ret = load_returns().iloc[WARMUP:]
    row = wf.iloc[0]
    pos = ret.index.get_loc(pd.Timestamp(row["bas"]))
    past = ret.index[pos - TRAIN:pos]
    assert past[-1] < pd.Timestamp(row["bas"])
    sh = {n: (lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252))(run_backtest(w.reindex(ret.index), ret)["net"].loc[past])
          for n, w in sub.items()}
    assert row["secilen"] == max(sh, key=sh.get) and np.isclose(row["egitim_sharpe"], sh[row["secilen"]])
    assert set(summ["yontem"]) >= {"walk-forward secim", "kiyas xu100"}
