
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
