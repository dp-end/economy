
import numpy as np
import pandas as pd

from src.events.calendar_patterns import (
    welch, bh_fdr, run_length, streak_masks, weekday_portfolio, weekday_per_stock,
    pattern_table, _data,
)


def test_welch_known():
    t, p = welch([1, 2, 3, 4], [1, 2, 3, 4])
    assert np.isclose(t, 0) and np.isclose(p, 1)
    t, p = welch([10, 11, 12, 13], [1, 2, 3, 4])
    assert np.isclose(t, 9.859, atol=1e-3) and p < 1e-6


def test_bh_fdr_known():
    q = bh_fdr([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_run_length():
    b = pd.Series([True, True, False, True, True, True])
    assert run_length(b).tolist() == [1, 2, 0, 1, 2, 3]


def test_valid_mask_properties():
    d = _data()
    idx = d["ret"].index
    gap = pd.Series(idx, index=idx).diff().dt.days
    v = d["valid"]
    mon = pd.Series(idx.day_of_week == 0, index=idx)
    assert (gap[v & mon] == 3).all() and (gap[v & ~mon] == 1).all()
    assert not v.loc["2023-02-15"]          # deprem sonrası ilk gün hariç
    assert v.sum() > 1000


def test_streak_masks_manual():
    idx = pd.bdate_range("2024-01-01", periods=6)
    ret = pd.DataFrame({"A": [0.01, -0.01, -0.02, -0.01, 0.01, -0.01]}, index=idx)
    valid = pd.Series(True, index=idx)
    m = streak_masks(ret, valid, ks=(2, 3))
    assert m[("down", 2)]["A"].tolist() == [False, False, True, False, False, False]
    assert m[("down", 3)]["A"].tolist() == [False, False, False, True, False, False]
    valid2 = valid.copy()
    valid2.iloc[2] = False                     # geçersiz gün seriyi böler
    assert not streak_masks(ret, valid2, ks=(3,))[("down", 3)]["A"].any()


def test_weekday_portfolio_synthetic():
    idx = pd.bdate_range("2024-01-01", periods=200)
    noise = np.where(np.arange(200) % 2 == 0, 0.001, -0.001)
    r = noise + (idx.day_of_week == 0) * 0.01
    ret = pd.DataFrame({"A": r, "B": r}, index=idx)
    out = weekday_portfolio(ret, pd.Series(True, index=idx))
    assert 0.9 < out.loc["Pzt", "diff"] < 1.1 and out.loc["Pzt", "t"] > 5
    assert abs(out.loc["Çar", "diff"]) < 0.5


def test_real_tables_shapes():
    d = _data()
    ws = weekday_per_stock(d["ret"], d["valid"])
    assert len(ws) == 150 and (ws["q_bh"] >= ws["p"] - 1e-12).all()
    pt = pattern_table(d["ret"], d["panel"], d["valid"])
    assert len(pt) == 18 and (pt["n_events"] >= 100).all()
    assert pt["p"].between(0, 1).all()
