import numpy as np
import pandas as pd

from src.data.disclosure_dates import (START_PERIOD, disclosure_table, excess_volume_z,
                                       legal_deadline, season_mask, visible_asof)


def test_legal_deadline_hand_computed():
    assert legal_deadline("2026-09-30") == pd.Timestamp("2026-11-09")   # +40 gün, Pazartesi
    assert legal_deadline("2025-09-30") == pd.Timestamp("2025-11-10")   # 9 Kasım Pazar -> Pazartesi
    assert legal_deadline("2025-12-31") == pd.Timestamp("2026-03-11")   # +70 gün
    assert legal_deadline("2024-06-30") == pd.Timestamp("2024-08-29")   # +60 gün


def test_visible_asof_no_lookahead():
    tab = pd.DataFrame({"ticker": ["A", "A"],
                        "period_end": pd.to_datetime(["2024-06-30", "2024-09-30"])})
    tab["deadline"] = tab["period_end"].map(legal_deadline)            # 2024-08-29, 2024-11-11
    assert len(visible_asof(tab, "2024-08-28")) == 0                   # son tarihten önce görünmez
    assert visible_asof(tab, "2024-08-29")["period_end"].tolist() == [pd.Timestamp("2024-06-30")]
    assert len(visible_asof(tab, "2024-10-15")) == 1                   # 2024/9 dönemi bitti ama henüz yok
    assert len(visible_asof(tab, "2024-11-11")) == 2


def test_excess_volume_z_synthetic():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=120)
    vol = pd.DataFrame(np.exp(rng.normal(10, 0.2, (120, 5))), index=idx, columns=list("ABCDE"))
    day = idx[100]
    one = vol.copy()
    one.loc[day, "A"] *= 10                       # sadece A'da sıçrama
    ex = excess_volume_z(one)
    assert ex.loc[day, "A"] > 5
    assert ex.loc[day, ["B", "C", "D", "E"]].abs().max() < 3
    assert ex.iloc[:40].isna().all().all()        # ısınma dönemi
    # geleceğe bakma yok: veriyi o günde kesince değer değişmez
    assert np.isclose(excess_volume_z(one.loc[:day]).loc[day, "A"], ex.loc[day, "A"])
    allup = vol.copy()
    allup.loc[day] *= 10                          # piyasa geneli sıçrama -> elenir
    assert excess_volume_z(allup).loc[day].abs().max() < 3


def test_real_table_and_season_mask():
    tab = disclosure_table()
    assert tab["ticker"].nunique() == 30
    assert tab["period_end"].min() == START_PERIOD
    assert not tab.duplicated(["ticker", "period_end"]).any()
    assert (tab["window_start"] < tab["deadline"]).all()
    assert (tab["deadline"].dt.day_of_week < 5).all()
    m = season_mask(pd.bdate_range("2024-01-01", "2024-12-31"))
    assert m[pd.Timestamp("2024-08-01")] and m[pd.Timestamp("2024-08-29")]     # 2024/6 penceresi
    assert not m[pd.Timestamp("2024-08-30")] and not m[pd.Timestamp("2024-07-12")]
