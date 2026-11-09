
import numpy as np
import pandas as pd

from src.data.fundamentals import _norm, _period_end, add_ratios, load_all, FINANCIALS


def test_norm_and_period():
    assert _norm("Ana Ortaklık Payları") == "ana ortaklık payları"
    assert _norm("DÖNEM NET KARI VEYA ZARARI") == "dönem net karı veya zararı"
    assert _norm("İŞ") == "iş"
    assert _period_end("2026/9") == pd.Timestamp("2026-09-30")
    assert _period_end("2016/12") == pd.Timestamp("2016-12-31")
    assert _period_end("2024/6") == pd.Timestamp("2024-06-30")


def test_known_values_from_fintables():
    a = load_all()
    d = a["ASELS"].loc["2026-09-30"]
    assert np.isclose(d["net_income"], 1.122725e10, rtol=1e-4)
    assert np.isclose(d["equity"], 3.349750e11, rtol=1e-4)
    assert np.isclose(d["total_assets"], 6.188709e11, rtol=1e-4)
    assert np.isclose(d["revenue"], 5.806407e10, rtol=1e-4)
    assert np.isclose(a["AKBNK"].loc["2026-09-30", "net_income"], 1.16e10, rtol=1e-3)
    assert np.isclose(a["AKBNK"].loc["2026-09-30", "equity"], 3.3026e11, rtol=1e-3)
    assert np.isclose(a["TURSG"].loc["2026-09-30", "net_income"], -5.420792e9, rtol=1e-4)
    assert np.isclose(a["TURSG"].loc["2026-09-30", "total_assets"], 1.826682e11, rtol=1e-4)


def test_ratios_synthetic():
    idx = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"])
    df = pd.DataFrame({"net_income": [10.0, 20, 30, 40, 50], "equity": 100.0, "total_assets": 200.0}, index=idx)
    r = add_ratios(df)
    assert np.isclose(r.loc["2025-03-31", "ni_yoy"], 4.0)     # (50-10)/10
    assert np.isclose(r.loc["2025-03-31", "ni_qoq"], 0.25)    # (50-40)/40
    assert np.isnan(r.loc["2024-03-31", "ni_yoy"]) and np.isnan(r.loc["2024-03-31", "ni_qoq"])
    assert np.isclose(r.loc["2025-03-31", "roe_q"], 0.5) and np.isclose(r.loc["2025-03-31", "leverage"], 2.0)


def test_coverage_all_core():
    a = load_all()
    assert len(a) == 30
    for t, d in a.items():
        assert d.index.is_monotonic_increasing and d.index.is_unique, t
        recent = d.loc["2019-12-31":]
        for col in ("net_income", "equity", "total_assets"):
            assert col in d.columns and recent[col].notna().sum() >= 20, (t, col)
        if t not in FINANCIALS:
            assert "revenue" in d.columns, t
