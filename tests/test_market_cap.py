"""Günlük piyasa değeri ve değerleme (Adım 1d)."""
import numpy as np
import pandas as pd

from src.data.macro import load_series
from src.data.market_cap import aligned_capital, check, core_tickers, fetch, market_cap
from src.data.valuation import load_valuation


def test_market_cap_is_price_times_capital_and_only_three_days_fixed():
    fixed = []
    for t in core_tickers():
        d = fetch(t)
        assert np.allclose(d["PD"], d["HG_KAPANIS"] * d["SERMAYE"], rtol=1e-6), t
        fixed += [(t, f["date"].date().isoformat()) for f in aligned_capital(t)[1]]
        mc = market_cap(t)
        off = ~np.isclose(mc, d["PD"], rtol=1e-6)
        assert off.sum() == len([x for x in fixed if x[0] == t]), t      # yalnızca düzeltilen günler farklı
    assert sorted(fixed) == [("BIMAS", "2026-05-13"), ("TURSG", "2025-06-03"), ("TURSG", "2026-06-12")]


def test_no_jump_on_bonus_issues():
    """Bedelsiz/bölünme günlerinde piyasa değeri getirisi düzeltilmiş fiyat getirisine eşit."""
    for t in core_tickers():
        c = check(t)
        assert c["max_gap_on_split"] < 0.01, t
    d = fetch("TURSG")
    mc = market_cap("TURSG")
    assert abs(mc["2025-06-03"] / mc["2025-06-02"] - 1) < 0.02          # eskiden PD bir günde +%99


def test_valuation_by_hand_asels():
    v = load_valuation().set_index(["ticker", "date"])
    ev = pd.read_parquet("reports/earnings_events.parquet")
    e = ev[(ev["ticker"] == "ASELS") & (ev["period_end"] == "2024-03-31")].iloc[0]
    cpi = load_series("cpi").set_index("data_date")["value"]
    day = pd.Timestamp("2024-09-10")                                    # son görünen rapor 2024/3, son TÜFE Ağustos
    f = cpi[pd.Timestamp("2024-08-01")] / cpi[pd.Timestamp("2024-03-01")]
    r = v.loc[("ASELS", day)]
    assert r["report_period"] == pd.Timestamp("2024-03-31") and r["cpi_month"] == pd.Timestamp("2024-08-01")
    assert np.isclose(r["pe"], r["market_cap"] / (e["net_income_ttm"] * f))
    assert np.isclose(r["pe_reported"], r["market_cap"] / e["net_income_ttm"])
    assert np.isclose(r["pb"], r["market_cap"] / (e["equity"] * f))
    ev_ = r["market_cap"] + e["net_debt"] + e["minority_interest"] * f   # net borç taşınmaz
    assert np.isclose(r["ev_ebitda"], ev_ / (e["ebitda_ttm"] * f))


def test_valuation_uses_only_public_information():
    v = load_valuation()
    cpi = load_series("cpi").set_index("data_date")["available_date"]
    assert (v["report_event_day"] <= v["date"]).all()
    has = v["cpi_month"].notna()
    assert (v.loc[has, "cpi_month"].map(cpi) <= v.loc[has, "date"]).all()


def test_not_meaningful_and_nominal_cases():
    v = load_valuation().set_index(["ticker", "date"])
    r = v.loc[("PETKM", pd.Timestamp("2026-11-06"))]                    # son 12 ay zarar
    assert np.isnan(r["pe"]) and "pe:anlamsiz" in r["val_dropped"]
    a = v.loc["AKBNK"]                                                  # nominal şirket: çeviri yok, EV/FAVÖK yok
    assert np.allclose(a["pe"].dropna(), a["pe_reported"].dropna()) and a["ev_ebitda"].isna().all()
