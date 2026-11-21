"""Zamanında bilinen temel veri katmanı (src/data/pit_fundamentals.py), Adım 1a.

Yayınlanmış rakamlar 2026-11-22'de basından ve KAP özetlerinden alındı:
  BİM 2026/6: cnbce.com (h35228); 2025 yılı: paratic.com "bim-2025-yilinda-18-7-milyar-tl-net-kar-elde-etti"
  ASELSAN 2026/6: borsamatik.com.tr (haber-180045); 2026/9: paraajansi.com.tr (22231153h); 2022/9: marketscreener
  Turkcell 2026/6: cnbce.com (h34983), 2. çeyrek 5.235 mn TL ve yıllık -%5,7 (KAP sunumu)
  Tüpraş 2026/6: borsaningundemi.com (1904028)   THY 2026/6: uzmanpara (7635491)   MAVI 2026/6: foreks.com
"""
import numpy as np
import pandas as pd
import pytest

import src.data.pit_fundamentals as pf
from src.data.fundamentals import load_all

B = 1e9


# ------------------------------------------------------------------ dönem türü verisi
def test_restater_list_matches_data():
    """Yeniden düzenlenen 24 şirkette 2022/9->2022/12 ve 2023/9->2023/12'de özkaynak sıçrar, Fintables'ta 2022/4. çeyrek
    yoktur. Diğer 6 şirkette sıçrama yoktur. Yeni bir Fintables indirmesi bu deseni bozarsa test düşer."""
    ev = pf.era_evidence().set_index("ticker")
    re, non = ev[ev["restated"]], ev[~ev["restated"]]
    assert len(re) == 24 and len(non) == 6
    assert (re["jump_2022q4"] > 3.0).all() and (re["jump_2023q4"] > 2.0).all()
    assert (non["jump_2022q4"] < 1.8).all() and (non["jump_2023q4"] < 1.8).all()
    assert not re["fintables_has_2022q4"].any() and non["fintables_has_2022q4"].all()


def test_era_rules():
    assert pf.era("ASELS", "2022-09-30", "flow") == "A"
    assert pf.era("ASELS", "2022-12-31", "flow") == "B" and pf.era("ASELS", "2022-12-31", "stock") == "B"
    assert pf.era("ASELS", "2023-06-30", "flow") == "B" and pf.era("ASELS", "2023-06-30", "stock") == "A"
    assert pf.era("ASELS", "2023-12-31", "flow") == "C" and pf.era("ASELS", "2026-09-30", "stock") == "C"
    assert all(pf.era("THYAO", p, k) == "A" for p in ("2022-12-31", "2023-06-30", "2026-09-30")
               for k in ("flow", "stock"))
    assert pf.known_from("ASELS", "2023-03-31", "B") == pf.event_day("ASELS", "2024-03-31")   # bir yıl sonra


def test_unit_month():
    assert pf.unit_month("ASELS") == pd.Timestamp("2026-09-30")
    assert pf.unit_month("MAVI") == pd.Timestamp("2026-07-31")       # mali yıl Şubat-Ocak, etiket "2026/6"
    assert pf.unit_month("THYAO") is None
    assert pf.actual_end("MAVI", "2025-12-31") == pd.Timestamp("2026-01-31")


# ------------------------------------------------------------------ yayınlanmış rakamlar
PUBLISHED = [   # (hisse, dönem, kalem, tek çeyrek mi, açıklanan, tolerans)
    ("ASELS", "2026-06-30", "net_income", False, 14.439306e9, 1e-3),
    ("BIMAS", "2026-06-30", "net_income", False, 14.934476e9, 1e-3),
    ("TCELL", "2026-06-30", "net_income", False, 10.194669e9, 1e-3),
    ("TUPRS", "2026-06-30", "net_income", False, 49.847684e9, 1e-3),
    ("ASELS", "2026-09-30", "net_income", False, 26.47e9, 1e-3),        # son dönem: çevrilmemiş
    ("THYAO", "2026-06-30", "net_income", False, 18.864e9, 1e-3),       # TMS 29 yok
    ("ASELS", "2022-09-30", "net_income", False, 5.79197e9, 1e-3),      # geçiş öncesi, açıklandığı gibi
    ("MAVI", "2026-06-30", "net_income", False, 0.9803e9, 2e-3),        # 1 Şubat - 31 Temmuz 2026
    ("BIMAS", "2025-12-31", "net_income", False, 18.632e9, 1e-3),       # Aralık 2025 TL'si
    ("BIMAS", "2025-12-31", "revenue", False, 721.062e9, 1e-3),
    ("ASELS", "2026-06-30", "net_income", True, 8.5e9, 6e-3),           # basında "8,5 milyar"
    ("ASELS", "2026-09-30", "net_income", True, 11.227e9, 1e-3),
    ("TCELL", "2026-06-30", "net_income", True, 5.235e9, 1e-3),
    ("TUPRS", "2026-06-30", "net_income", True, 45.85e9, 2e-3),         # basında 45,8-45,9
    ("THYAO", "2026-06-30", "net_income", True, 8.949e9, 1e-3),
    ("BIMAS", "2025-12-31", "net_income", True, 6.889111e9, 1e-3),
]


@pytest.mark.parametrize("t,p,item,q,pub,tol", PUBLISHED)
def test_as_published_matches_press(t, p, item, q, pub, tol):
    assert np.isclose(pf.as_published(t, p, item, quarterly=q)[0], pub, rtol=tol)


COMPARATIVES = [   # bir yıl sonraki raporda karşılaştırma olarak görünen rakam (o raporun parasıyla)
    ("BIMAS", "2025-06-30", "2026-06-30", 7.346117e9, 1e-3),
    ("ASELS", "2025-06-30", "2026-06-30", 8.46e9, 1e-3),
    ("ASELS", "2025-09-30", "2026-09-30", 15.11e9, 1e-3),
    ("BIMAS", "2024-12-31", "2025-12-31", 24.328622e9, 1e-3),
]


@pytest.mark.parametrize("t,p,rep,pub,tol", COMPARATIVES)
def test_comparatives_in_report_unit(t, p, rep, pub, tol):
    assert np.isclose(pf.in_report_unit(t, p, "net_income", rep), pub, rtol=tol)


def test_yoy_matches_company_report():
    """Turkcell 2026/2. çeyrek net kârı yıllık -%5,7 (5.235'e karşı 5.549 milyon TL)."""
    a = pf.pit_all()
    r = a[(a["ticker"] == "TCELL") & (a["period_end"] == "2026-06-30")].iloc[0]
    assert np.isclose(r["ni_yoy"], 5.235 / 5.549 - 1, atol=2e-3)
    assert r["unit_flow"] == "TL@2026-06" and np.isclose(r["net_income"], 5.235e9, rtol=1e-3)


# ------------------------------------------------------------------ tutarlılık
def test_quarters_match_fintables_quarterly_sheet():
    """Birikimliden türetilen tek çeyrek, Fintables'ın çeyreklik sayfasıyla aynı (birim karışmayan her yerde).
    Çalışma penceresi 2020 sonrası: Fintables'ın kendi çeyreklik sayfası TURSG 2016/4. çeyrekte birikimli
    rakamlarla tutarsız (23,1 yerine 6,8 milyon TL çıkmalı); örneklem dışında kaldığı için kapsam dışı.
    Tam eşleşmeyen tek çeyrek TRALT 2025/4 (%0,11): iki sayfa 2025/9'un farklı sürümlerini kullanıyor olmalı.
    Fintables en son sürümü sakladığı için şirketin sonradan yaptığı küçük düzeltmeler geçmişe yansır; bu
    test, böyle farkların küçük ve seyrek kaldığını denetler (yeni bir indirme daha büyük fark getirirse düşer)."""
    old = load_all()
    inexact = 0
    for t in pf.core_tickers():
        mine = pf.panel(t)["q_net_income"].loc["2020-03-31":]
        theirs = old[t]["net_income"].reindex(mine.index)
        both = mine.notna() & theirs.notna()
        assert both.sum() >= 18, t
        assert np.allclose(mine[both], theirs[both], rtol=2e-3, atol=1e3), t
        inexact += int((~np.isclose(mine[both], theirs[both], rtol=1e-6, atol=1e3)).sum())
        assert not (theirs.notna() & mine.isna()).any(), t      # Fintables'ta olup bizde düşen çeyrek yok
    assert inexact <= 3


def test_nothing_used_before_it_was_published():
    """Olay tablosundaki her dolu değerin elimizdeki sürümü, olay gününde yayınlanmıştı."""
    a = pf.pit_all()
    for t, g in a.groupby("ticker"):
        pan = pf.panel(t)
        for _, r in g.iterrows():
            for name, q in [(n, n) for n in pf.RATIOS] + list(pf.LEVELS.items()):
                if name in r.index and pd.notna(r[name]):
                    kn = pan.loc[r["period_end"], f"{q}__known"]
                    assert pd.notna(kn) and kn <= r["event_day"], (t, r["period_end"], name)


# ------------------------------------------------------------------ kurallar (gerçek veri)
def _row(t, p):
    a = pf.pit_all()
    return a[(a["ticker"] == t) & (a["period_end"] == p)].iloc[0]


def test_tms29_rules_asels():
    for p in ("2023-03-31", "2023-06-30", "2023-09-30"):      # düzeltilmiş gelir, kendi çeyreğinde kullanılmaz
        r = _row("ASELS", p)
        assert np.isnan(r["ni_yoy"]) and np.isnan(r["net_income"]) and "net_income:gec" in r["pit_dropped"]
        assert pd.notna(r["leverage"]) and r["unit_stock"] == "TL nominal"   # bilanço açıklandığı gibi
    r = _row("ASELS", "2023-12-31")                             # 4. çeyrek tek başına bilinemezdi
    assert np.isnan(r["net_income"]) and pd.notna(r["net_income_ttm"]) and pd.notna(r["leverage"])
    r = _row("ASELS", "2024-03-31")                             # yıllık değişim geçerli, çeyreklik değil
    assert pd.notna(r["ni_yoy"]) and np.isnan(r["ni_qoq"]) and pd.notna(r["net_income_ttm"])
    r = _row("ASELS", "2024-06-30")
    assert pd.notna(r["ni_yoy"]) and pd.notna(r["ni_qoq"])
    r = _row("ASELS", "2022-09-30")                             # geçiş öncesi: nominal ile nominal
    assert pd.notna(r["ni_yoy"]) and r["unit_flow"] == "TL nominal"


def test_non_restater_keeps_2023():
    r = _row("THYAO", "2023-06-30")
    assert pd.notna(r["ni_yoy"]) and pd.notna(r["ni_qoq"]) and r["unit_flow"] == "TL nominal"


# ------------------------------------------------------------------ kurallar (yapay şirket, veriden bağımsız)
def test_synthetic_restater(monkeypatch):
    t = "ZZSYN"
    per = pd.to_datetime(["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                          "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"])
    cum = [4.0, 10, 16, 100, 30, 70, 120, 200, 60, 130]        # birikimli net kâr (2022/12'den sonrası 'U' biriminde)
    raw = pd.DataFrame({"net_income": cum, "equity": 1000.0, "total_assets": 2000.0}, index=per)
    ev = {(t, p): p + pd.Timedelta(days=40) for p in per}
    months = pd.date_range("2020-01-01", "2026-12-01", freq="MS")
    cpi = pd.Series(np.linspace(100, 400, len(months)), index=months)
    monkeypatch.setattr(pf, "read_raw", lambda x: raw)
    monkeypatch.setattr(pf, "_event_days", lambda: ev)
    monkeypatch.setattr(pf, "_cpi", lambda: cpi)
    pf.unit_month.cache_clear()
    q = pf.pit_quarters(t).set_index("period_end")
    pf.unit_month.cache_clear()
    # 2024/6: tek çeyrek 130-60=70; geçen yılın 2. çeyreği 70-30=40 (ikisi de 2024/6 raporuyla bilinir)
    assert np.isclose(q.loc["2024-06-30", "ni_yoy"], (70 - 40) / 40)
    # 2024/3 son 12 ay: 60 + 200 - 30 = 230 (2023 ara dönemleri toplamda birbirini götürür)
    f = cpi[pd.Timestamp("2024-03-01")] / cpi[pd.Timestamp("2024-06-01")]   # U = son dönem (Haziran 2024)
    assert np.isclose(q.loc["2024-03-31", "net_income_ttm"], 230 * f)
    assert np.isclose(q.loc["2024-03-31", "net_income"], 60 * f)
    assert np.isnan(q.loc["2024-03-31", "ni_qoq"])                    # 2023/4. çeyrek o gün bilinemezdi
    assert np.isnan(q.loc["2023-12-31", "net_income"])                # aynı neden
    assert np.isclose(q.loc["2023-12-31", "net_income_ttm"], 200 * cpi[pd.Timestamp("2023-12-01")] / cpi[pd.Timestamp("2024-06-01")])
    assert np.isnan(q.loc["2023-06-30", "ni_yoy"])                    # düzeltilmiş (40) ile nominal (6) karışmaz
    assert "ni_yoy:birim" in q.loc["2023-06-30", "pit_dropped"]
    assert np.isnan(q.loc["2023-09-30", "net_income_ttm"])
    assert np.isclose(q.loc["2022-09-30", "ni_qoq"], (6 - 6) / 6)    # geçiş öncesi: 16-10 ile 10-4
    assert q.loc["2022-09-30", "unit_flow"] == "TL nominal"


# ------------------------------------------------------------------ Adım 1c: sanayi oranları
PUBLISHED_1C = [   # (hisse, dönem, kalem, açıklanan, kaynak)
    ("BIMAS", "2025-12-31", "ebitda", 43.484e9),     # BİM 2025 FAVÖK (paratic, 2025 yılı)
    ("ASELS", "2026-06-30", "ebitda", 23.233e9),     # ASELSAN 2026/6 FAVÖK (borsamatik 180045)
    ("TUPRS", "2026-06-30", "ebitda", 71.246e9),     # Tüpraş 2026/6 FAVÖK (borsaningundemi 1904028)
    ("BIMAS", "2026-06-30", "ocf", 37.9e9),          # BİM 2026/6 işletme faaliyetlerinden nakit akışı (cnbce h35228)
    ("THYAO", "2026-06-30", "net_debt", 620.3e9),    # THY 2026/6 net borç (cnbce h34421)
]


@pytest.mark.parametrize("t,p,item,pub", PUBLISHED_1C)
def test_industrial_items_match_published(t, p, item, pub):
    """FAVÖK = brüt kâr − genel yönetim − pazarlama − Ar-Ge + amortisman; net borç = finansal borç − nakit −
    kısa vadeli finansal yatırım; işletme nakit akışı = 'İşletme Faaliyetlerinden Nakit Akışları'."""
    assert np.isclose(pf.as_published(t, p, item)[0], pub, rtol=1e-3)


def test_ocf_comparative_in_next_report():
    assert np.isclose(pf.in_report_unit("BIMAS", "2025-06-30", "ocf", "2026-06-30"), 29.45e9, rtol=1e-3)


def test_ratios_recomputed_by_hand():
    raw, r = pf.read_raw("ASELS"), _row("ASELS", "2026-09-30")
    p, ye, l4 = pd.Timestamp("2026-09-30"), pd.Timestamp("2025-12-31"), pd.Timestamp("2025-09-30")
    x = raw.loc[p]
    ttm = lambda c: raw.loc[p, c] + raw.loc[ye, c] - raw.loc[l4, c]     # aynı birim (son rapor), faktör 1
    assert np.isclose(r["current_ratio"], x["current_assets"] / x["current_liab"])
    assert np.isclose(r["quick_ratio"], (x["current_assets"] - x["inventories"]) / x["current_liab"])
    assert np.isclose(r["debt_to_equity"], (x["fin_debt_st"] + x["fin_debt_lt"]) / x["total_equity"])
    assert np.isclose(r["nd_ebitda"], x["net_debt"] / ttm("ebitda"))
    assert np.isclose(r["cash_conversion"], ttm("ocf") / ttm("ebitda"))
    assert np.isclose(r["roa"], ttm("net_income") / x["total_assets"])
    assert np.isclose(r["roe_ttm"], ttm("net_income") / x["equity"])


def test_no_industrial_ratios_for_banks_insurers_brokers_and_kchol():
    a = pf.pit_all()
    for t in ("AKBNK", "TURSG", "ISMEN", "KCHOL"):
        assert "ebitda" not in pf.read_raw(t).columns and "net_debt" not in pf.read_raw(t).columns
        g = a[a["ticker"] == t]
        assert g["current_ratio"].isna().all() and g["nd_ebitda"].isna().all()
        assert g["roa"].notna().sum() >= 12                    # net kâr temelli oranlar var


def test_not_meaningful_when_ebitda_negative():
    r = _row("PETKM", "2024-12-31")
    assert r["ebitda_ttm"] < 0 and np.isnan(r["nd_ebitda"]) and np.isnan(r["cash_conversion"])
    assert "nd_ebitda:anlamsiz" in r["pit_dropped"]


def test_missing_expense_line_counts_as_zero_only_when_period_exists():
    raw = pf.read_raw("BIMAS")                                    # BİM'de Ar-Ge satırı yok
    has = raw["gross_profit"].notna()
    assert (raw.loc[has, "opex_rd"] == 0).all() and raw.loc[~has, "opex_rd"].isna().all()
