
import pandas as pd

from src.data.kap_dates import (KAP_OID, WINDOW_DAYS, event_day, load_kap_table, parse,
                                visible_asof_kap, windows)
from src.data.universe import load_universe


def test_windows_cover_range_without_gaps():
    w = windows("2026-11-11")
    assert w[0][0] == pd.Timestamp("2022-04-01") and w[-1][1] == pd.Timestamp("2026-11-11")
    assert all((e - s).days <= WINDOW_DAYS - 1 for s, e in w)            # KAP sınırı aşılmaz
    assert all(w[i + 1][0] == w[i][1] + pd.Timedelta(days=1) for i in range(len(w) - 1))


def test_all_core_tickers_have_kap_id():
    u = load_universe()
    assert set(u[u["is_core"] == 1]["ticker"]) == set(KAP_OID)
    assert len(set(KAP_OID.values())) == 30


def test_parse_keeps_only_financial_report():
    def item(title, year, donem, date):
        return {"disclosureBasic": {"title": title, "year": year, "donem": donem,
                                    "publishDate": date, "disclosureIndex": 1}}
    data = [item("Finansal Rapor", 2026, 3, "27.10.2026 18:34:03"),
            item("Faaliyet Raporu (Konsolide)", 2026, 3, "27.10.2026 18:34:27"),
            item("Finansal Rapor", 2025, 4, "24.02.2026 18:27:36"),
            item("Finansal Rapor", None, None, "01.01.2026 10:00:00")]
    rows = parse("ASELS", data)
    assert [r["period_end"] for r in rows] == [pd.Timestamp("2026-09-30"), pd.Timestamp("2025-12-31")]
    assert rows[0]["publish_time"] == pd.Timestamp("2026-10-27 18:34:03")


def test_event_day_after_close_goes_to_next_trading_day():
    days = pd.DatetimeIndex(["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-30", "2026-11-02"])
    t = pd.Series(pd.to_datetime(["2026-10-27 18:34:03",    # kapanıştan sonra -> 28 Ekim
                                  "2026-10-27 10:00:00",    # seans içinde -> aynı gün
                                  "2026-10-28 19:00:00",    # 29 Ekim tatil -> 30 Ekim
                                  "2026-10-31 12:00:00",    # Cumartesi -> Pazartesi
                                  "2026-11-02 18:00:00"]))  # son günden sonrası yok -> NaT
    out = event_day(t, days)
    assert out.iloc[:4].tolist() == list(pd.to_datetime(["2026-10-28", "2026-10-27", "2026-10-30", "2026-11-02"]))
    assert pd.isna(out.iloc[4])


def test_visible_asof_kap_no_lookahead():
    tab = load_kap_table()
    assert len(tab) == 545 and tab["publish_time"].notna().all()
    for d in pd.to_datetime(["2022-05-10", "2024-03-11", "2024-05-10", "2025-11-10"]):
        v = visible_asof_kap(tab, d)
        assert (v["publish_time"] <= d + pd.Timedelta(hours=18)).all()   # yayınlanmamış bilanço görünmez
    a = tab[(tab["ticker"] == "ASELS") & (tab["period_end"] == "2026-09-30")]
    assert len(visible_asof_kap(a, "2026-10-27")) == 0     # 27 Ekim 18:34'te yayınlandı
    assert len(visible_asof_kap(a, "2026-10-28")) == 1     # ertesi işlem günü görünür