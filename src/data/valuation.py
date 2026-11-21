"""Günlük değerleme oranları (Adım 1d, kullanıcı kararları 2026-11-22).

Her işlem günü t ve hisse için, t kapanışında bilinen bilgilerle:
  F/K       = piyasa değeri / son 12 ay net kâr (ana ortaklık)
  PD/DD     = piyasa değeri / ana ortaklık özkaynağı
  FD/FAVÖK  = (piyasa değeri + net borç + azınlık payları) / son 12 ay FAVÖK   (banka, sigorta, aracı kurum, KCHOL hariç)

- Piyasa değeri: src/data/market_cap.py (İş Yatırım PD; sermaye-fiyat gün kayması düzeltilmiş), t gününün parası.
- Bilanço değerleri: t'de görünen SON raporun açıklama günündeki değerleri (reports/earnings_events.parquet).
- Para birimi (karar 1): enflasyon muhasebeli (TMS 29) kâr, FAVÖK, özkaynak ve azınlık payı, t gününe kadar
  yayınlanmış son TÜFE ile t'nin parasına taşınır: çarpan = TÜFE(son yayınlanan ay) / TÜFE(raporun ayı).
  Nominal dönemler ve nominal şirketler olduğu gibi kalır (tek bir ayın parası değiller). Net borç taşınmaz: TL
  borcun tutarı enflasyonla büyümez. Karşılaştırma için 'pe_reported' raporun parasıyla F/K'yı (ekranda görünen) verir.
- Payda ≤ 0 ise oran boş, nedeni 'anlamsiz' (karar 2). Azınlık payları firma değerine eklenir (karar 3).
- Raporun değeri açıklama gününde bilinmiyorsa (TMS 29 şirketlerinin 2023/1-3. çeyrekleri) oran boş kalır.

Çalıştırma: python -m src.data.valuation    # reports/valuation_daily.parquet
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.macro import load_series
from src.data.market_cap import core_tickers, market_cap

OUT = "reports/valuation_daily.parquet"
FROM_EVENTS = ["period_end", "event_day", "net_income_ttm", "equity", "ebitda_ttm", "net_debt", "minority_interest",
               "unit_flow", "unit_stock"]


@lru_cache(maxsize=1)
def _cpi():
    c = load_series("cpi")
    return c.set_index("data_date")["value"], c.sort_values("available_date")


def cpi_factor(unit, day) -> tuple:
    """'TL@YYYY-MM' birimindeki tutarı 'day' gününe kadar yayınlanmış son TÜFE ayının parasına çeviren çarpan.
    Nominal ya da birimsiz tutarda 1. Döner: (çarpan, kullanılan TÜFE ayı)."""
    if not isinstance(unit, str) or not unit.startswith("TL@"):
        return 1.0, pd.NaT
    by_month, by_avail = _cpi()
    known = by_avail[by_avail["available_date"] <= pd.Timestamp(day)]
    m = known["data_date"].iloc[-1]
    return float(by_month.loc[m] / by_month.loc[pd.Timestamp(unit[3:] + "-01")]), m


def daily(ticker: str, events: pd.DataFrame = None) -> pd.DataFrame:
    ev = pd.read_parquet("reports/earnings_events.parquet") if events is None else events
    ev = ev[(ev["ticker"] == ticker) & ev["event_day"].notna()].sort_values("event_day")[FROM_EVENTS]
    mc = market_cap(ticker).rename("market_cap").to_frame()
    mc.index.name = "date"
    d = pd.merge_asof(mc.reset_index(), ev, left_on="date", right_on="event_day", direction="backward")
    d = d[d["event_day"].notna()].copy()            # ilk KAP raporundan (2022/1. çeyrek) önce bilanço yok
    rows = []
    for r in d.itertuples(index=False):
        ff, m = cpi_factor(r.unit_flow, r.date)
        fs, _ = cpi_factor(r.unit_stock, r.date)
        why = []

        def ratio(name, num, den):
            if pd.isna(num) or pd.isna(den):
                why.append(f"{name}:rapor_degeri_yok")
                return np.nan
            if den <= 0:
                why.append(f"{name}:anlamsiz")
                return np.nan
            return num / den

        ni, eq = r.net_income_ttm * ff, r.equity * fs
        ebitda = r.ebitda_ttm * ff
        ev_ = r.market_cap + r.net_debt + (0.0 if pd.isna(r.minority_interest) else r.minority_interest * fs)
        out = dict(date=r.date, ticker=ticker, market_cap=r.market_cap, report_period=r.period_end,
                   report_event_day=r.event_day, cpi_month=m, factor_flow=ff, factor_stock=fs,
                   pe=ratio("pe", r.market_cap, ni), pb=ratio("pb", r.market_cap, eq),
                   ev=ev_ if pd.notna(r.net_debt) else np.nan,
                   ev_ebitda=(ratio("ev_ebitda", ev_, ebitda) if pd.notna(r.net_debt) else np.nan),
                   pe_reported=(r.market_cap / r.net_income_ttm if pd.notna(r.net_income_ttm)
                                and r.net_income_ttm > 0 else np.nan))
        if pd.isna(r.net_debt):
            why.append("ev_ebitda:sanayi_disi_ya_da_rapor_degeri_yok")
        out["val_dropped"] = ";".join(why)
        rows.append(out)
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def valuation_all() -> pd.DataFrame:
    ev = pd.read_parquet("reports/earnings_events.parquet")
    return pd.concat([daily(t, ev) for t in core_tickers()], ignore_index=True)


def load_valuation() -> pd.DataFrame:
    return pd.read_parquet(OUT)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    v = valuation_all()
    v.to_parquet(OUT)
    print(f"satır: {len(v)}  F/K dolu: {int(v['pe'].notna().sum())}  PD/DD: {int(v['pb'].notna().sum())}  "
          f"FD/FAVÖK: {int(v['ev_ebitda'].notna().sum())}")
    last = v[v["date"] == v["date"].max()].set_index("ticker")[["pe", "pe_reported", "pb", "ev_ebitda"]]
    print(last.round(2).to_string())
