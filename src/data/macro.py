
import os
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.prices import get_index_history

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "macro"
BASE = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
START = pd.Timestamp("2020-08-01")      # TÜFE yıllık değişimi için 1 yıldan fazla geriden başlar
PAUSE = 1.0

# ad -> (EVDS seri kodu, sıklık)
EVDS_SERIES = {
    "usdtry": ("TP.DK.USD.A.YTL", "D"),          # TCMB gösterge döviz alış kuru
    "eurtry": ("TP.DK.EUR.A.YTL", "D"),
    "policy_rate": ("TP.PY.P02.1H", "D"),        # TCMB 1 hafta repo (politika) faizi, %
    "cpi": ("TP.TUKFIY2025.GENEL", "M"),         # TÜİK TÜFE genel endeks (2025=100)
}
CPI_RELEASE_DAY = 3                              # TÜİK, TÜFE'yi izleyen ayın 3'ünde 10:00'da yayınlar

# Fed federal fon hedef aralığı: (yürürlük günü, alt, üst). Kaynak: federalreserve.gov/monetarypolicy/openmarket.htm
# Karar bir gün önce ABD saatiyle 14:00'te (Türkiye'de BIST kapandıktan sonra) açıklanır.
FED_TABLE = [
    ("2020-03-16", 0.00, 0.25), ("2022-03-17", 0.25, 0.50), ("2022-05-05", 0.75, 1.00), ("2022-06-16", 1.50, 1.75),
    ("2022-07-28", 2.25, 2.50), ("2022-09-22", 3.00, 3.25), ("2022-11-03", 3.75, 4.00), ("2022-12-15", 4.25, 4.50),
    ("2023-02-02", 4.50, 4.75), ("2023-03-23", 4.75, 5.00), ("2023-05-04", 5.00, 5.25), ("2023-07-27", 5.25, 5.50),
    ("2024-09-19", 4.75, 5.00), ("2024-11-08", 4.50, 4.75), ("2024-12-19", 4.25, 4.50), ("2025-09-18", 4.00, 4.25),
    ("2025-10-30", 3.75, 4.00), ("2025-12-11", 3.50, 3.75), ("2026-09-17", 3.75, 4.00),
]


def year_chunks(start, end) -> list:
    """EVDS tek istekte en fazla 1000 satır veriyor; yıllık dilimlere böl."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for y in range(start.year, end.year + 1):
        out.append((max(start, pd.Timestamp(y, 1, 1)), min(end, pd.Timestamp(y, 12, 31))))
    return out


def _fetch_evds(code: str, start, end) -> pd.Series:
    load_dotenv(ROOT / ".env")
    col, rows = code.replace(".", "_"), {}
    for a, b in year_chunks(start, end):
        url = f"{BASE}series={code}&startDate={a:%d-%m-%Y}&endDate={b:%d-%m-%Y}&type=json"
        r = requests.get(url, headers={"key": os.environ["EVDS_API_KEY"]}, timeout=30)
        r.raise_for_status()
        for item in r.json().get("items", []):
            if item.get(col) is not None:
                rows[item["Tarih"]] = float(item[col])
        time.sleep(PAUSE)
    return pd.Series(rows, dtype=float)


def cpi_release_date(month_start: pd.Timestamp) -> pd.Timestamp:
    """TÜFE ayı -> yayın günü (izleyen ayın 3'ü; hafta sonuysa sonraki iş günü). Kural, gerçek takvim değil."""
    d = (month_start + pd.offsets.MonthBegin(1)) + pd.Timedelta(days=CPI_RELEASE_DAY - 1)
    return pd.offsets.BDay().rollforward(d)


def load_series(name: str, force: bool = False) -> pd.DataFrame:
    """Sütunlar: data_date (verinin ait olduğu gün/ay), value, available_date (kamuya açık olduğu gün), source."""
    f = CACHE / f"{name}.csv"
    if f.exists() and not force:
        return pd.read_csv(f, parse_dates=["data_date", "available_date"])
    code, freq = EVDS_SERIES[name]
    raw = _fetch_evds(code, START, pd.Timestamp.today().normalize())
    if freq == "M":
        dates = pd.to_datetime(raw.index, format="%Y-%m")
        avail = [cpi_release_date(d) for d in dates]
    else:
        dates = pd.to_datetime(raw.index, format="%d-%m-%Y")
        avail = dates                           # kur ve faiz, tarihini taşıdığı gün kapanışta bilinir
    df = pd.DataFrame({"data_date": dates, "value": raw.to_numpy(), "available_date": avail,
                       "source": f"EVDS {code}"}).sort_values("data_date").reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


def fed_table() -> pd.DataFrame:
    df = pd.DataFrame(FED_TABLE, columns=["data_date", "fed_lower", "fed_upper"])
    df["data_date"] = pd.to_datetime(df["data_date"])
    df["available_date"] = df["data_date"]
    df["source"] = "federalreserve.gov open market operations"
    return df


def _asof(df: pd.DataFrame, days: pd.DatetimeIndex, col: str = "value") -> pd.Series:
    """Her gün için, o güne kadar YAYINLANMIŞ son değer (available_date <= gün)."""
    s = df.sort_values("available_date").drop_duplicates("available_date", keep="last").set_index("available_date")[col]
    return s.reindex(s.index.union(days)).ffill().reindex(days)


@lru_cache(maxsize=1)
def macro_panel() -> pd.DataFrame:
    """İşlem günlerine hizalı makro tablo. Her satır yalnızca o gün kapanışta bilinen değerleri içerir."""
    days = get_index_history("XU100").index
    out = pd.DataFrame(index=days)
    for name in ("usdtry", "eurtry", "policy_rate"):
        out[name] = _asof(load_series(name), days)
    cpi = load_series("cpi").copy()
    cpi["yoy"] = cpi["value"].pct_change(12) * 100          # yıllık enflasyon, %
    cpi["mom"] = cpi["value"].pct_change(1) * 100
    out["cpi_yoy"] = _asof(cpi, days, "yoy")
    out["cpi_mom"] = _asof(cpi, days, "mom")
    out["cpi_month"] = _asof(cpi.assign(m=cpi["data_date"].dt.strftime("%Y-%m")), days, "m")
    fed = fed_table()
    out["fed_lower"] = _asof(fed, days, "fed_lower")
    out["fed_upper"] = _asof(fed, days, "fed_upper")
    out["usdtry_chg_20d"] = out["usdtry"].pct_change(20) * 100
    out["real_rate"] = out["policy_rate"] - out["cpi_yoy"]  # basit reel faiz: politika faizi - yıllık TÜFE
    out["rate_change"] = out["policy_rate"].diff().fillna(0)
    return out


def get_macro_context(start, end=None) -> pd.DataFrame:
    """[start, end] aralığındaki işlem günleri için makro bağlam (end verilmezse tek gün)."""
    p = macro_panel()
    return p.loc[pd.Timestamp(start):pd.Timestamp(end or start)]


def soft_warnings() -> list:
    """Eksik aralıklar: 7 günden uzun kur/faiz boşlukları ve atlanan TÜFE ayları."""
    w = []
    for name in ("usdtry", "eurtry", "policy_rate"):
        d = load_series(name)["data_date"]
        gap = d.diff().dt.days
        for i in gap[gap > 7].index:
            w.append(f"{name}: {d[i - 1].date()} -> {d[i].date()} arası {int(gap[i])} gün veri yok")
    m = load_series("cpi")["data_date"]
    missing = pd.date_range(m.min(), m.max(), freq="MS").difference(m)
    if len(missing):
        w.append(f"cpi: eksik aylar {[x.strftime('%Y-%m') for x in missing]}")
    p = macro_panel()
    for c in ("usdtry", "eurtry", "policy_rate", "cpi_yoy", "fed_upper"):
        if p[c].isna().any():
            w.append(f"panel: {c} sütununda {int(p[c].isna().sum())} boş gün")
    return w


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for name in EVDS_SERIES:
        d = load_series(name)
        print(f"{name:12s} satır: {len(d):5d}  ilk: {d['data_date'].min().date()}  son: {d['data_date'].max().date()}")
    p = macro_panel()
    p.to_csv(ROOT / "reports" / "macro_panel.csv")
    print(f"\npanel: {len(p)} işlem günü, {p.index.min().date()} - {p.index.max().date()}")
    print("\nSon 3 gün:")
    print(p.tail(3).round(2).to_string())
    rate = load_series("policy_rate")
    ch = rate[rate["value"] != rate["value"].shift()].iloc[1:]
    print(f"\nTCMB politika faizi değişiklikleri ({len(ch)}):")
    print(ch[["data_date", "value"]].assign(gun=ch["data_date"].dt.day_name()).to_string(index=False))
    print("\nYıl sonu görünümü:")
    print(p.groupby(p.index.year).tail(1)[["usdtry", "policy_rate", "cpi_yoy", "real_rate", "fed_upper"]].round(2).to_string())
    print("\nSoft uyarılar:")
    for line in soft_warnings() or ["yok"]:
        print(" -", line)