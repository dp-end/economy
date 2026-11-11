
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.disclosure_dates import disclosure_table
from src.data.prices import get_index_history

URL = "https://www.kap.org.tr/tr/api/disclosure/list/main"
CACHE = Path("data/kap")
KAP_CSV = "reports/kap_disclosure_dates.csv"
FIRST_WINDOW = pd.Timestamp("2022-04-01")   # 2022/3 bilançosu en erken Nisan 2022'de yayınlanır
WINDOW_DAYS = 90                            # KAP sorgu aralığı sınırı
PAUSE = 4.0                                 # istekler arası bekleme (saniye)
RETRY_WAIT = 600                            # 429 gelirse bekleme (saniye), KAP süre bildirmezse
DROP_WAIT = 60                              # bağlantı koparsa bekleme (saniye)
MAX_RETRY = 6
CLOSE_HOUR = 18                             # 18:00'den sonra gelen bildirim ertesi işlem gününe sayılır

KAP_OID = {
    "AKBNK": "4028e4a240e8d1830140e905edcd0006", "KCHOL": "4028e4a140f2ed710140f2f4d6c70039",
    "ISMEN": "4028e4a140f275530140f277aa100008", "TURSG": "4028e4a140e95be70140ee2f466d0159",
    "THYAO": "4028e4a140f2ed720140f376bebb01a7", "PGSUS": "4028e4a2422d9a78014232e751bc22a4",
    "BIMAS": "4028e4a140e95be70140ee1b7b030119", "MGROS": "4028e4a141462df2014150162e1c3424",
    "MAVI": "4028e4a141733b5101417ea9aae11fec", "CCOLA": "4028e4a140ee35c00140ee586bdd0034",
    "TUPRS": "4028e4a140f2ed720140f37f139c01bc", "SASA": "4028e4a240ee866c0140f20abd9500cc",
    "GUBRF": "4028e4a241733d4a01417989f3760a4d", "PETKM": "4028e4a240f2ef470141165c566a03e4",
    "EREGL": "4028e4a240e95dc90140ede4c12a0133", "KRDMD": "4028e4a140f2ed7201412ace3ada0707",
    "TRALT": "4028e4a2422d9a6e0142bd91d3687a69", "FROTO": "4028e4a140f2ed71014106890fae0138",
    "TOASO": "4028e4a140f2ed710140f328bed700a5", "ARCLK": "4028e4a240e95dc90140ed55b43900cf",
    "ASELS": "4028e4a1413b7ef401413bc2251e0047", "TCELL": "4028e4a1486ec80a0148c55510d71d31",
    "TTKOM": "4028e4a241e538ab0141ea9c96912c8b", "ENKAI": "4028e4a240e95dc90140ed3883fe0093",
    "OYAKC": "4028e4a140f2ed720140f32377aa016c", "EKGYO": "4028e4a2422d9a780142513cda5b232e",
    "ENJSA": "4028e4a1422d98690142bceaa31957cf", "AKSEN": "4028e4a241733d42014179341a147ddb",
    "MPARK": "4028e4a1462cc5dd01465c9550782baa", "ECILC": "4028e4a1415f4d99014160034b683108",
}


def windows(today=None) -> list:
    """90 günlük dilimler: (başlangıç, bitiş). KAP daha uzun aralıkta hata veriyor."""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    starts = pd.date_range(FIRST_WINDOW, today, freq=f"{WINDOW_DAYS}D")
    return [(s, min(s + pd.Timedelta(days=WINDOW_DAYS - 1), today)) for s in starts]


def cache_file(ticker: str, start, end) -> Path:
    return CACHE / f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}.json"


def fetch(ticker: str, start, end) -> list:
    """Bir hisse + bir aralık için KAP Finansal Rapor bildirimleri. Her yanıt diske kaydedilir."""
    f = cache_file(ticker, start, end)
    if f.exists():                          # daha önce indiyse KAP'a tekrar sorulmaz
        return json.loads(f.read_text())
    body = {"fromDate": f"{start:%d.%m.%Y}", "toDate": f"{end:%d.%m.%Y}",
            "disclosureTypes": ["FR"], "memberTypes": ["IGS"], "mkkMemberOid": KAP_OID[ticker]}
    r = None
    for _ in range(MAX_RETRY):
        try:
            r = requests.post(URL, json=body, headers={"Accept-Language": "tr"}, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            print(f"  bağlantı koptu: {DROP_WAIT} sn bekleniyor ({ticker} {start:%Y-%m-%d})", flush=True)
            time.sleep(DROP_WAIT)
            continue
        if r.status_code != 429:
            break
        ra = r.headers.get("Retry-After", "")
        wait = int(ra) if ra.isdigit() else RETRY_WAIT
        print(f"  KAP istek sınırı: {wait} sn bekleniyor ({ticker} {start:%Y-%m-%d})", flush=True)
        time.sleep(wait)
    if r is None:
        raise RuntimeError(f"KAP'a bağlanılamadı ({ticker} {start:%Y-%m-%d})")
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"KAP beklenmeyen yanıt ({ticker} {start:%Y-%m-%d}): {str(data)[:200]}")
    CACHE.mkdir(parents=True, exist_ok=True)     # dosya adında bitiş tarihi var; ertesi gün yeniden sorulur
    f.write_text(json.dumps(data, ensure_ascii=False))
    time.sleep(PAUSE)
    return data


def parse(ticker: str, data: list) -> list:
    """Sadece 'Finansal Rapor' başlıklı bildirimler: dönem sonu ve yayın zamanı."""
    rows = []
    for item in data:
        b = item["disclosureBasic"]
        if b["title"] != "Finansal Rapor" or not b["year"] or not b["donem"]:
            continue
        period_end = pd.Timestamp(int(b["year"]), int(b["donem"]) * 3, 1) + pd.offsets.MonthEnd(0)
        rows.append(dict(ticker=ticker, period_end=period_end,
                         publish_time=pd.to_datetime(b["publishDate"], format="%d.%m.%Y %H:%M:%S"),
                         kap_index=b["disclosureIndex"]))
    return rows


def event_day(publish_time: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Piyasanın bildirime tepki verebildiği ilk işlem günü (18:00 sonrası -> ertesi işlem günü)."""
    day = publish_time.dt.normalize() + pd.to_timedelta((publish_time.dt.hour >= CLOSE_HOUR).astype(int), unit="D")
    pos = trading_days.searchsorted(day)
    out = pd.Series(pd.NaT, index=publish_time.index, dtype="datetime64[ns]")
    ok = pos < len(trading_days)
    out[ok] = trading_days[pos[ok]]
    return out


def kap_table(tickers=None, today=None) -> pd.DataFrame:
    """Her (hisse, çeyrek) için ilk Finansal Rapor bildirimi + yasal son tarihle karşılaştırma."""
    tickers_all = tickers or list(KAP_OID)
    w = windows(today)
    todo = sum(not cache_file(t, s, e).exists() for t in tickers_all for s, e in w)
    total = len(tickers_all) * len(w)
    print(f"dilim: {total}  kayıtlı: {total - todo}  KAP'a atılacak istek: {todo}", flush=True)
    rows = []
    for t in tickers_all:
        eksik = sum(not cache_file(t, s, e).exists() for s, e in w)
        print(f"{t}: {'önbellekten' if eksik == 0 else f'{eksik} istek'}", flush=True)
        for s, e in w:
            rows += parse(t, fetch(t, s, e))
    kap = (pd.DataFrame(rows).sort_values("publish_time")
           .drop_duplicates(["ticker", "period_end"], keep="first"))
    kap["event_day"] = event_day(kap["publish_time"], get_index_history("XU100").index)
    tab = disclosure_table()
    if tickers:
        tab = tab[tab["ticker"].isin(tickers)]
    out = tab.merge(kap, on=["ticker", "period_end"], how="left")
    out["lag_days"] = (out["publish_time"].dt.normalize() - out["period_end"]).dt.days
    out["after_deadline"] = out["publish_time"].dt.normalize() > out["deadline"]
    return out.sort_values(["ticker", "period_end"]).reset_index(drop=True)


def load_kap_table(path=KAP_CSV) -> pd.DataFrame:
    """Kaydedilmiş KAP tarih tablosu (KAP'a istek atmaz)."""
    return pd.read_csv(path, parse_dates=["period_end", "window_start", "deadline", "publish_time", "event_day"])


def visible_asof_kap(table: pd.DataFrame, date) -> pd.DataFrame:
    """Analiz tarihi 'date' (gün kapanışı) iken kullanılabilecek bilançolar: event_day <= date."""
    return table[table["event_day"] <= pd.Timestamp(date)]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out = kap_table()
    out.to_csv(KAP_CSV, index=False)
    found = out["publish_time"].notna()
    print(f"satır: {len(out)}  KAP tarihi bulunan: {found.sum()}  bulunamayan: {(~found).sum()}")
    print("\nBulunamayanlar (hisse başına):")
    print(out[~found].groupby("ticker").size().to_string())
    print("\nDönem ayına göre gecikme (gün): dönem sonu -> yayın")
    print(out[found].groupby(out["period_end"].dt.month)["lag_days"].describe().round(1).to_string())
    print(f"\nYasal son tarihten SONRA yayınlanan: {int(out['after_deadline'].sum())}")
    print(out[out["after_deadline"]].to_string(index=False))
    print(f"\n18:00 ve sonrası yayınlanan oranı: {(out.loc[found, 'publish_time'].dt.hour >= CLOSE_HOUR).mean():.2%}")
    print("\nASELS son 4 çeyrek:")
    print(out[out["ticker"] == "ASELS"].tail(4).to_string(index=False))