"""KAP özel durum açıklamaları (ÖDA): şirket haberlerinin resmi, zaman damgalı kaynağı.

Sadece başlık, özet, yayın zamanı ve bağlantı saklanır (haber metni değil). Her duyuru, piyasanın
ona tepki verebildiği ilk işlem gününe (event_day) bağlanır; geçmiş analizde yalnızca
event_day <= analiz günü olan duyurular görünür.
"""
from pathlib import Path

import pandas as pd

from src.data.kap_dates import KAP_OID, cache_file, event_day, fetch, windows
from src.data.prices import get_index_history

CACHE = Path("data/kap_oda")
NEWS_START = pd.Timestamp("2021-11-01")      # fiyat verisinin ilk günü
NEWS_FILE = "reports/kap_news.parquet"
DETAIL_URL = "https://www.kap.org.tr/tr/Bildirim/{}"


def parse_news(ticker: str, data: list) -> list:
    """KAP yanıtından haber satırları. Yayın zamanı olmayan kayıt atılır."""
    rows = []
    for item in data:
        b = item["disclosureBasic"]
        if not b.get("publishDate"):
            continue
        rows.append(dict(
            ticker=ticker,
            publish_time=pd.to_datetime(b["publishDate"], format="%d.%m.%Y %H:%M:%S"),
            title=(b.get("title") or "").strip(),
            summary=(b.get("summary") or "").strip(),
            related_stocks=b.get("relatedStocks") or "",
            kap_index=b["disclosureIndex"],
            url=DETAIL_URL.format(b["disclosureIndex"]),
            source="KAP",
        ))
    return rows


def kap_news(tickers=None, today=None) -> pd.DataFrame:
    """30 hissenin ÖDA duyuruları, 90 günlük dilimlerle (önbellekte olan dilim tekrar indirilmez)."""
    tickers = tickers or list(KAP_OID)
    w = windows(today, first=NEWS_START)
    todo = sum(not cache_file(t, s, e, CACHE).exists() for t in tickers for s, e in w)
    print(f"dilim: {len(tickers) * len(w)}  KAP'a atılacak istek: {todo}", flush=True)
    rows = []
    for t in tickers:
        for s, e in w:
            rows += parse_news(t, fetch(t, s, e, types=("ODA",), cache=CACHE))
        print(f"{t}: tamam", flush=True)
    news = (pd.DataFrame(rows).drop_duplicates(["ticker", "kap_index"])
            .sort_values(["publish_time", "ticker"]).reset_index(drop=True))
    news["event_day"] = event_day(news["publish_time"], get_index_history("XU100").index)
    return news


def load_kap_news(path=NEWS_FILE) -> pd.DataFrame:
    """Kaydedilmiş haber tablosu (KAP'a istek atmaz)."""
    return pd.read_parquet(path)


def visible_asof_news(news: pd.DataFrame, date) -> pd.DataFrame:
    """Analiz günü 'date' kapanışında bilinen duyurular: event_day <= date."""
    return news[news["event_day"] <= pd.Timestamp(date)]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    news = kap_news()
    news.to_parquet(NEWS_FILE, index=False)
    print(f"\nduyuru: {len(news)}  dönem: {news['publish_time'].min()} -> {news['publish_time'].max()}")
    print(f"event_day bulunamayan (son işlem gününden sonra): {news['event_day'].isna().sum()}")
    print("\nHisse başına duyuru:")
    print(news.groupby("ticker").size().sort_values(ascending=False).to_string())
    print("\nEn sık 15 başlık:")
    print(news["title"].value_counts().head(15).to_string())
    print("\nTURSG Ekim 2026:")
    m = (news["ticker"] == "TURSG") & (news["publish_time"].between("2026-10-01", "2026-11-07"))
    print(news.loc[m, ["publish_time", "event_day", "title", "summary"]].to_string(index=False))
