"""Ücretsiz RSS beslemelerinden güncel ekonomi başlıkları.

RSS yalnızca son birkaç günü verir, bu yüzden her çalıştırma bir arşive eklenir (data/rss).
Sadece başlık, bağlantı, kaynak ve yayın zamanı saklanır; haber metni indirilmez.
fetched_at sütunu, başlığın sisteme ne zaman girdiğini gösterir (point-in-time kaydı).
"""
import calendar
from pathlib import Path

import feedparser
import pandas as pd

from src.data.kap_dates import event_day
from src.data.prices import get_index_history

FEEDS = {
    "AA": "https://www.aa.com.tr/tr/rss/default?cat=ekonomi",
    "BloombergHT": "https://www.bloomberght.com/rss",
    "TRT Haber": "https://www.trthaber.com/ekonomi_articles.rss",
}
ARCHIVE = Path("data/rss/archive.parquet")
TZ = "Europe/Istanbul"
COLUMNS = ["source", "publish_time", "title", "url", "fetched_at"]


def to_local(struct_time) -> pd.Timestamp:
    """feedparser'ın UTC zamanını İstanbul saatine çevirir (saat dilimi bilgisi atılır)."""
    if struct_time is None:
        return pd.NaT
    ts = pd.Timestamp(calendar.timegm(struct_time), unit="s", tz="UTC")
    return ts.tz_convert(TZ).tz_localize(None)


def parse_feed(source: str, feed, fetched_at=None) -> pd.DataFrame:
    """feedparser çıktısından satırlar. Başlığı, bağlantısı ya da yayın zamanı olmayan kayıt atılır."""
    fetched_at = pd.Timestamp(fetched_at or pd.Timestamp.now(tz=TZ).tz_localize(None)).floor("s")
    rows = []
    for e in feed.entries:
        t = to_local(e.get("published_parsed") or e.get("updated_parsed"))
        title, link = (e.get("title") or "").strip(), (e.get("link") or "").strip()
        if title and link and pd.notna(t):
            rows.append(dict(source=source, publish_time=t, title=title, url=link, fetched_at=fetched_at))
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_rss(feeds=FEEDS) -> pd.DataFrame:
    """Tüm beslemeleri bir kez okur. Çalışmayan besleme uyarı verir, diğerlerini durdurmaz."""
    parts = []
    for source, url in feeds.items():
        f = feedparser.parse(url)
        df = parse_feed(source, f)
        if df.empty:
            print(f"  uyarı: {source} boş döndü ({f.get('bozo_exception', 'kayıt yok')})", flush=True)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def update_archive(new: pd.DataFrame, path: Path = ARCHIVE) -> pd.DataFrame:
    """Yeni başlıkları arşive ekler. Aynı bağlantı ikinci kez gelirse ilk kaydı (ilk görüldüğü an) kalır."""
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    arc = pd.concat([old, new], ignore_index=True)
    arc = (arc.sort_values("fetched_at").drop_duplicates("url", keep="first")
           .sort_values("publish_time").reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    arc.to_parquet(path, index=False)
    return arc


def load_rss_news(path: Path = ARCHIVE) -> pd.DataFrame:
    """Arşiv + event_day (piyasanın tepki verebildiği ilk işlem günü). Ağa çıkmaz."""
    arc = pd.read_parquet(path)
    arc["event_day"] = event_day(arc["publish_time"], get_index_history("XU100").index)
    return arc


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    new = fetch_rss()
    arc = update_archive(new)
    print(f"bu çalıştırmada gelen: {len(new)}  arşivdeki toplam: {len(arc)}")
    print(arc.groupby("source")["publish_time"].agg(["count", "min", "max"]).to_string())
    print("\nSon 10 başlık:")
    print(arc.tail(10)[["source", "publish_time", "title"]].to_string(index=False))
