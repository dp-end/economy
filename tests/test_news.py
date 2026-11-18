from pathlib import Path

import feedparser
import pandas as pd
import pytest

from src.data.kap_news import parse_news, visible_asof_news
from src.data.news_tags import (NEWS_EVENTS, categories, find_tickers, get_news_events,
                                load_news_events, scope, tr_lower)
from src.data.rss_news import parse_feed, update_archive

RSS_SAMPLE = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>TCMB politika faizini sabit bıraktı</title><link>https://ornek.com/a</link>
<pubDate>Thu, 23 Oct 2026 11:00:00 GMT</pubDate></item>
<item><title>Bağlantısı olmayan başlık</title><pubDate>Thu, 23 Oct 2026 11:00:00 GMT</pubDate></item>
<item><title>Zamanı olmayan başlık</title><link>https://ornek.com/b</link></item>
</channel></rss>"""


def test_tr_lower_handles_turkish_i():
    assert tr_lower("İSTANBUL IŞIK") == "istanbul ışık"


def test_categories_rules():
    assert categories("Genel Müdür Ayrılışı") == ["yonetim"]
    assert categories("Pay Geri Alım Programı Başlatılması") == ["geri_alim"]
    assert categories("TCMB politika faizini sabit bıraktı") == ["faiz"]
    assert categories("Genel Kurul İşlemlerine İlişkin Bildirim") == ["genel_kurul"]   # 'kurul' kur sayılmaz
    assert categories("Dolar/TL kuru rekor kırdı")[0] == "kur"
    assert categories("COP31'de gönüllüler görev aldı") == ["diger"]


def test_find_tickers_by_name_and_code():
    assert find_tickers("ASELSAN ve ROKETSAN'dan dev anlaşma") == ["ASELS"]
    assert find_tickers("THYAO ve PGSUS hisseleri yükseldi") == ["THYAO", "PGSUS"]
    assert find_tickers("Türk Hava Yolları yeni uçak aldı") == ["THYAO"]
    assert find_tickers("Bimeks konkordato ilan etti") == []          # BİM ile karışmaz
    assert find_tickers("Kurban Bayramı tatili") == []


def test_scope():
    assert scope("ASELS", ["sozlesme"]) == "sirket"
    assert scope(None, ["faiz"]) == "makro"
    assert scope(float("nan"), ["diger"]) == "genel"


def test_parse_feed_converts_to_istanbul_time_and_drops_incomplete():
    df = parse_feed("Ornek", feedparser.parse(RSS_SAMPLE), fetched_at="2026-10-23 15:00:00")
    assert len(df) == 1                                               # linksiz ve zamansız atıldı
    assert df.iloc[0]["publish_time"] == pd.Timestamp("2026-10-23 14:00:00")   # 11:00 GMT = 14:00 TR
    assert df.iloc[0]["url"] == "https://ornek.com/a" and df.iloc[0]["source"] == "Ornek"


def test_update_archive_keeps_first_seen(tmp_path: Path):
    f = tmp_path / "arc.parquet"
    row = dict(source="A", publish_time=pd.Timestamp("2026-11-18 10:00"), title="x", url="u1")
    update_archive(pd.DataFrame([{**row, "fetched_at": pd.Timestamp("2026-11-18 11:00")}]), f)
    arc = update_archive(pd.DataFrame([{**row, "fetched_at": pd.Timestamp("2026-11-18 12:00")}]), f)
    assert len(arc) == 1 and arc.iloc[0]["fetched_at"] == pd.Timestamp("2026-11-18 11:00")


def test_parse_news_kap_item():
    item = {"disclosureBasic": {"publishDate": "15.10.2026 20:08:03", "title": "Özel Durum Açıklaması (Genel)",
                                "summary": "Genel Müdür Ayrılışı", "disclosureIndex": 123, "relatedStocks": None}}
    rows = parse_news("TURSG", [item, {"disclosureBasic": {"publishDate": None, "disclosureIndex": 1}}])
    assert len(rows) == 1
    r = rows[0]
    assert r["publish_time"] == pd.Timestamp("2026-10-15 20:08:03")
    assert r["url"] == "https://www.kap.org.tr/tr/Bildirim/123" and r["source"] == "KAP"


def test_get_news_events_asof_no_lookahead():
    news = pd.DataFrame(dict(
        ticker=["A", "A", "B"], categories=["faiz", "yonetim|hukuk_duzenleme", "faiz"],
        publish_time=pd.to_datetime(["2026-10-01 10:00", "2026-10-01 19:00", "2026-10-02 10:00"]),
        event_day=pd.to_datetime(["2026-10-01", "2026-10-02", pd.NaT])))
    assert len(get_news_events(news, asof="2026-10-01")) == 1        # 19:00 duyurusu ertesi gün görünür
    assert len(get_news_events(news, asof="2026-10-02")) == 2        # event_day'i olmayan görünmez
    assert len(get_news_events(news, ticker="A", topic="hukuk_duzenleme")) == 1
    assert len(get_news_events(news, start="2026-10-02", end="2026-10-02")) == 1
    assert len(visible_asof_news(news, "2026-10-01")) == 1


@pytest.mark.skipif(not Path(NEWS_EVENTS).exists(), reason="önce: python -m src.data.news_tags")
def test_real_news_table_has_source_and_time():
    news = load_news_events()
    assert news["url"].str.startswith("http").all()                  # her kayıtta kaynak bağlantısı
    assert news["publish_time"].notna().all() and news["source"].notna().all()
    kap = news[news["source"] == "KAP"]
    assert kap["ticker"].nunique() == 30 and (kap["scope"] == "sirket").all()
    ok = kap["event_day"].notna()
    assert (kap.loc[ok, "event_day"] >= kap.loc[ok, "publish_time"].dt.normalize()).all()   # geçmişe kaymaz
    t = kap[(kap["ticker"] == "TURSG") & (kap["publish_time"] == "2026-10-15 20:08:03")]
    assert len(t) == 1 and t.iloc[0]["event_day"] == pd.Timestamp("2026-10-16")
    assert "yonetim" in t.iloc[0]["categories"]
