"""Haber etiketleme: şirket, sektör, kategori ve kapsam. Kurallar anahtar kelimeye dayanır (LLM yok).

Aynı başlık her çalıştırmada aynı etiketi alır. Haber yalnızca bağlamdır: hiçbir sayısal
piyasa/bilanço verisinin yerine geçmez. get_news_events() MCP aracının (Adım 12) temelidir.
"""
import re

import pandas as pd

from src.data.kap_news import load_kap_news
from src.data.rss_news import load_rss_news
from src.data.universe import load_universe

NEWS_EVENTS = "reports/news_events.parquet"

# Sıra önemli: ilk eşleşen kategori "ana kategori" olur. Desenler küçük harfli metinde aranır.
CATEGORY_RULES = {
    "rutin": [r"bağımsız denetim kuruluş", r"sorumluluk sigorta", r"kurumsal yönetim ilke",
              r"kayıtlı sermaye tavan", r"yatırımcı sunum", r"bilgilendirme sunum",
              r"ilişkili taraf", r"sürdürülebilirlik"],
    "bilanco": [r"finansal rapor", r"finansal sonuç", r"finansal tablo", r"bilanço", r"çeyrek sonuç",
                r"net kâr", r"net kar", r"faaliyet rapor"],
    "temettu": [r"kar payı", r"kâr payı", r"temettü"],
    "geri_alim": [r"geri alım", r"geri alın"],
    "sermaye": [r"sermaye artırım", r"sermaye azaltım", r"bedelsiz", r"bedelli"],
    "derecelendirme": [r"derecelendirme", r"kredi notu", r"\brating"],
    "borclanma": [r"borçlanma aracı", r"tahvil", r"\bbono", r"sukuk", r"kira sertifika", r"ihraç tavan", r"eurobond",
                  r"kredi", r"sendikasyon", r"sermaye piyasası aracı"],
    "sozlesme": [r"sözleşme", r"ihale", r"sipariş", r"yeni iş", r"iş ilişkisi", r"anlaşma"],
    "birlesme_satinalma": [r"birleşme", r"satın al", r"pay devri", r"paylar\w* devri", r"pay satış", r"iştirak",
                           r"bölünme", r"duran varlık", r"ortaklık"],
    "yonetim": [r"genel müdür", r"yönetim kurulu", r"atama", r"ayrılış", r"istifa", r"görevden", r"yönetici değişikli",
                r"yönetim değişikli"],
    "faaliyet": [r"yeni proje", r"yeni hat", r"mağaza açılış", r"trafik sonuç", r"prim üretim",
                 r"yatırım karar", r"kapasite", r"üretim tesis"],
    "hukuk_duzenleme": [r"\bdava", r"\bceza", r"soruşturma", r"rekabet kurul", r"\bspk\b", r"\bbddk\b",
                        r"\bepdk\b", r"lisans", r"düzenleme", r"\bvergi", r"ifadeye", r"kanun", r"damping", r"gümrük"],
    "genel_kurul": [r"genel kurul"],
    "faiz": [r"faiz", r"\bppk\b", r"\bfed\b", r"\becb\b", r"merkez bankası", r"\btcmb\b"],
    "kur": [r"\bkur\b", r"\bkurlar", r"döviz", r"\bdolar", r"\beuro\b", r"\bavro\b"],
    "enflasyon": [r"enflasyon", r"\btüfe\b", r"\büfe\b"],
    "emtia": [r"petrol", r"\bbrent", r"\baltın", r"doğalgaz", r"doğal gaz", r"emtia", r"bakır",
              r"çelik fiyat"],
    "jeopolitik": [r"savaş", r"saldırı", r"ateşkes", r"yaptırım", r"gerilim", r"çatışma", r"jeopolitik"],
    "pay_alim_satim": [r"pay alım satım", r"paylarında alım", r"paylarında satım"],
    "varant": [r"varant"],
}
MACRO_CATEGORIES = {"faiz", "kur", "enflasyon", "emtia", "jeopolitik"}

# RSS başlıklarında şirket tanıma (KAP duyuruları zaten hisseye bağlı gelir).
COMPANY_ALIASES = {
    "ASELS": [r"aselsan"], "TUPRS": [r"tüpraş"], "BIMAS": [r"\bbim\b", r"bim birleşik"],
    "THYAO": [r"\bthy\b", r"türk hava yolları"], "AKBNK": [r"akbank"], "KCHOL": [r"koç holding"],
    "EREGL": [r"erdemir", r"ereğli demir"], "TCELL": [r"turkcell"],
    "CCOLA": [r"coca-cola içecek", r"coca cola içecek"], "MGROS": [r"migros"], "SASA": [r"\bsasa\b"],
    "TRALT": [r"türk altın", r"koza altın"], "FROTO": [r"ford otosan"], "ENKAI": [r"\benka\b"],
    "PGSUS": [r"pegasus"], "MPARK": [r"mlp sağlık", r"medical park"], "EKGYO": [r"emlak konut"],
    "GUBRF": [r"gübre fabrika", r"gübretaş"], "TURSG": [r"türkiye sigorta"], "ENJSA": [r"enerjisa"],
    "TTKOM": [r"türk telekom"], "OYAKC": [r"oyak çimento"], "AKSEN": [r"aksa enerji"],
    "ISMEN": [r"iş yatırım"], "ARCLK": [r"arçelik"], "ECILC": [r"eczacıbaşı ilaç"],
    "MAVI": [r"mavi giyim"], "PETKM": [r"petkim"], "TOASO": [r"tofaş"], "KRDMD": [r"kardemir"],
}


def tr_lower(text: str) -> str:
    """Türkçe küçük harf: 'I' -> 'ı', 'İ' -> 'i' (Python'un lower()'ı bunu doğru yapmaz)."""
    return (text or "").replace("I", "ı").replace("İ", "i").lower()


def categories(text: str) -> list:
    """Metne uyan tüm kategoriler, CATEGORY_RULES sırasıyla. Hiçbiri uymazsa ['diger']."""
    t = tr_lower(text)
    found = [c for c, pats in CATEGORY_RULES.items() if any(re.search(p, t) for p in pats)]
    return found or ["diger"]


def find_tickers(text: str) -> list:
    """Başlıkta geçen çekirdek hisseler: şirket adı ya da büyük harfli hisse kodu."""
    t = tr_lower(text)
    return [k for k, pats in COMPANY_ALIASES.items()
            if re.search(rf"\b{k}\b", text or "") or any(re.search(p, t) for p in pats)]


def scope(ticker, cats: list) -> str:
    """'sirket' (bir hisseye bağlı), 'makro' (faiz/kur/enflasyon/emtia/jeopolitik) ya da 'genel'."""
    if isinstance(ticker, str) and ticker:
        return "sirket"
    return "makro" if MACRO_CATEGORIES & set(cats) else "genel"


def tag(df: pd.DataFrame, text_cols=("title",)) -> pd.DataFrame:
    """categories (| ile ayrılmış), category (ana), sector ve scope sütunlarını ekler."""
    out = df.copy()
    text = out[list(text_cols)].fillna("").agg(" ".join, axis=1)
    cats = text.map(categories)
    out["categories"] = cats.map("|".join)
    out["category"] = cats.map(lambda c: c[0])
    sector = load_universe().set_index("ticker")["sector_pdf"]
    out["sector"] = out["ticker"].map(sector)
    out["scope"] = [scope(t, c) for t, c in zip(out["ticker"], cats)]
    return out


def build_news_table(kap: pd.DataFrame = None, rss: pd.DataFrame = None) -> pd.DataFrame:
    """KAP + RSS tek tablo. RSS başlığı birden çok hisse içeriyorsa her hisse için bir satır olur."""
    kap = load_kap_news() if kap is None else kap
    rss = load_rss_news() if rss is None else rss
    k = tag(kap.assign(fetched_at=pd.NaT), text_cols=("title", "summary"))
    r = rss.assign(summary="", ticker=rss["title"].map(find_tickers)).explode("ticker")
    r = tag(r)
    cols = ["source", "publish_time", "event_day", "ticker", "sector", "scope", "category",
            "categories", "title", "summary", "url", "fetched_at"]
    out = pd.concat([k[cols], r[cols]], ignore_index=True)
    return out.sort_values(["publish_time", "source"]).reset_index(drop=True)


def get_news_events(news: pd.DataFrame, ticker=None, topic=None, start=None, end=None, asof=None):
    """Filtrelenmiş haber olayları. asof verilirse yalnızca o gün kapanışında bilinenler (event_day <= asof)."""
    m = pd.Series(True, index=news.index)
    if ticker:
        m &= news["ticker"] == ticker
    if topic:
        m &= news["categories"].str.split("|").map(lambda c: topic in c)
    if start:
        m &= news["publish_time"] >= pd.Timestamp(start)
    if end:
        m &= news["publish_time"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    if asof:
        m &= news["event_day"] <= pd.Timestamp(asof)
    return news[m]


def load_news_events(path=NEWS_EVENTS) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    news = build_news_table()
    news.to_parquet(NEWS_EVENTS, index=False)
    print(f"satır: {len(news)}")
    print(news.groupby(["source", "scope"]).size().to_string())
    print("\nAna kategori dağılımı (KAP):")
    print(news[news["source"] == "KAP"]["category"].value_counts().to_string())
    print("\nAna kategori dağılımı (RSS):")
    print(news[news["source"] != "KAP"]["category"].value_counts().to_string())
    print("\n'diger' kalan en sık 10 KAP başlık+özet:")
    d = news[(news["source"] == "KAP") & (news["category"] == "diger")]
    print((d["title"] + " | " + d["summary"]).value_counts().head(10).to_string())
    print("\nRSS'te şirket bulunan başlıklar:")
    print(news[(news["source"] != "KAP") & news["ticker"].notna()][["ticker", "title"]].to_string(index=False))
