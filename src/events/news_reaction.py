"""KAP haberlerinden olumlu/olumsuz sinyal olayları, olay sonrası fiyat tepkisi ve strateji E bayrakları.

Kurallar (NEGATIVE / POSITIVE) sonuçlara bakılmadan önce sabitlendi (2026-11-18). Olay çalışması
yalnızca rapor içindir; kuralı değiştirmek için kullanılmaz (veri madenciliği olmaması için).
Haber, event_day (piyasanın tepki verebildiği ilk işlem günü) kapanışından itibaren bilinir.
"""
import re

import numpy as np
import pandas as pd
from scipy import stats

from src.data.news_tags import load_news_events, tr_lower
from src.data.prices import get_index_history, get_stock_history
from src.events.calendar_patterns import bh_fdr
from src.events.earnings_reaction import WINDOWS, baseline, car
from src.events.sector_catchup import quake_flags_wide

NEWS_WINDOW = 10           # haber bayrağı kaç işlem günü geçerli (şok bayrağıyla aynı pencere)
NEGATIVE = {
    "yonetici_ayrilis": r"ayrılış|istifa|görevden al|görevinden ayr",
    "bilanco_erteleme": r"ertelen|ek süre",
    "dava_ceza": r"\bdava|\bceza|soruşturma|idari para",
    "tasfiye_iflas": r"tasfiye|iflas|konkordato",
}
POSITIVE = {
    "yeni_is_sozlesme": r"yeni iş ilişkisi|sözleşme imza|sipariş|ihale(?:yi)? kazan",
    "geri_alim_baslangic": r"geri alım programı başlat|geri alım programına başlan",
}
EVENTS_FILE = "reports/news_signal_events.parquet"


def classify(text: str) -> tuple:
    """(yön, tür) ya da (None, None). Olumsuz kurallar önce bakılır: ikisi birden uyarsa olumsuz sayılır."""
    t = tr_lower(text)
    for direction, rules in (("olumsuz", NEGATIVE), ("olumlu", POSITIVE)):
        for kind, pat in rules.items():
            if re.search(pat, t):
                return direction, kind
    return None, None


def news_signal_events(news: pd.DataFrame = None) -> pd.DataFrame:
    """KAP duyurularından sinyal olayları. Aynı hisse + aynı gün + aynı tür tek olay sayılır."""
    news = load_news_events() if news is None else news
    k = news[(news["source"] == "KAP") & news["event_day"].notna()].copy()
    lab = (k["title"] + " " + k["summary"]).map(classify)
    k["direction"], k["kind"] = lab.str[0], lab.str[1]
    k = k.dropna(subset=["direction"])
    return (k.drop_duplicates(["ticker", "event_day", "kind"])
            [["ticker", "event_day", "publish_time", "direction", "kind", "title", "summary", "url"]]
            .sort_values(["event_day", "ticker"]).reset_index(drop=True))


def add_returns(ev: pd.DataFrame) -> pd.DataFrame:
    """Her olay için Adım 7C ile aynı pencerelerde XU100'e göre anormal getiri. Deprem penceresi işaretlenir."""
    xu = get_index_history("XU100")["Close"]
    quake = quake_flags_wide(xu.index, before=20, after=5)
    out = ev.copy()
    for k in WINDOWS:
        out[k] = np.nan
    out["quake_window"] = out["event_day"].map(lambda d: bool(quake.get(d, False)))
    for t, g in out.groupby("ticker"):
        close = get_stock_history(t)["Close"]
        x = xu.reindex(close.index)
        for i, day in zip(g.index, g["event_day"]):
            if day in close.index:
                pos = close.index.get_loc(day)
                for k, (a, b) in WINDOWS.items():
                    out.at[i, k] = car(close, x, pos, a, b)
    return out


def summarize(ev: pd.DataFrame, base: pd.Series = None) -> pd.DataFrame:
    """Türe göre: olay sayısı, ortalama anormal getiri (%), baz çizgisine göre fark, t testi ve BH-FDR q."""
    base = baseline() if base is None else base
    e = ev[~ev["quake_window"]]
    rows = []
    for (direction, kind), g in e.groupby(["direction", "kind"]):
        row = {"direction": direction, "kind": kind, "n": len(g), "hisse": g["ticker"].nunique()}
        for k in ("pre5", "d0", "d0_4", "post1_5"):
            s = g[k].dropna() * 100
            row[f"{k}_mean"] = s.mean()
            row[f"{k}_vs_base"] = s.mean() - base[k]
            row[f"{k}_p"] = stats.ttest_1samp(s, base[k]).pvalue if len(s) >= 5 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    for k in ("pre5", "d0", "d0_4", "post1_5"):
        p = out[f"{k}_p"]
        out[f"{k}_q"] = np.nan
        out.loc[p.notna(), f"{k}_q"] = bh_fdr(p[p.notna()].to_numpy())
    out["az_olay"] = out["n"] < 20                 # az örneklem uyarısı
    return out.set_index(["direction", "kind"])


def news_flags(index: pd.DatetimeIndex, tickers: list, ev: pd.DataFrame = None, window: int = NEWS_WINDOW):
    """(olumsuz, olumlu) bool tabloları: hissede son 'window' işlem gününde (bugün dahil) o yönde haber var mı."""
    ev = news_signal_events() if ev is None else ev
    out = []
    for direction in ("olumsuz", "olumlu"):
        d = ev[ev["direction"] == direction]
        hits = pd.crosstab(d["event_day"], d["ticker"]).reindex(index=index, columns=tickers, fill_value=0)
        out.append(hits.rolling(window, min_periods=1).sum() > 0)
    return tuple(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ev = add_returns(news_signal_events())
    ev.to_parquet(EVENTS_FILE, index=False)
    print(f"sinyal olayı: {len(ev)}  (olumsuz {int((ev['direction'] == 'olumsuz').sum())}, "
          f"olumlu {int((ev['direction'] == 'olumlu').sum())})")
    print("\nHisse başına (en çok 8):")
    print(ev.groupby(["ticker", "direction"]).size().unstack(fill_value=0).sort_values("olumlu", ascending=False).head(8).to_string())
    base = baseline()
    print("\nBaz çizgisi (herhangi bir gün, %):", base[["pre5", "d0", "d0_4", "post1_5"]].round(3).to_dict())
    s = summarize(ev, base)
    s.to_csv("reports/news_reaction_summary.csv")
    cols = ["n", "hisse", "pre5_mean", "d0_mean", "d0_vs_base", "d0_q", "d0_4_mean", "d0_4_vs_base", "d0_4_q",
            "post1_5_vs_base", "post1_5_q", "az_olay"]
    print("\nHaber türüne göre anormal getiri (XU100'e göre, %; q = BH-FDR):")
    print(s[cols].round(3).to_string())
