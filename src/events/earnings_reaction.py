
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from src.data.fundamentals import FINANCIALS, load_all
from src.data.kap_dates import load_kap_table
from src.data.pit_fundamentals import panel, pit_all
from src.data.prices import get_index_history, get_stock_history
from src.events.calendar_patterns import bh_fdr
from src.events.sector_catchup import quake_flags_wide

# pencere adı -> (başlangıç, bitiş) işlem günü; 0 = event_day (piyasanın tepki verebildiği ilk gün)
WINDOWS = {"pre5": (-5, -1), "d0": (0, 0), "d0_1": (0, 1), "d0_4": (0, 4), "post1_5": (1, 5), "post1_20": (1, 20)}
MIN_STOCKS = 15            # bir çeyrekte sıralama yapmak için en az hisse
GROUPS = ["alt", "orta", "ust"]
# Açıklama gününde bilinebilen temel veriler (src/data/pit_fundamentals.py, Adım 1b). TL tutarları o raporun parasıyla.
PIT_COLS = ["net_income", "net_income_prev", "ni_yoy", "ni_qoq", "loss", "net_income_ttm", "revenue", "gross_profit",
            "op_income", "equity", "total_assets", "roe_q", "leverage", "net_margin", "revenue_yoy", "op_income_yoy",
            "margin_chg", "roe_chg", "era_flow", "era_stock", "unit_flow", "unit_stock", "pit_dropped"]


def car(close: pd.Series, xu: pd.Series, pos: int, a: int, b: int) -> float:
    """[a, b] penceresinde hisse getirisi eksi XU100 getirisi. Pencere veride yoksa NaN."""
    lo, hi = pos + a - 1, pos + b                 # lo: pencereden önceki son kapanış
    if lo < 0 or hi >= len(close):
        return np.nan
    return (close.iloc[hi] / close.iloc[lo] - 1) - (xu.iloc[hi] / xu.iloc[lo] - 1)


# ---------------------------------------------------------------- üçte bir sıralaması (Adım 1b, kullanıcı kararı A)
# Bir olayın üçte biri, açıklama gününde bilinen bir ölçütle belirlenir: aynı hisse grubunun BİR ÖNCEKİ çeyrekteki ve
# o güne kadar bilinen değerlerinin 1/3 ve 2/3 noktaları (en az MIN_STOCKS değer). Aynı çeyreğin bütün hisseleri
# arasında sıralamak (eski yöntem) erken açıklayanın sınıfını sonra açıklayanların kârına bağlıyordu: geleceğe bakış.
# B (aynı çeyrekte o güne kadar açıklayanlar arasında) Rapor 4'te sağlamlık kontrolü olarak gösterilir.

def _label(x: float, lo: float, hi: float) -> str:
    return "alt" if x <= lo else ("orta" if x <= hi else "ust")


def _pool(col: str, tickers) -> pd.DataFrame:
    """Eşik havuzu: her (hisse, çeyrek) değeri ve elimizdeki sürümünün bilindiği gün (zamanında bilinen panel).
    2021 değerleri KAP tablosundan önce: bilinme günü yasal son tarih (pit_fundamentals.known_from)."""
    parts = []
    for t in tickers:
        pan = panel(t)
        if col not in pan.columns:
            continue
        s = pan[[col, f"{col}__known"]].dropna()
        parts.append(pd.DataFrame({"period_end": s.index, "value": s[col].to_numpy(),
                                   "known": pd.to_datetime(s[f"{col}__known"]).to_numpy()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["period_end", "value", "known"])


def terciles_prior(df: pd.DataFrame, col: str = "ni_yoy", min_n: int = MIN_STOCKS) -> pd.DataFrame:
    """A (ana yöntem): eşikler bir önceki çeyrekten, yalnızca olay gününe kadar bilinen değerlerle. Havuz, df'deki
    hisselerle sınırlıdır (ör. finans dışı alt kümede yalnızca finans dışı hisseler)."""
    d = df.copy()
    pool = _pool(col, d["ticker"].unique())
    d["tercile"], d["tercile_ref"] = None, pd.NaT
    d["tercile_n"], d["tercile_lo"], d["tercile_hi"] = np.nan, np.nan, np.nan
    for i, r in d.iterrows():
        if pd.isna(r[col]) or pd.isna(r["event_day"]):
            continue
        ref = pd.Timestamp(r["period_end"]) - pd.offsets.QuarterEnd(1)
        p = pool.loc[(pool["period_end"] == ref) & (pool["known"] <= r["event_day"]), "value"]
        if len(p) < min_n:
            continue
        lo, hi = p.quantile([1 / 3, 2 / 3])
        d.at[i, "tercile"] = _label(r[col], lo, hi)
        d.at[i, "tercile_ref"], d.at[i, "tercile_n"], d.at[i, "tercile_lo"], d.at[i, "tercile_hi"] = ref, len(p), lo, hi
    return d


def terciles_expanding(df: pd.DataFrame, col: str = "ni_yoy", min_n: int = MIN_STOCKS) -> pd.DataFrame:
    """B (sağlamlık kontrolü): aynı çeyrekte olay gününe kadar açıklamış hisseler arasında sıralama (kendisi dahil).
    Erken açıklayanlar yeterli karşılaştırma olmadığı için sıralanmaz."""
    d = df.copy()
    d["tercile"] = None
    for _, g in d.dropna(subset=[col, "event_day"]).groupby("period_end"):
        for i, r in g.iterrows():
            same = g.loc[g["event_day"] <= r["event_day"], col]
            if len(same) >= min_n:
                pct = same.rank(pct=True).loc[i]
                d.at[i, "tercile"] = "alt" if pct <= 1 / 3 else ("orta" if pct <= 2 / 3 else "ust")
    return d


def terciles_expost(df: pd.DataFrame, col: str = "ni_yoy") -> pd.DataFrame:
    """Eski yöntem, YALNIZCA karşılaştırma için: çeyreğin bütün hisseleri arasında sıralama. Erken açıklayanın sınıfı
    sonra açıklayanların kârına bağlı olduğu için geleceğe bakar; stratejide ve harness'te kullanılmaz."""
    d = df.copy()
    n = d.groupby("period_end")[col].transform("count")
    pct = d.groupby("period_end")[col].rank(pct=True)
    d["tercile"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=GROUPS).astype(object)
    d.loc[d[col].isna() | (n < MIN_STOCKS), "tercile"] = np.nan
    return d


add_terciles = terciles_prior          # ana yöntem; Rapor 4 ve strateji bu adla kullanır


@lru_cache(maxsize=1)
def earnings_events() -> pd.DataFrame:
    """Her (hisse, çeyrek) bilanço açıklaması: açıklama gününde bilinebilen temel veriler, sıralama ve pencere
    getirileri. TMS 29 şirketlerinin sonradan düzeltilmiş rakamları kendi çeyreklerinde kullanılmaz (boş kalır)."""
    kap = load_kap_table()
    pit = pit_all().drop(columns=["event_day"])
    fund = pit[["ticker", "period_end"] + [c for c in PIT_COLS if c in pit.columns]]
    ev = add_terciles(kap.merge(fund, on=["ticker", "period_end"], how="left"))
    ev["loss"] = ev["loss"].astype("boolean")          # bilinmiyorsa boş (NA), False değil
    xu = get_index_history("XU100")["Close"]
    quake = quake_flags_wide(xu.index, before=20, after=5)
    rows = []
    for t, g in ev.groupby("ticker"):
        close = get_stock_history(t)["Close"]
        x = xu.reindex(close.index)
        for i, day in zip(g.index, g["event_day"]):
            if pd.isna(day) or day not in close.index:
                continue
            pos = close.index.get_loc(day)
            row = {"i": i, "quake_window": bool(quake.get(day, False))}
            row.update({k: car(close, x, pos, a, b) for k, (a, b) in WINDOWS.items()})
            rows.append(row)
    ev = ev.join(pd.DataFrame(rows).set_index("i"))
    ev["quake_window"] = ev["quake_window"].fillna(False).astype(bool)
    ev["financial"] = ev["ticker"].isin(FINANCIALS)
    return ev


def baseline() -> pd.Series:
    """Baz çizgisi: 30 hissenin herhangi bir gününde aynı pencerelerin ortalama anormal getirisi (%)."""
    xu = get_index_history("XU100")["Close"]
    keep = ~quake_flags_wide(xu.index, before=20, after=5).to_numpy()
    close = pd.DataFrame({t: get_stock_history(t)["Close"] for t in load_all()})
    out = {}
    for k, (a, b) in WINDOWS.items():
        ar = (close.shift(-b) / close.shift(1 - a) - 1).sub(xu.shift(-b) / xu.shift(1 - a) - 1, axis=0)
        out[k] = ar[keep].stack().mean() * 100
    return pd.Series(out, name="baz")


def _clean(ev: pd.DataFrame) -> pd.DataFrame:
    return ev[ev["event_day"].notna() & ~ev["quake_window"]]


def summarize(ev: pd.DataFrame, by: str = "tercile") -> pd.DataFrame:
    """Gruba göre ortalama anormal getiri (%) ve pozitif oran (%). Deprem penceresi hariç."""
    rows = []
    for key, g in _clean(ev).dropna(subset=[by]).groupby(by):
        row = {by: key, "n": len(g)}
        for k in WINDOWS:
            s = g[k].dropna()
            row[f"{k}_mean"] = s.mean() * 100
            row[f"{k}_hit"] = (s > 0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


def tercile_spread(d: pd.DataFrame, windows, min_ranked: int = MIN_STOCKS) -> pd.DataFrame:
    """Çeyrek başına üst eksi alt üçte bir ortalaması. Bir çeyrek, en az min_ranked olay sıralanmışsa gözlem olur
    (her çeyrek testte eşit ağırlıklı; birkaç olaylık çeyrek gürültüyü büyütür)."""
    r = d.dropna(subset=["tercile"])
    r = r[r.groupby("period_end")["tercile"].transform("size") >= min_ranked]
    g = r.groupby(["period_end", "tercile"])[list(windows)].mean()
    if not {"ust", "alt"} <= set(g.index.get_level_values("tercile")):
        return pd.DataFrame(columns=list(windows))
    return (g.xs("ust", level="tercile") - g.xs("alt", level="tercile")).dropna(how="all")


def quarterly_spread(ev: pd.DataFrame) -> pd.DataFrame:
    """Her çeyrek için üst üçte bir eksi alt üçte bir ortalama anormal getiri (puan)."""
    return tercile_spread(_clean(ev), WINDOWS) * 100


def spread_tests(spread: pd.DataFrame) -> pd.DataFrame:
    """Çeyrekler bağımsız gözlem sayılır (aynı çeyrekteki hisseler değil). t testi + çoklu test düzeltmesi."""
    half = len(spread) // 2
    rows = []
    for k in spread.columns:
        s = spread[k].dropna()
        t = s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))
        rows.append(dict(pencere=k, n_ceyrek=len(s), ort_fark=s.mean(), t=t, p=2 * stats.t.sf(abs(t), len(s) - 1),
                         pozitif_ceyrek=int((s > 0).sum()),
                         ilk_yari=spread[k].iloc[:half].mean(), ikinci_yari=spread[k].iloc[half:].mean()))
    out = pd.DataFrame(rows).set_index("pencere")
    out["q_bh"] = bh_fdr(out["p"])
    out["p_bonf"] = np.minimum(out["p"] * len(out), 1.0)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ev = earnings_events()
    ev.to_parquet("reports/earnings_events.parquet")
    c = _clean(ev)
    print(f"açıklama: {len(ev)}  tepki günü veride yok: {int(ev['event_day'].isna().sum())}  "
          f"deprem penceresi: {int(ev['quake_window'].sum())}  analizde: {len(c)}  sıralaması olan: {int(c['tercile'].notna().sum())}")
    by_t = summarize(ev).reindex(GROUPS)
    by_t.to_csv("reports/earnings_by_tercile.csv")
    print("\nNet kâr yıllık değişim sıralamasına göre anormal getiri (%, XU100'e göre):")
    print(by_t.round(2).to_string())
    print("\nBaz çizgisi (herhangi bir gün, aynı pencereler, %):")
    print(baseline().round(2).to_frame().T.to_string())
    print("\nZarar açıklayan (True) / kâr açıklayan (False):")
    print(summarize(ev, "loss").round(2).to_string())
    print("\nFinans dışı hisseler, sıralamaya göre:")
    print(summarize(ev[~ev["financial"]]).reindex(GROUPS).round(2).to_string())
    sp = quarterly_spread(ev)
    sp.to_csv("reports/earnings_spread_by_quarter.csv")
    tests = spread_tests(sp)
    tests.to_csv("reports/earnings_spread_tests.csv")
    print("\nÜst eksi alt fark (puan), çeyrekler üzerinden:")
    print(tests.round(3).to_string())
    print("\nTUPRS örneği:")
    cols = ["period_end", "event_day", "ni_yoy", "tercile", "pre5", "d0", "d0_4", "post1_20"]
    print(ev[ev["ticker"] == "TUPRS"][cols].tail(8).to_string(index=False, float_format="%.3f"))