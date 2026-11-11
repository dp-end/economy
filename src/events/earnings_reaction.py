
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from src.data.fundamentals import FINANCIALS, load_all
from src.data.kap_dates import load_kap_table
from src.data.prices import get_index_history, get_stock_history
from src.events.calendar_patterns import bh_fdr
from src.events.sector_catchup import quake_flags_wide

# pencere adı -> (başlangıç, bitiş) işlem günü; 0 = event_day (piyasanın tepki verebildiği ilk gün)
WINDOWS = {"pre5": (-5, -1), "d0": (0, 0), "d0_1": (0, 1), "d0_4": (0, 4), "post1_5": (1, 5), "post1_20": (1, 20)}
MIN_STOCKS = 15            # bir çeyrekte sıralama yapmak için en az hisse
GROUPS = ["alt", "orta", "ust"]


def car(close: pd.Series, xu: pd.Series, pos: int, a: int, b: int) -> float:
    """[a, b] penceresinde hisse getirisi eksi XU100 getirisi. Pencere veride yoksa NaN."""
    lo, hi = pos + a - 1, pos + b                 # lo: pencereden önceki son kapanış
    if lo < 0 or hi >= len(close):
        return np.nan
    return (close.iloc[hi] / close.iloc[lo] - 1) - (xu.iloc[hi] / xu.iloc[lo] - 1)


def add_terciles(df: pd.DataFrame, col: str = "ni_yoy") -> pd.DataFrame:
    """Aynı çeyrekteki hisseler arasında sıralama: alt / orta / ust üçte bir (enflasyonu sıfırlar)."""
    d = df.copy()
    n = d.groupby("period_end")[col].transform("count")
    pct = d.groupby("period_end")[col].rank(pct=True)
    d["tercile"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=GROUPS).astype(object)
    d.loc[d[col].isna() | (n < MIN_STOCKS), "tercile"] = np.nan
    return d


@lru_cache(maxsize=1)
def earnings_events() -> pd.DataFrame:
    """Her (hisse, çeyrek) bilanço açıklaması: temel veriler, sıralama ve pencere getirileri."""
    kap = load_kap_table()
    fund = pd.concat(load_all(), names=["ticker"]).reset_index()[["ticker", "period_end", "net_income", "ni_yoy", "ni_qoq", "loss"]]
    ev = add_terciles(kap.merge(fund, on=["ticker", "period_end"], how="left"))
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


def quarterly_spread(ev: pd.DataFrame) -> pd.DataFrame:
    """Her çeyrek için üst üçte bir eksi alt üçte bir ortalama anormal getiri (puan)."""
    g = _clean(ev).dropna(subset=["tercile"]).groupby(["period_end", "tercile"])[list(WINDOWS)].mean() * 100
    return (g.xs("ust", level="tercile") - g.xs("alt", level="tercile")).dropna(how="all")


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