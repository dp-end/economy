
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.prices import get_stock_history
from src.data.universe import load_universe
from src.events.technical_events import quake_flags
from src.events.sector_catchup import quake_flags_wide

WEEKDAYS = ["Pzt", "Sal", "Çar", "Per", "Cum"]
STREAK_KS = (2, 3, 4)
PATTERN_H = (1, 3, 5)


def welch(a, b):
    """Welch t testi (iki yönlü p, normal yaklaşımı)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = (a.mean() - b.mean()) / se
    return t, math.erfc(abs(t) / math.sqrt(2))


def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg düzeltilmiş p (q değeri)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


@lru_cache(maxsize=1)
def _data() -> dict:
    u = load_universe()
    core = u[u["is_core"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].tolist()
    panel = pd.DataFrame({t: get_stock_history(t)["Close"] for t in core})
    ret = panel.pct_change()
    idx = panel.index
    gap = pd.Series(idx, index=idx).diff().dt.days
    normal = np.where(idx.day_of_week == 0, gap <= 3, gap <= 1)   # tatil sonrası günler normal değil
    valid = pd.Series(normal, index=idx) & ~quake_flags(idx, max_h=0)
    return dict(core=core, panel=panel, ret=ret, valid=valid)


# ---------- A) Haftanın günü ----------
def weekday_portfolio(ret: pd.DataFrame, valid: pd.Series) -> pd.DataFrame:
    port = ret.mean(axis=1)
    idx = ret.index
    dow = pd.Series(idx.day_of_week, index=idx)
    half = idx[len(idx) // 2]
    rows = []
    for d, name in enumerate(WEEKDAYS):
        a_m, b_m = valid & (dow == d), valid & (dow != d)
        t, p = welch(port[a_m], port[b_m])
        row = dict(day=name, n=int(a_m.sum()), mean=port[a_m].mean() * 100,
                   diff=(port[a_m].mean() - port[b_m].mean()) * 100, t=t, p=p)
        for lab, part in (("IS", idx < half), ("OOS", idx >= half)):
            am, bm = a_m & part, b_m & part
            row[f"diff_{lab}"] = (port[am].mean() - port[bm].mean()) * 100
        rows.append(row)
    out = pd.DataFrame(rows).set_index("day")
    out["q_bh"] = bh_fdr(out["p"].to_numpy())
    return out


def weekday_per_stock(ret: pd.DataFrame, valid: pd.Series) -> pd.DataFrame:
    dow = pd.Series(ret.index.day_of_week, index=ret.index)
    rows = []
    for t in ret.columns:
        r = ret[t]
        for d, name in enumerate(WEEKDAYS):
            a = r[valid & (dow == d)].dropna()
            b = r[valid & (dow != d)].dropna()
            tt, p = welch(a, b)
            rows.append(dict(ticker=t, day=name, diff=(a.mean() - b.mean()) * 100, t=tt, p=p))
    out = pd.DataFrame(rows)
    out["q_bh"] = bh_fdr(out["p"].to_numpy())
    out["p_bonf"] = np.minimum(out["p"] * len(out), 1.0)
    return out


# ---------- B) Çok günlü desenler ----------
def run_length(b: pd.Series) -> pd.Series:
    b = b.astype(int)
    return b.groupby((b == 0).cumsum()).cumsum()


def _apply_mask(df: pd.DataFrame, vec) -> pd.DataFrame:
    """Her sütuna aynı satır maskesini uygula (bool & bool)."""
    return pd.DataFrame(df.to_numpy() & np.asarray(vec, bool)[:, None], index=df.index, columns=df.columns)


def streak_masks(ret: pd.DataFrame, valid: pd.Series, ks=STREAK_KS) -> dict:
    """(yön, k) -> k. ardışık günün kapandığı günler. Geçersiz gün seriyi böler."""
    out = {}
    for side, cond in (("down", ret < 0), ("up", ret > 0)):
        cond = _apply_mask(cond, valid.to_numpy())
        rl = cond.apply(run_length)
        for k in ks:
            out[(side, k)] = rl == k
    return out


def pattern_table(ret, panel, valid, ks=STREAK_KS, horizons=PATTERN_H) -> pd.DataFrame:
    masks = streak_masks(ret, valid, ks)
    idx = ret.index
    flag = quake_flags_wide(idx, before=max(horizons), after=0).to_numpy()
    half = idx[len(idx) // 2]
    ab = {}
    for h in horizons:
        fwd = panel.shift(-h) / panel - 1
        ab[h] = fwd.sub(fwd.mean(axis=1), axis=0)     # aynı günün hisse ortalamasından fark
    rows = []
    for (side, k), m in masks.items():
        m = _apply_mask(m, ~flag)
        for h in horizons:
            A = ab[h].where(m)
            by_date = A.mean(axis=1).dropna()
            n_dates = len(by_date)
            mean = by_date.mean()
            t = mean / (by_date.std(ddof=1) / np.sqrt(n_dates)) if n_dates > 2 else np.nan
            rows.append(dict(
                pattern=f"{side}{k}", h=h, n_events=int(A.notna().sum().sum()), n_dates=n_dates,
                mean_ab=mean * 100, t=t, p=math.erfc(abs(t) / math.sqrt(2)),
                IS=by_date[by_date.index < half].mean() * 100,
                OOS=by_date[by_date.index >= half].mean() * 100))
    out = pd.DataFrame(rows).set_index(["pattern", "h"])
    out["q_bh"] = bh_fdr(out["p"].to_numpy())
    out["p_bonf"] = np.minimum(out["p"] * len(out), 1.0)
    return out


if __name__ == "__main__":
    d = _data()
    pd.set_option("display.width", 200)
    print(f"geçerli gün: {int(d['valid'].sum())} / {len(d['valid'])}")

    wp = weekday_portfolio(d["ret"], d["valid"])
    wp.to_csv("reports/weekday_portfolio.csv")
    print("\n--- A1) haftanın günü, 30 hissenin eşit ağırlıklı portföyü (getiri %, diff = diğer günlere göre) ---")
    print(wp.round(3).to_string())

    ws = weekday_per_stock(d["ret"], d["valid"])
    ws.to_csv("reports/weekday_per_stock.csv", index=False)
    print("\n--- A2) hisse bazında 150 test ---")
    print(f"ham p<0.05: {(ws['p'] < 0.05).sum()}  (sadece şansla beklenen ≈ {0.05 * len(ws):.1f})")
    print(f"FDR sonrası q<0.05: {(ws['q_bh'] < 0.05).sum()} | Bonferroni sonrası p<0.05: {(ws['p_bonf'] < 0.05).sum()}")
    print(ws.sort_values("p").head(5).round(4).to_string(index=False))

    pt = pattern_table(d["ret"], d["panel"], d["valid"])
    pt.to_csv("reports/streak_patterns.csv")
    print("\n--- B) art arda düşüş/yükseliş sonrası anormal getiri (aynı günün hisse ortalamasına göre, %) ---")
    print(pt.round(3).to_string())
    print(f"\nham p<0.05: {(pt['p'] < 0.05).sum()} / {len(pt)} | FDR q<0.05: {(pt['q_bh'] < 0.05).sum()} "
          f"| Bonferroni: {(pt['p_bonf'] < 0.05).sum()}")