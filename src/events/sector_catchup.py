
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.prices import get_index_history, get_stock_history
from src.data.universe import load_universe, build_sector_peer_groups
from src.indicators.relative import sector_lag
from src.events.technical_events import dedupe, QUAKE_START, QUAKE_END

FWD_H = (5, 10, 20)
CONFIGS = {"lag10_5pct": (10, 0.05), "lag20_10pct": (20, 0.10)}  # (geriye bakış günü, eşik)
MIN_GROUP = 3
EVENT_GAP = 10     # aynı hissede olaylar arası en az gün
LIMIT = 0.095      # günlük limit hareketi eşiği (BIST ±%10)


def quake_flags_wide(index: pd.DatetimeIndex, before: int = 20, after: int = 20) -> pd.Series:
    """Deprem penceresi: öncesi `before` gün (ileri getiri), sonrası `after` gün (geriye bakış)."""
    pos0 = index.searchsorted(QUAKE_START)
    pos1 = index.searchsorted(QUAKE_END, side="right") - 1
    pos = np.arange(len(index))
    return pd.Series((pos >= pos0 - before) & (pos <= pos1 + after), index=index)


def lag_triggers(panel: pd.DataFrame, groups: dict, n: int, thr: float, side: str):
    """Koşulun yeni sağlandığı günler (True/False) ve lag tablosu. Sadece geçmiş veriyi kullanır."""
    lag = sector_lag(panel, groups, n)
    cond = (lag <= -thr) if side == "laggard" else (lag >= thr)
    first = cond & ~cond.shift(1, fill_value=False)
    return first, lag


@lru_cache(maxsize=1)
def _context() -> dict:
    u = load_universe()
    core = u[u["is_core"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].tolist()
    panel = pd.DataFrame({t: get_stock_history(t)["Close"] for t in u["ticker"]})
    xu = get_index_history()["Close"]
    groups, _ = build_sector_peer_groups()
    group_of = {t: g for g, members in groups.items() for t in members}
    eligible = [t for t in core if t in group_of and len(groups[group_of[t]]) >= MIN_GROUP]
    shock = (panel.pct_change().abs() >= LIMIT).astype(float).rolling(10).sum() >= 1
    fwd, exc, gap = {}, {}, {}
    for h in FWD_H:
        fwd[h] = panel.shift(-h) / panel - 1
        exc[h] = fwd[h].sub(xu.shift(-h) / xu - 1, axis=0)
        gap[h] = sector_lag(panel, groups, h).shift(-h)   # t+h anındaki "h günlük sektör farkı"
    return dict(core=core, panel=panel, groups=groups, group_of=group_of, eligible=eligible,
                shock=shock, fwd=fwd, exc=exc, gap=gap)


@lru_cache(maxsize=1)
def sector_events() -> pd.DataFrame:
    c = _context()
    q = quake_flags_wide(c["panel"].index)
    parts = []
    for cfg, (n, thr) in CONFIGS.items():
        for side in ("laggard", "leader"):
            first, lag = lag_triggers(c["panel"], c["groups"], n, thr, side)
            for t in c["eligible"]:
                idx = first.index[dedupe(first[t], EVENT_GAP).to_numpy()]
                if len(idx) == 0:
                    continue
                p = pd.DataFrame({
                    "ticker": t, "peer_group": c["group_of"][t], "config": cfg, "side": side,
                    "lag": lag[t].loc[idx].to_numpy(),
                    "shock": c["shock"][t].loc[idx].to_numpy(),
                    "quake_window": q.loc[idx].to_numpy(),
                }, index=idx)
                for h in FWD_H:
                    for k in ("fwd", "exc", "gap"):
                        p[f"{k}_{h}"] = c[k][h][t].loc[idx].to_numpy()
                parts.append(p)
    out = pd.concat(parts).sort_index()
    out.index.name = "date"
    return out


def catchup_history(panel: pd.DataFrame, lookback: int, thr: float, start, end, horizons=FWD_H,
                    gap: int = EVENT_GAP) -> pd.DataFrame:
    """Verilen akran grubunda geçmiş geride kalma olayları ve sonraki toparlanma (Brief §11, §31; MCP run_backtest
    strategy='catch_up'). Aynı yapı taşları R2 ile ortak: sector_lag (kendisi hariç akran ortancası) ve dedupe.

    panel : grubun kapanışları (sütun = hisse). 'end' sonrası kesilir, yani sonuç 'end' sonrasını hiç görmez.
    Olay  : 'lookback' günlük geride kalma ilk kez <= -thr olan gün (aynı hissede en az `gap` gün arayla), [start, end].
    rel_h : olay günü kapanışından h gün sonrasına kadar hisse getirisi - kendisi hariç akran ortancası getirisi.
            Penceresi 'end' sonrasına taşan olayda boş (NaN): henüz bilinmiyor.
    quake : Şubat 2023 deprem penceresi (R2 gibi istatistik dışında tutulur).
    """
    panel = panel.loc[:pd.Timestamp(end)]
    lag = sector_lag(panel, {"grup": list(panel.columns)}, lookback)
    quake = quake_flags_wide(panel.index)
    rows = []
    for t in panel.columns:
        cond = lag[t] <= -thr
        first = cond & ~cond.shift(1, fill_value=False)
        idx = first.index[dedupe(first, gap).to_numpy()]
        idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        for d in idx:
            k = panel.index.get_loc(d)
            row = dict(date=d, ticker=t, lag=float(lag.at[d, t]), quake=bool(quake.loc[d]))
            for h in horizons:
                if k + h >= len(panel):
                    row[f"rel_{h}"] = np.nan
                    continue
                r = panel.iloc[k + h] / panel.iloc[k] - 1
                row[f"rel_{h}"] = float(r[t] - r.drop(t).median())
            rows.append(row)
    cols = ["date", "ticker", "lag", "quake"] + [f"rel_{h}" for h in horizons]
    return pd.DataFrame(rows, columns=cols).sort_values(["date", "ticker"]).reset_index(drop=True)


def catchup_summary(ev: pd.DataFrame, horizons=FWD_H, cost_round_trip: float = 0.0) -> list:
    """Ufuk başına (sıralı liste): tamamlanmış olay sayısı, henüz kapanmamış pencere, isabet (fark > 0), tam kapanma
    (fark >= olay günündeki geride kalma), ortanca/ortalama toparlanma, maliyet sonrası ortalama."""
    out = []
    for h in horizons:
        s = ev[f"rel_{h}"].dropna()
        need = -ev.loc[s.index, "lag"]
        out.append(dict(horizon_days=h, n=int(len(s)), n_incomplete=int(ev[f"rel_{h}"].isna().sum()),
                        hit_rate=float((s > 0).mean()) if len(s) else None,
                        full_close_rate=float((s >= need).mean()) if len(s) else None,
                        median_rel=float(s.median()) if len(s) else None,
                        mean_rel=float(s.mean()) if len(s) else None,
                        mean_rel_after_cost=float(s.mean() - cost_round_trip) if len(s) else None))
    return out


def baseline() -> pd.DataFrame:
    """Uygun hisselerin olaysız herhangi bir günü: ortalama getiri (%). Deprem hariç."""
    c = _context()
    keep = ~quake_flags_wide(c["panel"].index).to_numpy()
    rows = {}
    for kind in ("fwd", "exc", "gap"):
        rows[kind] = {f"mean_{h}": c[kind][h].loc[keep, c["eligible"]].stack().dropna().mean() * 100
                      for h in FWD_H}
    return pd.DataFrame(rows).T


def summarize(ev: pd.DataFrame, by=("config", "side", "shock"), kind: str = "gap") -> pd.DataFrame:
    """Ortalama getiri (%) ve pozitif oran (%). Deprem penceresi hariç."""
    ev = ev[~ev["quake_window"]]
    rows = []
    for keys, g in ev.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["n"] = len(g)
        for h in FWD_H:
            s = g[f"{kind}_{h}"].dropna()
            row[f"mean_{h}"] = s.mean() * 100
            row[f"hit_{h}"] = (s > 0).mean() * 100
            if h == 10:
                row["t_10"] = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index(list(by))


if __name__ == "__main__":
    c = _context()
    ev = sector_events()
    pd.set_option("display.width", 200)
    print("grup boyutları:", {g: len(m) for g, m in c["groups"].items()})
    print("uygun hisse sayısı:", len(c["eligible"]), "| toplam olay:", len(ev),
          "| deprem penceresinde:", int(ev["quake_window"].sum()))
    ev.reset_index().to_parquet("reports/sector_events.parquet")
    print("\n--- baz çizgisi: uygun hisselerin olaysız herhangi bir günü (%) ---")
    print(baseline().round(2).to_string())
    for kind, title in (("gap", "akranlarına göre fark (gap)"), ("exc", "XU100'e göre fazla getiri (exc)"),
                        ("fwd", "ham getiri (fwd)")):
        s = summarize(ev, kind=kind)
        s.to_csv(f"reports/sector_summary_{kind}.csv")
        print(f"\n--- {title}: ortalama %, pozitif oran %, deprem hariç ---")
        print(s.round(2).to_string())
    summarize(ev, by=("config", "side", "peer_group"), kind="gap").to_csv("reports/sector_summary_by_group.csv")